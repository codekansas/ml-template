"""Defines a Distributed Data Parallel trainer that launches Slurm jobs.

This is a light-weight wrapper around the PyTorch's built-in Distributed Data
Parallel class. Note that this only supports one GPU per task, and one task
per GPU.

Steps
-----

1. Stages the environment to a new working directory
2. Writes an `sbatch.sh` file
3. Schedules `sbatch.sh` file

This allows for repeatability by just scheduling the same `sbatch.sh` file.
"""


import logging
import os
import random
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from types import FrameType
from typing import Callable, List, Optional

from omegaconf import II, MISSING, OmegaConf
from torch import nn
from torch.optim import Optimizer

from ml.core.config import conf_field
from ml.core.env import get_distributed_backend
from ml.core.registry import Objects, register_trainer, stage_environment
from ml.core.state import State
from ml.lr_schedulers.base import SchedulerAdapter
from ml.models.base import BaseModel
from ml.scripts.train import train_main
from ml.tasks.base import BaseTask
from ml.trainers.vanilla import VanillaTrainer, VanillaTrainerConfig
from ml.utils.distributed import (
    get_master_addr,
    get_master_port,
    get_world_size,
    init_process_group,
    is_master,
    set_init_method,
    set_master_addr,
    set_rank,
    set_world_size,
)
from ml.utils.logging import configure_logging

logger = logging.getLogger(__name__)

SBATCH_TEMPLATE = """
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --requeue
#SBATCH --signal=USR1@90
#SBATCH --time={time_limit}
#SBATCH --comment='{comment}'
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node={tasks_per_node}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --gres={gres}
#SBATCH --gpu-bind={gpu_bind}
#SBATCH --output={output_path}
#SBATCH --error={error_path}
#SBATCH --open-mode=append
{extra_sbatch_lines}

export PYTHONPATH={pythonpath}
export MASTER_PORT={master_port}

# Set some debugging flags.
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1
export NCCL_DEBUG=1

echo "***"
echo "Job ID: ${{SLURM_JOBID}}"
echo "***"
echo ""

# Runs the training command.
srun \\
    --nodes={num_nodes} \\
    --ntasks-per-node={tasks_per_node} \\
    --cpus-per-task={cpus_per_task} \\
    --gres={gres} \\
    --gpu-bind={gpu_bind} \\
    python {stage_dir}/ml/trainers/slurm.py {config_path}

echo ""
""".strip()


def get_random_port() -> int:
    return (hash(time.time()) + random.randint(0, 100000)) % (65_535 - 10_000) + 10_000


@dataclass
class SlurmTrainerConfig(VanillaTrainerConfig):
    partition: str = conf_field(II("oc.env:SLURM_PARTITION,none"), help="Which partition to launch")
    time_limit: str = conf_field(II("oc.env:SLURM_TIME_LIMIT,3-00:00:00"), help="Time limit string")
    num_nodes: int = conf_field(MISSING, help="Total number of nodes to use")
    gpus_per_node: int = conf_field(II("oc.env:SLURM_GPUS_PER_NODE,8"), help="Number of GPUs per node")
    cpus_per_gpu: int = conf_field(II("oc.env:SLURM_CPUS_PER_GPU,1"), help="Number of CPUs per task")
    num_jobs: int = conf_field(1, help="Number of redundant jobs to launch")
    comment: Optional[str] = conf_field(None, help="An optional comment to add to the experiment")
    master_port: int = conf_field(get_random_port, help="The master port to use")


def ignore_signal(signum: int, _: FrameType | None) -> None:
    sig = signal.Signals(signum)
    logger.info("Ignoring signal %s", sig.name)


@register_trainer("slurm", SlurmTrainerConfig)
class SlurmTrainer(VanillaTrainer[SlurmTrainerConfig]):
    def get_task_model(self, task: BaseTask, model: BaseModel) -> nn.Module:
        task_model = super().get_task_model(task, model)
        if get_world_size() > 1:
            task_model = nn.parallel.DistributedDataParallel(task_model)
        return task_model

    def on_exit(
        self,
        sig: signal.Signals,
        state: State,
        task: BaseTask,
        model: BaseModel,
        optim: Optimizer,
        lr_scheduler: SchedulerAdapter,
    ) -> None:
        super().on_exit(sig, state, task, model, optim, lr_scheduler)

        if is_master():
            if "SLURM_JOB_ID" in os.environ:
                cmd = ["scontrol", "requeue", os.environ["SLURM_JOB_ID"]]
                logger.info("Running %s", " ".join(cmd))
                subprocess.check_call(cmd)
            else:
                logger.info("SLURM_JOB_ID environment variable not found; not requeueing")

    def set_signal_handler(self, handler: Callable[[int, FrameType | None], None]) -> None:
        signal.signal(signal.SIGUSR1, handler)
        signal.signal(signal.SIGTERM, ignore_signal)

    def launch(self) -> None:
        # Gets some configuration options.
        gpus_per_node = self.config.gpus_per_node
        tasks_per_node = gpus_per_node
        cpus_per_task = self.config.cpus_per_gpu

        # GRES and GPU Bind SBatch options.
        gres = f"gpu:{gpus_per_node}"
        gpu_bind = f"map_gpu:{','.join(str(i) for i in range(gpus_per_node))}"

        # Gets extra SBatch options.
        sbatch_lines: List[str] = []
        if "EMAIL" in os.environ:
            sbatch_lines += [f"--mail-user={os.environ['EMAIL']}", "--mail-type=ALL"]

        # Writes all Slurm stuff (including logs) to this folder.
        slurm_log_dir = self.exp_dir / "logs"
        slurm_log_dir.mkdir(exist_ok=True, parents=True)
        sbatch_path = self.exp_dir / "sbatch.sh"

        # Stages all files to a new directory.
        stage_dir = stage_environment()

        # Gets the python path with the new output directory.
        python_path_parts = [str(stage_dir)] + os.environ.get("PYTHONPATH", "").split(":")
        python_path = ":".join(p for p in python_path_parts if p)

        # Comment miscellaneous stuff here.
        comments: List[str] = []
        if self.config.comment is not None:
            comments += [self.config.comment]
        comments += [f"Log directory: {self.exp_dir}"]
        comments += [f"Code location: {stage_dir}"]

        # Saves the config that is used to launch the Slurm job.
        self.save_config()

        # Builds the SBatch file.
        sbatch_file = SBATCH_TEMPLATE.format(
            job_name=self.config.exp_name,
            partition=self.config.partition,
            time_limit=self.config.time_limit,
            comment="; ".join(comments),
            num_nodes=self.config.num_nodes,
            tasks_per_node=tasks_per_node,
            cpus_per_task=cpus_per_task,
            gres=gres,
            gpu_bind=gpu_bind,
            output_path=slurm_log_dir / "slurm_out.txt",
            error_path=slurm_log_dir / "slurm_err.%j.txt",
            extra_sbatch_lines="\n".join(f"#SBATCH {line}" for line in sbatch_lines),
            pythonpath=python_path,
            master_port=self.config.master_port,
            config_path=self.exp_dir / "config.yaml",
            lock_file_path=self.exp_dir / ".lock_running",
            stage_dir=stage_dir,
        )

        with open(sbatch_path, "w", encoding="utf-8") as f:
            f.write(sbatch_file)
        logger.info("Wrote sbatch file to %s", sbatch_path)

        # Call `sbatch` on the given file.
        all_run_ids: List[str] = []
        for _ in range(self.config.num_jobs):
            command = ["sbatch", str(sbatch_path)]
            if all_run_ids:
                command += ["--dependency", all_run_ids[-1]]
            proc = subprocess.Popen(  # pylint: disable=consider-using-with
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            assert proc is not None and proc.stdout is not None
            proc.wait()
            log_line = proc.stdout.read().decode("utf-8").strip()
            run_ids = re.findall(r"Submitted batch job (\d+)", log_line)
            assert len(run_ids) == 1, f"Unexpected log line: {log_line}"
            all_run_ids += [run_ids[0]]

        run_ids_str = "".join(f"\n - {run_id}" for run_id in all_run_ids)
        logger.info("Launched %d job(s):%s", len(all_run_ids), run_ids_str)

        self.add_lock_file("scheduled", exists_ok=False)


def slurm_main() -> None:
    args = sys.argv[1:]
    assert len(args) == 1, f"Unexpected arguments to `slurm_main`: {sys.argv}"

    # Loads the raw config.
    raw_config = OmegaConf.load(args[0])

    # Gets node list for current job.
    node_list = os.environ.get("SLURM_STEP_NODELIST")
    if node_list is None:
        node_list = os.environ.get("SLURM_JOB_NODELIST")
    assert node_list is not None, "`SLURM_JOB_NODELIST` environment variable not set"

    # Resolves the complete node list.
    hostnames = subprocess.check_output(["scontrol", "show", "hostnames", node_list])
    host = hostnames.split()[0].decode("utf-8")
    set_master_addr(host)

    # Resolves the rank and world size.
    node_id = int(os.environ["SLURM_NODEID"])
    local_id = int(os.environ["SLURM_LOCALID"])
    tasks_per_node = int(os.environ["SLURM_NTASKS_PER_NODE"])
    num_nodes = int(os.environ["SLURM_NNODES"])
    rank = node_id * tasks_per_node + local_id
    world_size = num_nodes * tasks_per_node
    set_rank(rank)
    set_world_size(world_size)

    # Sets the initialization method and configures per-rank logging.
    set_init_method(f"tcp://{get_master_addr()}:{get_master_port()}")
    configure_logging(rank=rank, world_size=world_size)
    init_process_group(backend=get_distributed_backend())

    objs = Objects.parse_raw_config(raw_config)  # type: ignore
    assert (trainer := objs.trainer) is not None
    if is_master():
        trainer.remove_lock_file("scheduled", missing_ok=True)
    train_main(objs)


if __name__ == "__main__":
    slurm_main()
