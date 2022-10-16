from __future__ import annotations

import atexit
import logging
import multiprocessing as mp
import os
import re
import shutil
import subprocess as sp
from dataclasses import dataclass
from typing import Dict, Iterable, TypeVar

from torch.optim.optimizer import Optimizer

from ml.core.config import conf_field
from ml.core.state import State
from ml.core.types import Batch
from ml.lr_schedulers.base import SchedulerAdapter
from ml.models.base import BaseModel
from ml.tasks.base import BaseTask
from ml.trainers.base import BaseTrainer, BaseTrainerConfig

logger = logging.getLogger(__name__)


@dataclass
class GPUStatsConfig(BaseTrainerConfig):
    ping_interval: int = conf_field(1, help="How often to check stats (in seconds)")


ConfigT = TypeVar("ConfigT", bound=GPUStatsConfig)

NUMBER_REGEX = re.compile(r"[\d\.]+")


@dataclass(frozen=True)
class GPUStats:
    index: int
    memory_used: float
    temperature: float
    gpu_utilization: float


def parse_number(s: str) -> float:
    match = NUMBER_REGEX.search(s)
    if match is None:
        raise ValueError(s)
    return float(match.group())


def parse_gpu_stats(row: str) -> GPUStats:
    cols = row.split(",")
    index = int(cols[0].strip())
    memory_total, memory_used, temperature, gpu_utilization = (parse_number(col) for col in cols[1:])

    return GPUStats(
        index=index,
        memory_used=100 * memory_used / memory_total,
        temperature=temperature,
        gpu_utilization=gpu_utilization,
    )


def gen_gpu_stats(loop_secs: int = 5) -> Iterable[GPUStats]:
    fields = ",".join(["index", "memory.total", "memory.used", "temperature.gpu", "utilization.gpu"])
    command = f"nvidia-smi --query-gpu={fields} --format=csv --loop={loop_secs}"
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible_device_ids = None if visible_devices is None else {int(i.strip()) for i in visible_devices.split(",")}
    try:
        with sp.Popen(command.split(), stdout=sp.PIPE, universal_newlines=True) as proc:
            stdout = proc.stdout
            assert stdout is not None
            rows = iter(stdout.readline, "")
            for row in rows:
                try:
                    stats = parse_gpu_stats(row)
                except ValueError:
                    continue
                if visible_device_ids is None or stats.index in visible_device_ids:
                    yield stats
    except sp.CalledProcessError:
        logger.exception("Caught exception while trying to query `nvidia-smi`")


def worker(config: ConfigT, queue: "mp.Queue[GPUStats]") -> None:
    for gpu_stat in gen_gpu_stats(config.ping_interval):
        queue.put(gpu_stat)


class GPUStatsMixin(BaseTrainer[ConfigT]):
    """Defines a trainer mixin for getting gradient statistics."""

    def __init__(self, config: ConfigT) -> None:
        super().__init__(config)

        self.gpu_stats: Dict[int, GPUStats] = {}
        self.queue: "mp.Queue[GPUStats]" | None = None

        if shutil.which("nvidia-smi") is not None:
            self.queue = mp.Queue()
            proc = mp.Process(target=worker, args=(config, self.queue), daemon=True)
            proc.start()

            atexit.register(proc.kill)

    def on_step_start(
        self,
        state: State,
        train_batch: Batch,
        task: BaseTask,
        model: BaseModel,
        optim: Optimizer,
        lr_sched: SchedulerAdapter,
    ) -> None:
        super().on_step_start(state, train_batch, task, model, optim, lr_sched)

        while self.queue is not None and not self.queue.empty():
            gpu_stat: GPUStats = self.queue.get()
            self.gpu_stats[gpu_stat.index] = gpu_stat
        for gpu_stat in self.gpu_stats.values():
            self.logger.log_scalar(f"gpu/{gpu_stat.index}/mem_used", gpu_stat.memory_used, namespace="trainer")
            self.logger.log_scalar(f"gpu/{gpu_stat.index}/temp", gpu_stat.temperature, namespace="trainer")
            self.logger.log_scalar(f"gpu/{gpu_stat.index}/gpu_util", gpu_stat.gpu_utilization, namespace="trainer")
