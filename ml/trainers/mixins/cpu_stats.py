import atexit
import logging
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from typing import Optional, TypeVar

import psutil
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
class CPUStatsConfig(BaseTrainerConfig):
    ping_interval: int = conf_field(1, help="How often to check stats (in seconds)")


CPUStatsConfigT = TypeVar("CPUStatsConfigT", bound=CPUStatsConfig)


@dataclass(frozen=True)
class CPUStats:
    cpu_percent: float
    mem_percent: float
    mem_rss: float
    mem_vms: float
    mem_shared: float
    mem_rss_total: float
    mem_vms_total: float
    max_child_cpu_percent: float
    max_child_mem_percent: float
    num_child_procs: int


def worker(config: CPUStatsConfigT, queue: "mp.Queue[CPUStats]", pid: int) -> None:
    proc = psutil.Process(pid)
    child_procs = {p.pid: p for p in proc.children(recursive=True)}
    while True:
        try:
            # Updates child processes, preserving the previous child process
            # object. Otherwise the CPU percentage will be zero.
            new_children = {p.pid: p for p in proc.children(recursive=True) if p.pid not in child_procs}
            child_procs = {k: v for k, v in child_procs.items() if k in new_children}
            child_procs.update(new_children)

            # Gets process memory info.
            mem_info = proc.memory_info()
            mem_rss_total = sum(p.memory_info().rss for p in child_procs.values()) + mem_info.rss
            mem_vms_total = sum(p.memory_info().vms for p in child_procs.values()) + mem_info.vms

            # Gets the CPU statistics.
            cpu_stats = CPUStats(
                cpu_percent=proc.cpu_percent(),
                mem_percent=proc.memory_percent(),
                mem_rss=mem_info.rss,
                mem_vms=mem_info.vms,
                mem_shared=getattr(mem_info, "shared", 0.0),
                mem_rss_total=mem_rss_total,
                mem_vms_total=mem_vms_total,
                max_child_cpu_percent=max(p.cpu_percent() for p in child_procs.values()) if child_procs else 0.0,
                max_child_mem_percent=max(p.memory_percent() for p in child_procs.values()) if child_procs else 0.0,
                num_child_procs=len(child_procs),
            )
            queue.put(cpu_stats)
        except psutil.NoSuchProcess:
            logger.info("No parent process; probably cleaning up")
        time.sleep(config.ping_interval)


class CPUStatsMixin(BaseTrainer[CPUStatsConfigT]):
    """Defines a trainer mixin for getting CPU statistics."""

    def __init__(self, config: CPUStatsConfigT) -> None:
        super().__init__(config)

        self._cpu_stats: Optional[CPUStats] = None
        self._cpu_stats_queue: "mp.Queue[CPUStats]" = mp.Queue()

        proc = mp.Process(target=worker, args=(config, self._cpu_stats_queue, os.getpid()), daemon=True)
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

        while self._cpu_stats_queue is not None and not self._cpu_stats_queue.empty():
            self._cpu_stats = self._cpu_stats_queue.get()
        if self._cpu_stats is not None:
            self.logger.log_scalar("cpu/percent", self._cpu_stats.cpu_percent, namespace="trainer")
            self.logger.log_scalar("cpu/max_child_percent", self._cpu_stats.max_child_cpu_percent, namespace="trainer")
            self.logger.log_scalar("mem/percent", self._cpu_stats.mem_percent, namespace="trainer")
            self.logger.log_scalar("mem/rss", self._cpu_stats.mem_rss, namespace="trainer")
            self.logger.log_scalar("mem/vms", self._cpu_stats.mem_vms, namespace="trainer")
            self.logger.log_scalar("mem/shared", self._cpu_stats.mem_shared, namespace="trainer")
            self.logger.log_scalar("mem/rss/total", self._cpu_stats.mem_rss_total, namespace="trainer")
            self.logger.log_scalar("mem/vms/total", self._cpu_stats.mem_vms_total, namespace="trainer")
            self.logger.log_scalar("mem/max_child_percent", self._cpu_stats.max_child_mem_percent, namespace="trainer")
            self.logger.log_scalar("child_procs", self._cpu_stats.num_child_procs, namespace="trainer")
