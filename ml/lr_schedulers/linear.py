from dataclasses import dataclass
from typing import Dict

from omegaconf import II, MISSING, OmegaConf

from ml.core.config import conf_field
from ml.core.registry import register_lr_scheduler
from ml.core.state import State
from ml.lr_schedulers.base import BaseLRScheduler, BaseLRSchedulerConfig


@dataclass
class LinearLRSchedulerConfig(BaseLRSchedulerConfig):
    warmup_steps: int = conf_field(MISSING, help="Number of warmup steps")
    total_steps: int = conf_field(II("task.finished.max_steps"), help="Total number of steps to run")
    warmup_percent: float = conf_field(0.01, help="Percentage of total steps to use as warmup steps, if not specified")
    min_scale: float = conf_field(1e-4, help="Minimum learning rate scale")

    @classmethod
    def get_defaults(cls) -> Dict[str, "LinearLRSchedulerConfig"]:
        return {
            "linear_100k": LinearLRSchedulerConfig(total_steps=100_000),
            "linear_500k": LinearLRSchedulerConfig(total_steps=500_000),
        }

    @classmethod
    def resolve(cls, config: "LinearLRSchedulerConfig") -> None:
        if OmegaConf.is_missing(config, "warmup_steps"):
            config.warmup_steps = int(config.total_steps * config.warmup_percent)
        super().resolve(config)


@register_lr_scheduler("linear", LinearLRSchedulerConfig)
class LinearLRScheduler(BaseLRScheduler[LinearLRSchedulerConfig]):
    def get_lr_scale(self, state: State) -> float:
        if state.num_steps < self.config.warmup_steps:
            return max(self.config.min_scale, state.num_steps / self.config.warmup_steps)
        if state.num_steps < self.config.total_steps:
            return max(
                self.config.min_scale,
                (self.config.total_steps - state.num_steps) / (self.config.total_steps - self.config.warmup_steps),
            )
        return self.config.min_scale
