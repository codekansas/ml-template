import math
from dataclasses import dataclass

from omegaconf import II, MISSING, OmegaConf

from ml.core.config import conf_field
from ml.core.registry import register_lr_scheduler
from ml.core.state import State
from ml.lr_schedulers.base import BaseLRScheduler, BaseLRSchedulerConfig


@dataclass
class CosineDecayLRSchedulerConfig(BaseLRSchedulerConfig):
    total_steps: int = conf_field(II("task.finished.max_steps"), help="Total number of steps to run")
    num_resets: int = conf_field(0, help="Number of times to reset learning")
    phase: int = conf_field(MISSING, help="Number of steps in a phase")
    ramp_up_percent: float = conf_field(0.05, help="Percent of phase to spend ramping up")
    ramp_up_steps: int = conf_field(MISSING, help="Number of steps to spend ramping up")
    eta_min: float = conf_field(0.01, help="Minimum learning rate scale")
    eta_max: float = conf_field(1.0, help="Maximum learning rate scale")
    min_decay: float = conf_field(1e-4, help="Minimum learning rate decay")

    @classmethod
    def resolve(cls, config: "CosineDecayLRSchedulerConfig") -> None:
        if OmegaConf.is_missing(config, "phase"):
            config.phase = int(config.total_steps / (config.num_resets + 1))
        if OmegaConf.is_missing(config, "ramp_up_steps"):
            assert 0 <= config.ramp_up_percent < 1
            config.ramp_up_steps = int(config.phase * config.ramp_up_percent)
        else:
            assert config.ramp_up_steps < config.phase
        super().resolve(config)


@register_lr_scheduler("cosine_decay", CosineDecayLRSchedulerConfig)
class CosineDecayLRScheduler(BaseLRScheduler[CosineDecayLRSchedulerConfig]):
    def get_lr_scale(self, state: State) -> float:
        phase, total, ramp_up = self.config.phase, self.config.total_steps, self.config.ramp_up_steps
        eta_min, eta_max, min_decay = self.config.eta_min, self.config.eta_max, self.config.min_decay
        decay = (1.0 - min_decay) * (total - state.num_steps) / total + min_decay
        phase_steps = state.num_steps % (phase + ramp_up)
        if phase_steps < ramp_up:
            return max(phase_steps / ramp_up, eta_min) * decay
        sigma = (phase_steps - ramp_up) / phase
        lr_scale_no_decay = eta_min + (eta_max - eta_min) * (1 + math.cos(math.pi * sigma)) / 2
        return lr_scale_no_decay * decay
