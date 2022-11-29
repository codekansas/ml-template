import math
from dataclasses import dataclass

from omegaconf import MISSING, OmegaConf

from ml.core.config import conf_field
from ml.core.registry import register_lr_scheduler
from ml.core.state import State
from ml.lr_schedulers.base import BaseLRScheduler, BaseLRSchedulerConfig


@dataclass
class CosineLRSchedulerConfig(BaseLRSchedulerConfig):
    phase: int = conf_field(1000, help="Number of steps in a phase")
    ramp_up_percent: float = conf_field(0.05, help="Percent of phase to spend ramping up")
    ramp_up_steps: int = conf_field(MISSING, help="Number of steps to spend ramping up")
    eta_min: float = conf_field(0.01, help="Minimum learning rate scale")
    eta_max: float = conf_field(1.0, help="Maximum learning rate scale")

    @classmethod
    def resolve(cls, config: "CosineLRSchedulerConfig") -> None:
        if OmegaConf.is_missing(config, "ramp_up_steps"):
            assert 0 <= config.ramp_up_percent < 1
            config.ramp_up_steps = int(config.phase * config.ramp_up_percent)
        else:
            assert config.ramp_up_steps < config.phase
        return super().resolve(config)


@register_lr_scheduler("cosine", CosineLRSchedulerConfig)
class CosineLRScheduler(BaseLRScheduler[CosineLRSchedulerConfig]):
    def get_lr_scale(self, state: State) -> float:
        phase, ramp_up = self.config.phase, self.config.ramp_up_steps
        eta_min, eta_max = self.config.eta_min, self.config.eta_max
        phase_steps = state.num_steps % (phase + ramp_up)
        if phase_steps < ramp_up:
            return (1.0 - eta_min) * (phase_steps / ramp_up) + eta_min
        sigma = (phase_steps - ramp_up) / phase
        return eta_min + (eta_max - eta_min) * (1 + math.cos(math.pi * sigma)) / 2
