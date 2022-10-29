from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, TypeVar

import torch
from torch import Tensor
from torch.optim import Optimizer

from ml.core.config import conf_field
from ml.trainers.base import BaseTrainer, BaseTrainerConfig


@dataclass
class FP16:
    enabled: bool = conf_field(True, help="If set, should FP16 training be enabled")
    init_scale: float = conf_field(2.0**16, help="Initial scaling factor")
    growth_factor: float = conf_field(2.0, help="Factor by which the scale is multiplied if no gradient NaNs occur")
    backoff_factor: float = conf_field(0.5, help="Factor by which the scale is multiplied if gradient NaNs occur")
    growth_interval: int = conf_field(2000, help="How often to grow the scale")


@dataclass
class MixedPrecisionTrainerConfig(BaseTrainerConfig):
    fp16: FP16 = FP16()


ConfigT = TypeVar("ConfigT", bound=MixedPrecisionTrainerConfig)


class MixedPrecisionTrainerMixin(BaseTrainer[ConfigT]):
    """Defines a trainer mixin for doing FP16 scaling."""

    def __init__(self, config: ConfigT) -> None:
        super().__init__(config)

        self.grad_scaler: torch.cuda.amp.GradScaler | None
        if torch.cuda.is_available():
            self.grad_scaler = torch.cuda.amp.GradScaler(
                init_scale=self.config.fp16.init_scale,
                growth_factor=self.config.fp16.growth_factor,
                backoff_factor=self.config.fp16.backoff_factor,
                growth_interval=self.config.fp16.growth_interval,
                enabled=self.config.fp16.enabled,
            )
        else:
            self.grad_scaler = None

    def scale_mixed_precision(self, tensor: Tensor) -> Tensor:
        if self.grad_scaler is not None:
            return self.grad_scaler.scale(tensor)
        return tensor

    def unscale_mixed_precision(self, optim: Optimizer) -> None:
        if self.grad_scaler is not None:
            self.grad_scaler.unscale_(optim)

    def step_optimizer(self, optim: Optimizer) -> None:
        if self.grad_scaler is None:
            optim.step()
        else:
            self.grad_scaler.step(optim)
            self.grad_scaler.update()

    def log_mp_scale(self) -> None:
        if (scaler := self.grad_scaler) is not None:
            if (scale := getattr(scaler, "_scale", None)) is not None:
                self.logger.log_scalar("fp16_scale", scale)

    def load_state_dict(self, ckpt: Dict[str, Any]) -> None:
        if self.grad_scaler is not None:
            self.grad_scaler.load_state_dict(ckpt["grad_scaler"])

        super().load_state_dict(ckpt)

    def update_state_dict(self, ckpt: Dict[str, Any]) -> None:
        if self.grad_scaler is not None:
            assert "grad_scaler" not in ckpt

            ckpt["grad_scaler"] = self.grad_scaler.state_dict()

        super().update_state_dict(ckpt)
