from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeVar

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from ml.core.config import conf_field
from ml.trainers.base import BaseTrainer, BaseTrainerConfig
from ml.trainers.mixins.mixed_precision import (
    MixedPrecisionTrainerConfig,
    MixedPrecisionTrainerMixin,
)


@dataclass
class GradientClipping:
    clip_grad_norm: Optional[float] = conf_field(None, help="What to clip the gradient norm to")
    norm_type: Any = conf_field(2, help="Type of norm to use")
    clip_grad_value: Optional[float] = conf_field(None, help="What to clip the gradient value to")
    clip_global_grad_norm: Optional[float] = conf_field(None, help="What to clip global gradient norm to")
    global_norm_type: Any = conf_field(2, help="Type of global norm to use")


@dataclass
class GradientClippingConfig(MixedPrecisionTrainerConfig, BaseTrainerConfig):
    grad_clipping: GradientClipping = GradientClipping()


GradientClippingConfigT = TypeVar("GradientClippingConfigT", bound=GradientClippingConfig)


def get_clip_grad_func(clip_value: float) -> Callable[[Tensor], Tensor]:
    def func(grad: Tensor) -> Tensor:
        return grad.clamp(-clip_value, clip_value)

    return func


def get_clip_norm_func(clip_value: float, norm_type: Any) -> Callable[[Tensor], Tensor]:
    def func(grad: Tensor) -> Tensor:
        grad_norm = torch.norm(grad, p=norm_type)
        return grad * (grad_norm.clamp_max(clip_value) / grad_norm)

    return func


class GradientClippingTrainerMixin(
    MixedPrecisionTrainerMixin[GradientClippingConfigT],
    BaseTrainer[GradientClippingConfigT],
):
    """Defines a trainer mixin for doing gradient clipping."""

    def maybe_add_grad_clipping(self, model: nn.Module) -> None:
        clip_value = self.config.grad_clipping.clip_grad_value
        clip_norm = self.config.grad_clipping.clip_grad_norm
        if clip_value is not None:
            for p in model.parameters():
                if p.requires_grad:
                    p.register_hook(get_clip_grad_func(clip_value))
        if clip_norm is not None:
            for p in model.parameters():
                if p.requires_grad:
                    p.register_hook(get_clip_norm_func(clip_norm, self.config.grad_clipping.norm_type))

    def clip_grads(self, model: nn.Module, optim: Optimizer) -> None:
        clip_norm = self.config.grad_clipping.clip_global_grad_norm
        if clip_norm is not None:
            self.unscale_mixed_precision(optim)
            total_norm = nn.utils.clip_grad.clip_grad_norm_(
                model.parameters(),
                max_norm=clip_norm,
                norm_type=self.config.grad_clipping.global_norm_type,
            )
            self.logger.log_scalar("total_norm", total_norm.item(), namespace="optim")
