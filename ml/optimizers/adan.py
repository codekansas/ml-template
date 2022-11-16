from dataclasses import dataclass
from typing import Callable, Iterable, Tuple

import torch
from torch import nn
from torch.optim import Optimizer

from ml.core.config import conf_field
from ml.core.registry import register_optimizer
from ml.optimizers.base import BaseOptimizer, BaseOptimizerConfig


class Adan(Optimizer):
    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float, float] = (0.1, 0.1, 0.001),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        assert len(betas) == 3

        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss: float | None = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            lr = group["lr"]
            beta1, beta2, beta3 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                data, grad = p.data, p.grad.data
                assert not grad.is_sparse

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = grad.clone()
                    state["v"] = torch.zeros_like(grad)
                    state["n"] = grad**2

                step, m, v, n = state["step"], state["m"], state["v"], state["n"]

                zeroth_step = step == 0
                first_step = step == 1

                if not zeroth_step:
                    prev_grad = state["prev_grad"]
                    m.mul_(1 - beta1).add_(grad, alpha=beta1)
                    grad_diff = grad - prev_grad
                    if not first_step:
                        v.mul_(1 - beta2).add_(grad_diff, alpha=beta2)
                    else:
                        v.add_(grad_diff)
                    next_n = (grad + (1 - beta2) * grad_diff) ** 2
                    n.mul_(1 - beta3).add_(next_n, alpha=beta3)

                weighted_step_size = lr / (n + eps).sqrt()
                denom = 1 + weight_decay * lr

                data.addcmul_(weighted_step_size, (m + (1 - beta2) * v), value=-1.0).div_(denom)
                state["prev_grad"] = grad.clone()
                state["step"] += 1

        return loss


@dataclass
class AdanOptimizerConfig(BaseOptimizerConfig):
    lr: float = conf_field(1e-3, help="Learning rate")
    betas: Tuple[float, float, float] = conf_field((0.1, 0.1, 0.001), help="Beta coefficients")
    eps: float = conf_field(1e-4, help="Epsilon term")
    weight_decay: float = conf_field(1e-5, help="Weight decay regularization to use")


@register_optimizer("adan", AdanOptimizerConfig)
class AdanOptimizer(BaseOptimizer[AdanOptimizerConfig]):
    def get(self, model: nn.Module) -> Adan:
        return Adan(
            model.parameters(),
            lr=self.config.lr,
            betas=self.config.betas,
            eps=self.config.eps,
            weight_decay=self.config.weight_decay,
            **self.common_kwargs,
        )
