from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

from torch import nn
from torch.optim.adamw import AdamW

from ml.core.config import conf_field
from ml.core.registry import register_optimizer
from ml.optimizers.base import BaseOptimizer, BaseOptimizerConfig


def separate_weight_decayable_params(params: Iterable[nn.Parameter]) -> Iterable[Dict[str, Any]]:
    """Don't weight decay biases.

    This was something that `lucidrains` does.

    Args:
        params: The full parameter list

    Returns:
        The dictionary to pass to the optimizer
    """

    wd_params: List[nn.Parameter] = []
    no_wd_params: List[nn.Parameter] = []
    for param in params:
        param_list = no_wd_params if param.ndim < 2 else wd_params
        param_list.append(param)

    return [
        {"params": wd_params},
        {"params": no_wd_params, "weight_decay": 0.0},
    ]


@dataclass
class AdamWOptimizerConfig(BaseOptimizerConfig):
    lr: float = conf_field(1e-3, help="Learning rate")
    betas: Tuple[float, float] = conf_field((0.9, 0.999), help="Beta coefficients")
    eps: float = conf_field(1e-4, help="Epsilon term to add to the denominator for stability")
    weight_decay: float = conf_field(1e-5, help="Weight decay regularization to use")
    amsgrad: bool = conf_field(False, help="Whether to use the AMSGrad variant of the algorithm")
    weight_decay_bias: bool = conf_field(False, help="If set, apply weight decay to biases")

    @classmethod
    def get_defaults(cls) -> Dict[str, "AdamWOptimizerConfig"]:
        return {
            "gpt-3-small": AdamWOptimizerConfig(
                lr=6e-4,
                betas=(0.9, 0.95),
                eps=1e-8,
                weight_decay=0.1,
            ),
            "gpt-3-medium": AdamWOptimizerConfig(
                lr=3e-4,
                betas=(0.9, 0.95),
                eps=1e-8,
                weight_decay=0.1,
            ),
            "gpt-3-large": AdamWOptimizerConfig(
                lr=2.5e-4,
                betas=(0.9, 0.95),
                eps=1e-8,
                weight_decay=0.1,
            ),
            "roberta-base": AdamWOptimizerConfig(
                lr=6e-4,
                betas=(0.9, 0.98),
                eps=1e-6,
                weight_decay=0.01,
            ),
            "roberta-large": AdamWOptimizerConfig(
                lr=4e-4,
                betas=(0.9, 0.98),
                eps=1e-6,
                weight_decay=0.01,
            ),
        }


@register_optimizer("adamw", AdamWOptimizerConfig)
class AdamWOptimizer(BaseOptimizer[AdamWOptimizerConfig]):
    def get(self, model: nn.Module) -> AdamW:
        params = model.parameters()

        return AdamW(
            params if self.config.weight_decay_bias else separate_weight_decayable_params(params),
            lr=self.config.lr,
            betas=self.config.betas,
            eps=self.config.eps,
            weight_decay=self.config.weight_decay,
            amsgrad=self.config.amsgrad,
            **self.common_kwargs,
        )
