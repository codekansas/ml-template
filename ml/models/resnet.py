from dataclasses import dataclass
from typing import Callable, Dict, Optional

from omegaconf import MISSING
from torch import Tensor, nn
from torchvision.models.resnet import (
    ResNet,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)

from ml.core.config import conf_field
from ml.core.registry import register_model
from ml.models.base import BaseModel, BaseModelConfig

MODELS: Dict[int, Callable[[bool], ResNet]] = {
    18: resnet18,
    34: resnet34,
    50: resnet50,
    101: resnet101,
    152: resnet152,
}


@dataclass
class ResNetModelConfig(BaseModelConfig):
    size: int = conf_field(MISSING, help="ResNet size to use")
    pretrained: bool = conf_field(True, help="Load pretrained model")
    num_classes: Optional[int] = conf_field(None, help="If set, adds an output head with this many classes")

    @classmethod
    def get_defaults(cls) -> Dict[str, "ResNetModelConfig"]:
        return {
            "resnet18": ResNetModelConfig(size=18),
            "resnet34": ResNetModelConfig(size=34),
            "resnet50": ResNetModelConfig(size=50),
            "resnet101": ResNetModelConfig(size=101),
            "resnet152": ResNetModelConfig(size=152),
        }


@register_model("resnet", ResNetModelConfig)
class ResNetModel(BaseModel[ResNetModelConfig]):
    def __init__(self, config: ResNetModelConfig) -> None:
        super().__init__(config)

        if config.size not in MODELS:
            raise KeyError(f"Invalid model size: {config.size} Choices are: {sorted(MODELS.keys())}")

        # ResNet model always has 1000 classes (since it was pretrained on
        # ImageNet). So if we want a different number of classes we have to
        # attach a new head.
        if config.num_classes is None or config.num_classes == 1000:
            self.model = MODELS[config.size](config.pretrained)
        else:
            self.model = nn.Sequential(
                MODELS[config.size](config.pretrained),
                nn.ReLU(),
                nn.Linear(1000, config.num_classes),
            )

    def forward(self, image: Tensor) -> Tensor:
        return self.model(image)
