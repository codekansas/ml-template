from dataclasses import dataclass
from typing import Tuple

import torch.nn.functional as F
import torchvision
from torch import Tensor
from torch.utils.data.dataset import Dataset

from ml.core.env import get_data_dir
from ml.core.registry import register_task
from ml.core.state import Phase, State
from ml.models.base import BaseModel
from ml.tasks.base import BaseTask, BaseTaskConfig


@dataclass
class CIFARDemoTaskConfig(BaseTaskConfig):
    pass


@register_task("cifar_demo", CIFARDemoTaskConfig)
class CIFARDemoTask(BaseTask[CIFARDemoTaskConfig]):
    def run_model(
        self,
        model: BaseModel,
        batch: Tuple[Tensor, Tensor],
        state: State,
    ) -> Tensor:
        image, _ = batch
        return model(image)

    def compute_loss(
        self,
        model: BaseModel,
        batch: Tuple[Tensor, Tensor],
        state: State,
        output: Tensor,
    ) -> Tensor:
        (_, classes), preds = batch, output
        return F.cross_entropy(preds, classes.flatten().long(), reduction="none")

    def get_dataset(self, phase: Phase) -> Dataset:
        return torchvision.datasets.CIFAR10(
            root=get_data_dir(),
            train=phase == "train",
            download=True,
        )
