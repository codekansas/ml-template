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
        (image, classes), preds = batch, output

        # Passing in a callable function ensures that we don't compute the
        # metric unless it's going to be logged, for example, when the logger
        # is rate-limited.
        self.logger.log_scalar("accuracy", lambda: (classes == preds.argmax(dim=1, keepdim=True)).float().mean())

        # On validation and test steps, logs images to each image logger.
        if state.phase in ("valid", "test"):
            self.logger.log_images("image", image)

        return F.cross_entropy(preds, classes.flatten().long(), reduction="none")

    def get_dataset(self, phase: Phase) -> Dataset:
        return torchvision.datasets.CIFAR10(
            root=get_data_dir(),
            train=phase == "train",
            download=True,
        )
