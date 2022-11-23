from dataclasses import dataclass
from typing import Tuple, cast

import torch.nn.functional as F
import torchvision
from torch import Tensor
from torch.utils.data.dataset import Dataset

import ml.api as M
from ml.api import register_task


@dataclass
class CIFARDemoTaskConfig(M.BaseTaskConfig):
    pass


@register_task("cifar_demo", CIFARDemoTaskConfig)
class CIFARDemoTask(M.BaseTask[CIFARDemoTaskConfig]):
    def __init__(self, config: CIFARDemoTaskConfig) -> None:
        super().__init__(config)

        # Gets the class names for each index.
        class_to_idx = cast(torchvision.datasets.CIFAR10, self.get_dataset("test")).class_to_idx
        self.idx_to_class = {i: name for name, i in class_to_idx.items()}

    def get_label(self, true_class: int | float, pred_class: int | float) -> str:
        return "\n".join(
            [
                f"True: {self.idx_to_class.get(true_class, 'MISSING')}",
                f"Predicted: {self.idx_to_class.get(pred_class, 'MISSING')}",
            ]
        )

    def run_model(
        self,
        model: M.BaseModel,
        batch: Tuple[Tensor, Tensor],
        state: M.State,
    ) -> Tensor:
        image, _ = batch
        return model(image)

    def compute_loss(
        self,
        model: M.BaseModel,
        batch: Tuple[Tensor, Tensor],
        state: M.State,
        output: Tensor,
    ) -> Tensor:
        (image, classes), preds = batch, output
        pred_classes = preds.argmax(dim=1, keepdim=True)

        # Passing in a callable function ensures that we don't compute the
        # metric unless it's going to be logged, for example, when the logger
        # is rate-limited.
        self.logger.log_scalar("accuracy", lambda: (classes == pred_classes).float().mean())

        # On validation and test steps, logs images to each image logger.
        if state.phase in ("valid", "test"):
            bsz = classes.shape[0]
            texts = [self.get_label(classes[i].item(), pred_classes[i].item()) for i in range(bsz)]
            self.logger.log_labeled_images("image", (image, texts))

        return F.cross_entropy(preds, classes.flatten().long(), reduction="none")

    def get_dataset(self, phase: M.Phase) -> Dataset:
        return torchvision.datasets.CIFAR10(
            root=M.get_data_dir(),
            train=phase == "train",
            download=True,
        )
