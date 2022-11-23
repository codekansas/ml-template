from abc import ABC
from types import TracebackType
from typing import ContextManager, Literal, Optional, Type, TypeVar

from ml.trainers.base import BaseTrainer, BaseTrainerConfig

StepType = Literal[
    "backward",
    "change_mode",
    "clip_grads",
    "forward",
    "get_single_loss",
    "log_losses",
    "on_epoch_end",
    "on_epoch_start",
    "on_step_end",
    "on_step_start",
    "step",
    "update_state",
    "write_logs",
    "zero_grads",
]


BaseTrainerConfigT = TypeVar("BaseTrainerConfigT", bound=BaseTrainerConfig)


class StepContext(ContextManager):
    """Context manager to get the current step type."""

    CURRENT_STEP: Optional[StepType] = None

    def __init__(self, step: StepType) -> None:
        self.step = step

    def __enter__(self) -> None:
        StepContext.CURRENT_STEP = self.step

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        StepContext.CURRENT_STEP = None


class StepContextMixin(BaseTrainer[BaseTrainerConfigT], ABC):
    def step_context(self, step: StepType) -> ContextManager:
        return StepContext(step)
