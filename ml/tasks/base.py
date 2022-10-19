# pylint: disable=too-many-public-methods

from __future__ import annotations

import functools
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, is_dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Mapping,
    Optional,
    Sequence,
    Sized,
    Tuple,
    TypeVar,
)

import numpy as np
import torch
from omegaconf import II, MISSING
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler

from ml.core.config import BaseConfig, BaseObjectWithPointers, conf_field
from ml.core.env import is_debugging
from ml.core.state import Phase, State, cast_phase
from ml.core.types import Batch, Loss, Output
from ml.loggers.base import MultiLogger
from ml.lr_schedulers.base import SchedulerAdapter
from ml.models.base import BaseModel
from ml.tasks.datasets.collate import CollateMode, collate
from ml.tasks.datasets.error_handling import (
    ErrorHandlingConfig,
    get_error_handling_dataset,
)
from ml.tasks.losses.reduce import cast_reduce_type, reduce
from ml.utils.random import set_random_seed

logger = logging.getLogger(__name__)


class CumulativeTimer:
    """Defines a simple timer to track an average value."""

    def __init__(self) -> None:
        self.steps = 0
        self.elapsed_time = 0.0

    @functools.cached_property
    def start_time(self) -> float:
        return time.time()

    def step(self, steps: int, cur_time: float) -> None:
        if steps != self.steps:
            self.steps = steps
            self.elapsed_time = cur_time - self.start_time

    @property
    def steps_per_second(self) -> float:
        return 0.0 if self.elapsed_time < 1e-4 else self.steps / self.elapsed_time

    @property
    def steps_per_hour(self) -> float:
        return self.steps_per_second * 60 * 60

    @property
    def seconds_per_step(self) -> float:
        return 0.0 if self.steps <= 0 else self.elapsed_time / self.steps

    @property
    def hours_per_step(self) -> float:
        return self.seconds_per_step / (60 * 60)


class IterationTimer:
    """Defines a simple timer to track consecutive values."""

    def __init__(self) -> None:
        self.iteration_time = 0.0
        self.last_time = time.time()

    def step(self, cur_time: float) -> None:
        self.iteration_time = cur_time - self.last_time
        self.last_time = cur_time

    @property
    def iter_seconds(self) -> float:
        return self.iteration_time

    @property
    def iter_hours(self) -> float:
        return self.iter_seconds / (60 * 60)


class StateTimer:
    """Defines a timer for all state information."""

    def __init__(self) -> None:
        self.epoch_timer = CumulativeTimer()
        self.step_timer = CumulativeTimer()
        self.sample_timer = CumulativeTimer()
        self.iter_timer = IterationTimer()

    def step(self, state: State) -> None:
        cur_time = time.time()
        self.epoch_timer.step(state.num_epochs, cur_time)
        self.step_timer.step(state.num_steps, cur_time)
        self.sample_timer.step(state.num_samples, cur_time)
        self.iter_timer.step(cur_time)

    def log_dict(self) -> Dict[str, int | float]:
        logs: Dict[str, int | float] = {}

        # Logs epoch statistics (only if at least one epoch seen).
        if self.epoch_timer.steps > 0:
            logs["epoch"] = self.epoch_timer.steps
            logs["hours/epoch"] = self.epoch_timer.hours_per_step

        # Logs step statistics.
        logs["steps"] = self.step_timer.steps
        logs["steps/second"] = self.step_timer.steps_per_second
        logs["steps/hour"] = self.step_timer.steps_per_hour

        # Logs sample statistics.
        logs["samples"] = self.sample_timer.steps
        logs["samples/second"] = self.sample_timer.steps_per_second
        logs["samples/hour"] = self.sample_timer.steps_per_hour

        logs["dt/iter"] = self.iter_timer.iter_seconds
        return logs


@dataclass
class DataLoaderConfig:
    batch_size: int = conf_field(MISSING, help="Size of each batch")
    shuffle: bool = conf_field(MISSING, help="Should the batches be shuffled on each iteration")
    num_workers: int = conf_field(MISSING, help="Number of workers for loading samples")
    pin_memory: bool = conf_field(MISSING, help="Should memory be pinned to it's GPU location")
    drop_last: bool = conf_field(MISSING, help="Should the last batch be dropped if not full")
    timeout: float = conf_field(0, help="How long to wait for a sample to be ready")
    prefetch_factor: int = conf_field(2, help="Number of items to pre-fetch on each worker")
    persistent_workers: bool = conf_field(False, help="Persiste worker processes between epochs")
    seed: int = conf_field(1337, help="Dataloader random seed")


DEFAULT_DATALOADER_CONFIGS: Dict[str, DataLoaderConfig] = {
    "train": DataLoaderConfig(
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    ),
    "valid": DataLoaderConfig(
        batch_size=II("task.dataloader.train.batch_size"),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        persistent_workers=False,
    ),
    "test": DataLoaderConfig(
        batch_size=II("task.dataloader.valid.batch_size"),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        persistent_workers=False,
    ),
}


@dataclass
class FinishTrainingConfig:
    max_epochs: Optional[int] = conf_field(None, help="Maximum number of epochs to run")
    max_steps: Optional[int] = conf_field(None, help="Maximum number of steps to run")
    max_samples: Optional[int] = conf_field(None, help="Maximum number of samples to run")


@dataclass
class LossConfig:
    reduce_type: str = conf_field("mean", help="Loss reduction type to use")


@dataclass
class BaseTaskConfig(BaseConfig):
    """Defines the base config for all tasks."""

    dataloader: Dict[str, DataLoaderConfig] = conf_field(lambda: DEFAULT_DATALOADER_CONFIGS)
    finished: FinishTrainingConfig = FinishTrainingConfig()
    error_handling: ErrorHandlingConfig = ErrorHandlingConfig()
    loss: LossConfig = LossConfig()


TaskConfigT = TypeVar("TaskConfigT", bound=BaseTaskConfig)


class BaseTask(nn.Module, BaseObjectWithPointers[TaskConfigT], Generic[TaskConfigT], ABC):
    """Defines the base task type."""

    def __init__(self, config: TaskConfigT) -> None:
        nn.Module.__init__(self)
        BaseObjectWithPointers.__init__(self, config)

        self.dataloader_configs: Dict[Phase, DataLoaderConfig] = {
            cast_phase(k): v for k, v in config.dataloader.items()
        }

        # This flag can be toggled to end training from anywhere in the task.
        self.__training_over_flag = False

        # Timers for iterations.
        self.train_timer = StateTimer()
        self.valid_timer = StateTimer()
        self.test_timer = StateTimer()

        # Used to log values.
        self.logger = MultiLogger(default_namespace="task")

        # Final loss reduce type.
        self.__final_loss_reduce_type = cast_reduce_type(self.config.loss.reduce_type)

    @abstractmethod
    def run_model(self, model: BaseModel, batch: Batch, state: State) -> Output:
        """Runs a single training step and returns the outputs.

        Args:
            model: The current nn.Module
            batch: The current batch
            state: The current trainer state

        Returns:
            The outputs from the model
        """

    @abstractmethod
    def compute_loss(self, model: BaseModel, batch: Batch, state: State, output: Output) -> Loss:
        """Computes the loss for a given output.

        If the loss is a tensor, it should have shape (B). If the loss is a
        dictionary of tensors, each tensor should have the same shape (B).

        Args:
            model: The current nn.Module
            batch: The current batch
            state: The current trainer state
            output: The model output from `run_model`

        Returns:
            The computed loss, as a tensor or dictionary of tensors
        """

    def get_single_loss(self, loss: Loss) -> Tuple[Tensor, List[str]]:
        """Combines the output losses to get a single loss with shape (N, B).

        Args:
            loss: The computed loss or losses, either a tensor or dictionary of
                tensors. If a dictionary, all loss tensors need to have the
                same shape.

        Returns:
            The single loss with shape (N), where N is the number of losses,
            and the loss names, a list of length N.
        """

        if isinstance(loss, Tensor):
            assert loss.ndim >= 1, "Loss must not be a scalar"
            return reduce(loss.unsqueeze(0).flatten(1), self.__final_loss_reduce_type, 1), ["loss"]
        assert isinstance(loss, dict), f"Loss should be a scalar or dictionary, not {type(loss)}"
        for key, loss_tensor in loss.items():
            assert loss_tensor.ndim >= 1, f"Loss {key} must not be a scalar"
        keys = list(sorted(loss.keys()))
        single_loss = torch.stack([loss[k] for k in keys], dim=0)
        single_loss = reduce(single_loss.flatten(1), self.__final_loss_reduce_type, 1)
        return single_loss, keys

    def log_loss_dict(self, loss: Mapping[str, int | float | Tensor], state: State) -> None:
        for k, v in loss.items():
            self.logger.log_scalar(k, v, namespace="loss")

        if state.phase == "train":
            self.train_timer.step(state)
            for k, v in self.train_timer.log_dict().items():
                self.logger.log_scalar(k, v, namespace="timers")
        elif state.phase == "valid":
            self.valid_timer.step(state)
            for k, v in self.valid_timer.log_dict().items():
                self.logger.log_scalar(k, v, namespace="timers")
        elif state.phase == "test":
            self.test_timer.step(state)
            for k, v in self.test_timer.log_dict().items():
                self.logger.log_scalar(k, v, namespace="timers")
        else:
            raise NotImplementedError(f"Unexpected phase: {state.phase}")

    def get_batch_size(self, batch: Batch) -> int | None:
        if isinstance(batch, (np.ndarray, Tensor)):
            return batch.shape[0]
        if is_dataclass(batch):
            for v in batch.__dict__.values():
                if bsz := self.get_batch_size(v):
                    return bsz
        if isinstance(batch, Mapping):
            for v in batch.values():
                if bsz := self.get_batch_size(v):
                    return bsz
        if isinstance(batch, Sequence):
            for i in batch:
                if bsz := self.get_batch_size(i):
                    return bsz
        return None

    def set_training_over(self) -> None:
        self.__training_over_flag = True

    def get_remaining_percent(self, state: State) -> float | None:
        remaining_percents: List[float] = []
        cfg = self.config.finished
        if cfg.max_epochs is not None:
            remaining_percents.append((cfg.max_epochs - state.num_epochs) / cfg.max_epochs)
        if cfg.max_steps is not None:
            remaining_percents.append((cfg.max_steps - state.num_steps) / cfg.max_steps)
        if cfg.max_samples is not None:
            remaining_percents.append((cfg.max_samples - state.num_samples) / cfg.max_samples)
        return None if len(remaining_percents) == 0 else min(remaining_percents)

    def is_training_over(self, state: State) -> bool:
        if self.__training_over_flag:
            return True
        remaining_percent = self.get_remaining_percent(state)
        if remaining_percent is None:
            return False
        self.logger.log_scalar("remaining", remaining_percent, namespace="timers")
        return remaining_percent <= 0.0

    def get_dataset(self, phase: Phase) -> Dataset:
        """Returns the dataset for a given phase.

        Args:
            phase: The dataset phase to get

        Raises:
            NotImplementedError: If this method is not overridden
        """

        raise NotImplementedError("`get_dataset` should be implemented by the task")

    def get_sampler(self, dataset: Dataset, cfg: DataLoaderConfig, phase: Phase) -> Sampler[int]:
        """Returns a dataset sampler to use instead of random sampling.

        The default behavior for a non-iterable dataset is to use a
        RandomSampler for all the elements from the dataset. The sampler
        should yield integer indices into the dataset.

        Args:
            dataset: The dataset to sample from
            cfg: The associated dataloader config
            phase: The dataset's phase

        Raises:
            NotImplementedError: If this method is not overridden
        """

        raise NotImplementedError("`get_sampler` should be implemented for the specific task")

    def get_batch_sampler(self, sampler: Sampler, cfg: DataLoaderConfig, phase: Phase) -> Sampler[List[int]]:
        """Returns a dataset batch sampler to use instead fo sequential sampling.

        The batch sampler should yield lists of integer indices, which
        are the samples that are passed to the dataset.

        Args:
            sampler: The underlying sampler
            cfg: The associated dataloader config
            phase: The dataset's phase

        Raises:
            NotImplementedError: If this method is not overridden
        """

        raise NotImplementedError("`get_sampler` should be implemented for the specific task")

    def get_dataloader(self, dataset: Dataset, phase: Phase) -> DataLoader:
        if phase not in self.dataloader_configs:
            raise KeyError(f"Missing {phase=} in dataloader configs")
        cfg = self.dataloader_configs[phase]

        debugging = is_debugging()
        if debugging:
            logger.warning("Parallel dataloaders disabled in debugging mode")

        # Wraps the dataset in an error-handling dataset.
        if self.config.error_handling.enabled:
            dataset = get_error_handling_dataset(dataset, self.config.error_handling)

        # Arguments shared by all dataloaders.
        common_kwargs = {
            "num_workers": 0 if debugging else cfg.num_workers,
            "collate_fn": self.collate_fn,
            "pin_memory": cfg.pin_memory,
            "timeout": 0 if debugging else cfg.timeout,
            "worker_init_fn": self.worker_init_fn,
            "multiprocessing_context": None,
            "generator": None,
            "prefetch_factor": 2 if debugging else cfg.prefetch_factor,
            "persistent_workers": False if debugging else cfg.persistent_workers,
        }

        try:
            sampler = self.get_sampler(dataset, cfg, phase)
        except NotImplementedError:
            return DataLoader(
                dataset=dataset,
                batch_size=cfg.batch_size,
                drop_last=cfg.drop_last,
                shuffle=cfg.shuffle if isinstance(dataset, Sized) else False,
                **common_kwargs,  # type: ignore
            )

        try:
            batch_sampler = self.get_batch_sampler(sampler, cfg, phase)
        except NotImplementedError:
            return DataLoader(
                dataset=dataset,
                sampler=sampler,
                batch_size=cfg.batch_size,
                drop_last=cfg.drop_last,
                **common_kwargs,  # type: ignore
            )

        return DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            **common_kwargs,  # type: ignore
        )

    @classmethod
    def worker_init_fn(cls, worker_id: int) -> None:
        set_random_seed(offset=worker_id)

    @classmethod
    def collate_fn(cls, items: List[Any], *, mode: CollateMode = "stack") -> Any | None:
        return collate(items, mode=mode)

    # -----
    # Hooks
    # -----

    def on_after_load_checkpoint(self, ckpt: Dict[str, Any]) -> None:
        pass

    def on_after_save_checkpoint(self, ckpt_path: Path) -> None:
        pass

    def on_before_forward_step(self, model: BaseModel, batch: Batch, state: State) -> None:
        pass

    def on_after_forward_step(self, model: BaseModel, batch: Batch, output: Output, state: State) -> None:
        pass

    def on_after_compute_loss(self, model: BaseModel, batch: Batch, output: Output, loss: Loss, state: State) -> None:
        pass

    def on_step_start(
        self,
        state: State,
        train_batch: Batch,
        model: BaseModel,
        optim: Optimizer,
        lr_sched: SchedulerAdapter,
    ) -> None:
        pass

    def on_step_end(
        self,
        state: State,
        train_batch: Batch,
        loss_dict: Dict[str, Tensor],
        model: BaseModel,
        optim: Optimizer,
        lr_sched: SchedulerAdapter,
    ) -> None:
        pass

    def on_epoch_start(
        self,
        state: State,
        model: BaseModel,
        optim: Optimizer,
        lr_sched: SchedulerAdapter,
    ) -> None:
        pass

    def on_epoch_end(
        self,
        state: State,
        model: BaseModel,
        optim: Optimizer,
        lr_sched: SchedulerAdapter,
    ) -> None:
        pass
