"""Defines a vanilla trainer which doesn't do any device or data manipulation.

This trainer expects the task to handle all the relevant movement of data and
models to their associated devices.

Summary table:

|         | device 1 - N |
|---------|--------------|
| data    | data[:]      |
| step    | model(x)     |
| loss    | E(x, o)      |
"""


import contextlib
import logging
import signal
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Callable, Dict, TypeVar

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from ml.core.config import conf_field
from ml.core.registry import register_trainer
from ml.core.state import State, set_phase
from ml.core.types import Batch, Loss
from ml.loggers.meter import MeterLogger, MeterLoggerConfig
from ml.lr_schedulers.base import BaseLRScheduler, SchedulerAdapter
from ml.models.base import BaseModel
from ml.optimizers.base import BaseOptimizer
from ml.tasks.base import BaseTask
from ml.trainers.base import BaseTrainer, BaseTrainerConfig
from ml.trainers.mixins.cpu_stats import CPUStatsConfig, CPUStatsMixin
from ml.trainers.mixins.device.base import InfinitePrefetcher
from ml.trainers.mixins.gpu_stats import GPUStatsConfig, GPUStatsMixin
from ml.trainers.mixins.grad_clipping import (
    GradientClippingConfig,
    GradientClippingTrainerMixin,
)
from ml.trainers.mixins.mixed_precision import (
    MixedPrecisionTrainerConfig,
    MixedPrecisionTrainerMixin,
)
from ml.trainers.mixins.profiler import ProfilerTrainerConfig, ProfilerTrainerMixin
from ml.utils.distributed import is_master
from ml.utils.timer import Timer

logger = logging.getLogger(__name__)


class TrainingFinishedException(Exception):
    pass


class TaskModel(nn.Module):
    def __init__(self, task: BaseTask, model: BaseModel) -> None:
        super().__init__()

        self.task = task
        self.model = model

    def forward(self, batch: Batch, state: State) -> Loss:
        self.task.on_before_forward_step(self.model, batch, state)
        output = self.task.run_model(self.model, batch, state)
        self.task.on_after_forward_step(self.model, batch, output, state)
        loss: Loss = self.task.compute_loss(self.model, batch, state, output)
        self.task.on_after_compute_loss(self.model, batch, output, loss, state)
        return loss


@dataclass
class VanillaTrainerConfig(
    ProfilerTrainerConfig,
    GradientClippingConfig,
    MixedPrecisionTrainerConfig,
    GPUStatsConfig,
    CPUStatsConfig,
    BaseTrainerConfig,
):
    set_to_none: bool = conf_field(True, help="Mode for clearing optimizer gradients")
    deterministic: bool = conf_field(False, help="If set, use determinstic algorithms")
    use_tf32: bool = conf_field(True, help="If set, use TensorFloat32")
    update_interval: int = conf_field(1, help="How often to update model parameters")
    device: str = conf_field("auto", help="The trainer device type being used")


VanillaTrainerConfigT = TypeVar("VanillaTrainerConfigT", bound=VanillaTrainerConfig)


@register_trainer("vanilla", VanillaTrainerConfig)
class VanillaTrainer(
    ProfilerTrainerMixin[VanillaTrainerConfigT],
    GradientClippingTrainerMixin[VanillaTrainerConfigT],
    MixedPrecisionTrainerMixin[VanillaTrainerConfigT],
    GPUStatsMixin[VanillaTrainerConfigT],
    CPUStatsMixin[VanillaTrainerConfigT],
    BaseTrainer[VanillaTrainerConfigT],
):
    def train_step(
        self,
        *,
        task_model: nn.Module,
        batch: Batch,
        state: State,
        task: BaseTask,
        model: BaseModel,
        optim: Optimizer,
        lr_sched: SchedulerAdapter,
    ) -> Dict[str, Tensor]:
        with self.step_context("change_mode"):
            task_model, state.phase = set_phase(task_model, "train")
        with self.step_context("forward"), self.autocast_context():
            loss = task_model(batch, state)
        with self.step_context("get_single_loss"):
            single_loss, loss_names = task.get_single_loss(loss)
        with self.step_context("backward"):
            self.scale_mixed_precision(single_loss.sum()).backward()
        with self.step_context("log_losses"):
            self.log_mp_scale()
            single_loss_detached = single_loss.detach()
            loss_dict = {name: single_loss_detached[i] for i, name in enumerate(loss_names)}
            task.log_loss_dict(loss_dict, state)
        if state.num_steps % self.config.update_interval == 0:
            with self.step_context("clip_grads"):
                self.clip_grads(model=task_model, optim=optim)
            with self.step_context("step"):
                self.step_optimizer(optim=optim)
                lr_sched.step(state)
                self.logger.log_scalar("lr_scale", lr_sched.lr_scale, namespace="optim")
            with self.step_context("zero_grads"):
                optim.zero_grad(set_to_none=self.config.set_to_none)
        with self.step_context("write_logs"):
            self.write_logs(task, model, state)
        with self.step_context("update_state"):
            state.num_steps += 1
            bsz = task.get_batch_size(batch)
            if bsz is not None:
                state.num_samples += bsz
        return loss_dict

    def val_step(
        self,
        *,
        task_model: nn.Module,
        batch: Batch,
        state: State,
        task: BaseTask,
        model: BaseModel,
    ) -> None:
        with torch.no_grad():
            with self.step_context("change_mode"):
                task_model, state.phase = set_phase(task_model, "valid")
            with self.step_context("forward"), self.autocast_context():
                loss = task_model(batch, state)
            with self.step_context("get_single_loss"):
                single_loss, loss_names = task.get_single_loss(loss)
            with self.step_context("log_losses"):
                single_loss_detached = single_loss.detach()
                loss_dict = {name: single_loss_detached[i] for i, name in enumerate(loss_names)}
                task.log_loss_dict(loss_dict, state)
            with self.step_context("write_logs"):
                self.write_logs(task, model, state)
            with self.step_context("update_state"):
                state.num_valid_steps += 1

    def test_step(
        self,
        *,
        task_model: nn.Module,
        batch: Batch,
        state: State,
        task: BaseTask,
        model: BaseModel,
    ) -> None:
        with torch.no_grad():
            with self.step_context("change_mode"):
                task_model, state.phase = set_phase(task_model, "test")
            with self.step_context("forward"), self.autocast_context():
                loss = task_model(batch, state)
            with self.step_context("get_single_loss"):
                single_loss, loss_names = task.get_single_loss(loss)
            with self.step_context("log_losses"):
                single_loss_detached = single_loss.detach()
                loss_dict = {name: single_loss_detached[i] for i, name in enumerate(loss_names)}
                task.log_loss_dict(loss_dict, state)
            with self.step_context("write_logs"):
                self.write_logs(task, model, state)
            with self.step_context("update_state"):
                state.num_test_steps += 1

    def get_task_model(self, task: BaseTask, model: BaseModel) -> nn.Module:
        device, dtype = self._device.get_device(), self._weight_precision
        model.init(device, dtype)
        task.to(device, dtype, non_blocking=True)
        return TaskModel(task=task, model=model)

    def on_exit(
        self,
        sig: signal.Signals,
        state: State,
        task: BaseTask,
        model: BaseModel,
        optim: Optimizer,
        lr_scheduler: SchedulerAdapter,
    ) -> None:
        logger.info("Handling interrupt %s", sig.name)
        self.save_checkpoint(state, task, model, optim, lr_scheduler)
        logger.info("Removing lock file")
        if is_master():
            self.remove_lock_file("running", missing_ok=True)

    def set_signal_handler(self, handler: Callable[[int, FrameType | None], None]) -> None:
        pass

    def train(self, model: BaseModel, task: BaseTask, optimizer: BaseOptimizer, lr_scheduler: BaseLRScheduler) -> None:
        """Runs the GPU-based training loop.

        Args:
            model: The current model
            task: The current task
            optimizer: The current optimizer
            lr_scheduler: The current learning rate scheduler
        """

        if is_master():
            self.add_lock_file("running", exists_ok=True)

        # Sets up environment.
        if self.config.deterministic:
            torch.use_deterministic_algorithms(True)
        if self.config.use_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True

        # Saves the config at the start of training.
        if is_master():
            self.save_config()

        # Builds a mega-model.
        task_model: nn.Module = self.get_task_model(task, model)
        self.maybe_add_grad_clipping(task_model)

        # Gets the optimizer and learning rate scheduler.
        optim = optimizer.get(model)
        lr_sched = lr_scheduler.get(optim)

        # Loads an existing checkpoint, if one exists.
        if (ckpt_path := self.get_ckpt_path()).exists():
            state = self.load_checkpoint(ckpt_path, task, model, optim, lr_sched)
        else:
            state = State.init_state()

        def on_exit(signum: int, _: FrameType | None) -> None:
            sig = signal.Signals(signum)
            self.on_exit(sig, state, task, model, optim, lr_sched)

        self.set_signal_handler(on_exit)

        def on_finish_training() -> None:
            self.save_checkpoint(state, task, model, optim, lr_sched)
            raise TrainingFinishedException

        # Handle user-defined interrupts.
        signal.signal(signal.SIGUSR1, on_exit)

        # Gets the datasets.
        train_ds = task.get_dataset("train")
        valid_ds = task.get_dataset("valid")

        # Gets the dataloaders.
        train_dl = task.get_dataloader(train_ds, "train")
        valid_dl = task.get_dataloader(valid_ds, "valid")

        # Gets the prefetchers.
        train_pf = self._device.get_prefetcher(train_dl)
        valid_pf = self._device.get_prefetcher(valid_dl)
        valid_pf_infinite = InfinitePrefetcher(valid_pf)

        try:
            with contextlib.ExitStack() as ctx:
                profile = self.get_profile()
                if profile is not None:
                    ctx.enter_context(profile)

                if (num_init_valid_steps := self.config.validation.num_init_valid_steps) is not None:
                    for _ in range(num_init_valid_steps):
                        self.val_step(
                            task_model=task_model,
                            batch=next(valid_pf_infinite),
                            state=state,
                            task=task,
                            model=model,
                        )

                while True:
                    with self.step_context("on_epoch_start"):
                        self.on_epoch_start(state, task, model, optim, lr_sched)

                    for train_batch in train_pf:
                        self.logger.log_scalar("num_queued_samples", train_pf.num_queued_samples, namespace="trainer")
                        self.logger.log_scalar("dt/get_batch", train_pf.get_batch_time, namespace="timers")

                        if task.is_training_over(state):
                            on_finish_training()

                        with self.step_context("on_step_start"):
                            self.on_step_start(state, train_batch, task, model, optim, lr_sched)

                        loss_dict = self.train_step(
                            task_model=task_model,
                            batch=train_batch,
                            state=state,
                            task=task,
                            model=model,
                            optim=optim,
                            lr_sched=lr_sched,
                        )

                        valid_every_n_steps = self.config.validation.valid_every_n_steps
                        if valid_every_n_steps is not None and state.num_steps % valid_every_n_steps == 0:
                            self.val_step(
                                task_model=task_model,
                                batch=next(valid_pf_infinite),
                                state=state,
                                task=task,
                                model=model,
                            )

                        if self.should_checkpoint(state):
                            self.save_checkpoint(state, task, model, optim, lr_sched)

                        if profile is not None:
                            profile.step()

                        with self.step_context("on_step_end"):
                            self.on_step_end(state, train_batch, loss_dict, task, model, optim, lr_sched)

                    with self.step_context("on_epoch_end"):
                        self.on_epoch_end(state, task, model, optim, lr_sched)

                    state.num_epochs += 1

        except TrainingFinishedException:
            logger.info("Finished training for %s", self.exp_dir / "config.yaml")

        except Exception:
            logger.exception("Caught exception during training loop")

        finally:
            if is_master():
                self.remove_lock_file("running", missing_ok=True)
            logger.info("Exiting training job for %s", self.exp_dir / "config.yaml")

    def evaluate(self, model: BaseModel, task: BaseTask) -> None:
        """Runs the GPU-based evaluation loop.

        Args:
            model: The current model
            task: The current task
        """

        # Saves the config at the start of evaluation.
        self.save_config()

        # Meter logger keeps track of value statistics.
        meter_logger = MeterLogger(MeterLoggerConfig())
        self.add_logger(meter_logger)

        # Gets the dataset, dataloader and prefetcher.
        test_ds = task.get_dataset("test")
        test_dl = task.get_dataloader(test_ds, "test")
        test_pf = self._device.get_prefetcher(test_dl)

        def load_checkpoint(ckpt_path: Path) -> State:
            with Timer("loading checkpoint"):
                ckpt = torch.load(ckpt_path)
                model.load_state_dict(ckpt["model"])
                state = ckpt["state"]
            return state

        if (ckpt_path := self.get_ckpt_path()).exists():
            state = load_checkpoint(ckpt_path)
        else:
            logger.warning("Missing checkpoint to evaluate; using uninitialized model")
            state = State.init_state()

        # Builds a mega-model.
        task_model: nn.Module = self.get_task_model(task, model)

        for test_batch in test_pf:
            self.test_step(
                task_model=task_model,
                batch=test_batch,
                state=state,
                task=task,
                model=model,
            )

        # Finally, saves meter logging results.
        self.logger.log_config(self.raw_config, meter_logger.get_value_dict())

    def launch(self) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} doesn't support multiprocess training")
