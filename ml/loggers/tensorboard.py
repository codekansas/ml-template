import atexit
import datetime
import functools
import logging
import os
import re
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import torch
from omegaconf import MISSING, OmegaConf
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from ml.core.config import conf_field
from ml.core.env import get_exp_name
from ml.core.registry import register_logger
from ml.core.state import Phase, State
from ml.loggers.base import BaseLogger, BaseLoggerConfig
from ml.loggers.multi import TARGET_FPS
from ml.utils.distributed import is_distributed, is_master
from ml.utils.networking import get_unused_port

logger = logging.getLogger(__name__)

WRITE_PROC_TEXT_EVERY_N_SECONDS = 60 * 2


def make_bold(strs: List[str]) -> str:
    strs = [s.strip() for s in strs]
    max_len = max(len(s) for s in strs)
    return "\n".join(["-" * max_len] + strs + ["-" * max_len])


@dataclass
class TensorboardLoggerConfig(BaseLoggerConfig):
    flush_seconds: float = conf_field(10, help="How often to flush logs")
    log_id: str = conf_field(MISSING, help="Unique log ID")
    start_in_subprocess: bool = conf_field(True, help="Start TensorBoard subprocess")

    @classmethod
    def resolve(cls, config: "TensorboardLoggerConfig") -> None:
        if OmegaConf.is_missing(config, "log_id"):
            config.log_id = datetime.datetime.now().strftime("%H-%M-%S")
        super().resolve(config)


@register_logger("tensorboard", TensorboardLoggerConfig)
class TensorboardLogger(BaseLogger[TensorboardLoggerConfig]):
    def __init__(self, config: TensorboardLoggerConfig) -> None:
        super().__init__(config)

        self.scalars: Dict[Phase, Dict[str, Callable[[], int | float | Tensor]]] = defaultdict(dict)
        self.strings: Dict[Phase, Dict[str, Callable[[], str]]] = defaultdict(dict)
        self.images: Dict[Phase, Dict[str, Callable[[], Tensor]]] = defaultdict(dict)
        self.videos: Dict[Phase, Dict[str, Callable[[], Tensor]]] = defaultdict(dict)
        self.histograms: Dict[Phase, Dict[str, Callable[[], Tensor]]] = defaultdict(dict)
        self.point_clouds: Dict[Phase, Dict[str, Callable[[], Tensor]]] = defaultdict(dict)

        self.line_str: str | None = None
        self.last_tensorboard_write_time = time.time()

    def initialize(self, log_directory: Path) -> None:
        super().initialize(log_directory)

        if "TENSORBOARD_PORT" in os.environ:
            port, use_localhost = int(os.environ["TENSORBOARD_PORT"]), True
        else:
            port, use_localhost = get_unused_port(), False

        def make_localhost(s: str) -> str:
            if use_localhost:
                s = re.sub(rf"://(.+?):{port}", f"://localhost:{port}", s)
            return s

        if is_master():
            if not self.config.start_in_subprocess or is_distributed():
                tensorboard_command_strs = [
                    "tensorboard serve \\",
                    f"  --logdir {self.tensorboard_log_directory} \\",
                    "  --bind_all \\",
                    f"  --port {port} \\",
                    "  --reload_interval 15",
                ]
                logger.info("Tensorboard command:\n%s", make_localhost(make_bold(tensorboard_command_strs)))

            else:
                command: List[str] = [
                    "tensorboard",
                    "serve",
                    "--logdir",
                    str(self.tensorboard_log_directory),
                    "--bind_all",
                    "--port",
                    str(port),
                    "--reload_interval",
                    "15",
                ]
                logger.info("Tensorboard command: %s", " ".join(command))

                proc = subprocess.Popen(  # pylint: disable=consider-using-with
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )

                # Gets the output line that shows the running address.
                assert proc is not None and proc.stdout is not None
                for line in proc.stdout:
                    line_str = line.decode("utf-8")
                    if line_str.startswith("TensorBoard"):
                        self.line_str = make_localhost(line_str)
                        logger.info("Running TensorBoard process:\n%s", make_bold([self.line_str]))
                        break

                # Close the process when the program terminates.
                atexit.register(proc.kill)

    @property
    def tensorboard_log_directory(self) -> Path:
        return self.log_directory / "tensorboard" / self.config.log_id

    @functools.cached_property
    def train_writer(self) -> SummaryWriter:
        return SummaryWriter(
            self.tensorboard_log_directory / "train",
            flush_secs=self.config.flush_seconds,
        )

    @functools.cached_property
    def valid_writer(self) -> SummaryWriter:
        return SummaryWriter(
            self.tensorboard_log_directory / "valid",
            flush_secs=self.config.flush_seconds,
        )

    @functools.cached_property
    def test_writer(self) -> SummaryWriter:
        return SummaryWriter(
            self.tensorboard_log_directory / "test",
            flush_secs=self.config.flush_seconds,
        )

    def get_writer(self, phase: Phase) -> SummaryWriter:
        if phase == "train":
            return self.train_writer
        if phase == "valid":
            return self.valid_writer
        if phase == "test":
            return self.test_writer
        raise NotImplementedError(f"Unexpected phase: {phase}")

    def log_scalar(self, key: str, value: Callable[[], int | float | Tensor], state: State, namespace: str) -> None:
        self.scalars[state.phase][f"{namespace}/{key}"] = value

    def log_string(self, key: str, value: Callable[[], str], state: State, namespace: str) -> None:
        self.strings[state.phase][f"{namespace}/{key}"] = value

    def log_image(self, key: str, value: Callable[[], Tensor], state: State, namespace: str) -> None:
        self.images[state.phase][f"{namespace}/{key}"] = value

    def log_video(self, key: str, value: Callable[[], Tensor], state: State, namespace: str) -> None:
        self.videos[state.phase][f"{namespace}/{key}"] = value

    def log_histogram(self, key: str, value: Callable[[], Tensor], state: State, namespace: str) -> None:
        self.histograms[state.phase][f"{namespace}/{key}"] = value

    def log_point_cloud(self, key: str, value: Callable[[], Tensor], state: State, namespace: str) -> None:
        self.point_clouds[state.phase][f"{namespace}/{key}"] = value

    def log_config(self, config: Dict[str, int | float | str | bool], metrics: Dict[str, int | float]) -> None:
        self.test_writer.add_hparams(config, metrics, run_name=get_exp_name())

    def write(self, state: State) -> None:
        if self.line_str is not None:
            cur_time = time.time()
            if cur_time - self.last_tensorboard_write_time > WRITE_PROC_TEXT_EVERY_N_SECONDS:
                logger.info("Running TensorBoard process:\n%s", make_bold([self.line_str]))
                self.last_tensorboard_write_time = cur_time
        writer = self.get_writer(state.phase)
        for scalar_key, scalar_value in self.scalars[state.phase].items():
            writer.add_scalar(scalar_key, scalar_value(), global_step=state.num_steps)
        for string_key, string_value in self.strings[state.phase].items():
            writer.add_text(string_key, string_value(), global_step=state.num_steps)
        for image_key, image_value in self.images[state.phase].items():
            writer.add_image(image_key, image_value(), global_step=state.num_steps)
        for video_key, video_value in self.videos[state.phase].items():
            writer.add_video(video_key, video_value().unsqueeze(0), global_step=state.num_steps, fps=TARGET_FPS)
        for hist_key, hist_value in self.histograms[state.phase].items():
            writer.add_histogram(hist_key, hist_value(), global_step=state.num_steps)
        for pc_key, pc_value_func in self.point_clouds[state.phase].items():
            pc_value = pc_value_func()
            bsz, _, _ = pc_value.shape
            colors = torch.randint(0, 255, (bsz, 1, 3), device=pc_value.device).expand_as(pc_value)
            pc_value, colors = pc_value.flatten(0, 1).unsqueeze(0), colors.flatten(0, 1).unsqueeze(0)
            writer.add_mesh(pc_key, pc_value, colors=colors, global_step=state.num_steps)

    def clear(self, state: State) -> None:
        self.scalars[state.phase].clear()
        self.strings[state.phase].clear()
        self.images[state.phase].clear()
        self.videos[state.phase].clear()
        self.histograms[state.phase].clear()
        self.point_clouds[state.phase].clear()
