from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from torch import Tensor

from ml.core.config import conf_field
from ml.core.registry import register_logger
from ml.core.state import Phase, State
from ml.loggers.base import BaseLogger, BaseLoggerConfig
from ml.utils.datetime import format_timedelta


@dataclass
class StdoutLoggerConfig(BaseLoggerConfig):
    precision: int = conf_field(4, help="Scalar precision to log")


@register_logger("stdout", StdoutLoggerConfig)
class StdoutLogger(BaseLogger[StdoutLoggerConfig]):
    def __init__(self, config: StdoutLoggerConfig) -> None:
        super().__init__(config)

        self.log_values: Dict[Phase, Dict[str, Dict[str, Callable[[], Any]]]] = {}
        self.logger = logging.getLogger("stdout")

    def initialize(self, log_directory: Path) -> None:
        super().initialize(log_directory)

        log_directory.mkdir(exist_ok=True, parents=True)
        file_handler = logging.FileHandler(log_directory / "stdout.log")
        self.logger.addHandler(file_handler)
        self.logger.debug("Finished initializing logger")

    def get_log_dict(self, state: State, namespace: Optional[str]) -> Dict[str, Callable[[], Any]]:
        if namespace is None:
            namespace = "default"
        if state.phase not in self.log_values:
            self.log_values[state.phase] = {}
        if namespace not in self.log_values[state.phase]:
            self.log_values[state.phase][namespace] = {}
        return self.log_values[state.phase][namespace]

    def format_number(self, value: int | float) -> str:
        if isinstance(value, int):
            return str(value)
        return f"{value:.{self.config.precision}g}"

    def log_scalar(self, key: str, value: Callable[[], int | float | Tensor], state: State, namespace: str) -> None:
        self.get_log_dict(state, namespace)[key] = value

    def log_string(self, key: str, value: Callable[[], str], state: State, namespace: str) -> None:
        self.get_log_dict(state, namespace)[key] = value

    def write(self, state: State) -> None:
        if not (phase_log_values := self.log_values.get(state.phase)):
            return

        # Gets elapsed time since last write.
        elapsed_time = datetime.datetime.now() - self.start_time
        elapsed_time_str = format_timedelta(elapsed_time)

        def as_str(value: Any) -> str:
            if isinstance(value, str):
                return f'"{value}"'
            if isinstance(value, Tensor):
                value = value.detach().float().cpu().item()
            if isinstance(value, (int, float)):
                return self.format_number(value)
            raise TypeError(f"Unexpected log type: {type(value)}")

        def get_section_string(name: str, section: Dict[str, Any]) -> str:
            return '"' + name + '": {' + ", ".join(f'"{k}": {as_str(v())}' for k, v in sorted(section.items())) + "}"

        # Writes a log string to stdout.
        log_string = ", ".join(get_section_string(k, v) for k, v in sorted(phase_log_values.items()))
        self.logger.info("%s [%s] {%s}", state.phase, elapsed_time_str, log_string)

    def clear(self, state: State) -> None:
        if state.phase in self.log_values:
            self.log_values[state.phase].clear()
