import atexit
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Set, Type

import pandas as pd
from torch import Tensor

from ml.core.registry import register_logger
from ml.core.state import Phase, State
from ml.loggers.base import BaseLogger, BaseLoggerConfig
from ml.utils.logging import get_log_item


@dataclass
class PandasLoggerConfig(BaseLoggerConfig):
    pass


def get_pd_type(column: pd.Series, t: Type) -> pd.Series:
    if t == str:
        return column.astype("string")
    if t == int:
        return column.astype("Int64")
    if t == float:
        return column.astype("float")
    if t == datetime.datetime:
        return pd.to_datetime(column)
    if t == datetime.timedelta:
        return pd.to_timedelta(column)
    raise NotImplementedError(t)


@register_logger("pandas", PandasLoggerConfig)
class PandasLogger(BaseLogger[PandasLoggerConfig]):
    def __init__(self, config: PandasLoggerConfig) -> None:
        super().__init__(config)

        self.log_values: Dict[Phase, Dict[str, Dict[str, Callable[[], Any]]]] = {}
        self.types: Dict[str, Type] = {}
        self.df = pd.DataFrame()

    def initialize(self, log_directory: Path) -> None:
        super().initialize(log_directory)

        log_directory.mkdir(exist_ok=True, parents=True)
        csv_file = log_directory / "df.csv"
        atexit.register(self.save, csv_file)

    def save(self, csv_file: Path) -> None:
        self.df.to_csv(csv_file)

    @property
    def columns(self) -> Set[str]:
        return set(self.df.columns)

    @property
    def num_rows(self) -> int:
        return self.df.shape[0]

    @property
    def num_columns(self) -> int:
        return self.df.shape[1]

    def add_column(self, key: str) -> None:
        if key in self.df.columns:
            raise KeyError(f"Can't add {key} column, it already exists")
        self.df[key] = [pd.NA] * self.num_rows

    def get_log_dict(self, state: State, namespace: Optional[str]) -> Dict[str, Callable[[], Any]]:
        if namespace is None:
            namespace = "default"
        if state.phase not in self.log_values:
            self.log_values[state.phase] = {}
        if namespace not in self.log_values[state.phase]:
            self.log_values[state.phase][namespace] = {}
        return self.log_values[state.phase][namespace]

    def check_type(self, key: str, value: Any, namespace: str) -> None:
        namespace_key = f"{namespace}_{key}"
        if namespace_key not in self.types:
            self.types[namespace_key] = type(value)
        elif not isinstance(value, self.types[namespace_key]):
            raise ValueError(f"Unexpected value type {type(value)} for {key}; expected {self.types[namespace_key]}")

    def add_item(self, key: str, value: Callable[[], Any], state: State, namespace: str) -> None:
        self.check_type(key, value, namespace)
        log_dict = self.get_log_dict(state, namespace)
        if key in log_dict:
            raise KeyError(f"Trying to insert duplicate values for {key}")
        log_dict[key] = value

    def log_scalar(self, key: str, value: Callable[[], int | float | Tensor], state: State, namespace: str) -> None:
        self.add_item(key, value, state, namespace)

    def log_string(self, key: str, value: Callable[[], str], state: State, namespace: str) -> None:
        self.add_item(key, value, state, namespace)

    def write(self, state: State) -> None:
        if not (phase_log_values := self.log_values.get(state.phase)):
            return

        # Gets elapsed time since last write.
        self.add_item("elapsed_time", lambda: datetime.datetime.now() - self.start_time, state, namespace="timers")

        # Flattens log values and calls item functions.
        log_values = {f"{k}_{kk}": get_log_item(vv()) for k, v in phase_log_values.items() for kk, vv in v.items()}

        # Adds missing columns.
        missing_columns = {column for column in log_values.keys() if column not in self.columns}
        for missing_column in missing_columns:
            self.add_column(missing_column)

        # Creates a new row.
        self.df.loc[self.num_rows] = [log_values.get(column, pd.NA) for column in self.df.columns]
        for column in missing_columns:
            self.df[column] = get_pd_type(self.df[column], self.types[column])

    def clear(self, state: State) -> None:
        if state.phase in self.log_values:
            self.log_values[state.phase].clear()
