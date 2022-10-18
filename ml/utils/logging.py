from __future__ import annotations

import logging
import math
import sys
from typing import Any, Dict

import tqdm
from torch import Tensor

from ml.core.env import is_debugging
from ml.utils.colors import Color, colorize

# Logging level to show on all ranks.
INFOALL = logging.INFO + 1


class RankFilter(logging.Filter):
    def __init__(self, *, rank: int | None = None) -> None:
        """Logging filter which filters out INFO logs on non-zero ranks.

        Args:
            rank: The current rank
        """

        super().__init__()

        self.rank = rank

        # Log using INFOALL to show on all ranks.
        logging.addLevelName(INFOALL, "INFOALL")
        levels_to_log_all_ranks = (INFOALL, logging.CRITICAL, logging.ERROR, logging.WARNING)
        self.log_all_ranks = {logging.getLevelName(level) for level in levels_to_log_all_ranks}

    def filter(self, record: logging.LogRecord) -> bool:
        if self.rank is None or self.rank == 0:
            return True
        if record.levelname in self.log_all_ranks:
            return True
        return False


class ColoredFormatter(logging.Formatter):
    """Defines a custom formatter for displaying logs."""

    RESET_SEQ = "\033[0m"
    COLOR_SEQ = "\033[1;%dm"
    BOLD_SEQ = "\033[1m"

    COLORS: Dict[str, Color] = {
        "WARNING": "yellow",
        "INFOALL": "magenta",
        "INFO": "cyan",
        "DEBUG": "white",
        "CRITICAL": "yellow",
        "FATAL": "red",
        "ERROR": "red",
    }

    def __init__(
        self,
        *,
        prefix: str | None = None,
        rank: int | None = None,
        world_size: int | None = None,
        use_color: bool = True,
    ):
        message = "{levelname:^19s} [{name}] {message}"
        if prefix is not None:
            message = colorize(prefix, "white") + " " + message
        if rank is not None or world_size is not None:
            assert rank is not None and world_size is not None
            digits = int(math.log10(world_size) + 1)
            message = "[" + colorize(f"{rank:>{digits}}", "blue") + "] " + message
        super().__init__(message, style="{")

        self.rank = rank
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        levelname = record.levelname

        if levelname == "DEBUG":
            record.levelname = ""
        else:
            if self.use_color and levelname in self.COLORS:
                record.levelname = colorize(levelname, self.COLORS[levelname])
        return logging.Formatter.format(self, record)


class TqdmHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def configure_logging(
    *,
    prefix: str | None = None,
    rank: int | None = None,
    world_size: int | None = None,
    use_tqdm: bool = False,
) -> None:
    """Instantiates print logging, to either stdout or tqdm.

    Args:
        prefix: An optional prefix to add to the logger
        rank: The current rank, or None if not using multiprocessing
        world_size: The total world size, or None if not using multiprocessing
        use_tqdm: Write using TQDM instead of sys.stdout
    """

    if rank is not None or world_size is not None:
        assert rank is not None and world_size is not None
    root_logger = logging.getLogger()
    while root_logger.hasHandlers():
        root_logger.removeHandler(root_logger.handlers[0])
    handler = TqdmHandler() if use_tqdm else logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColoredFormatter(prefix=prefix, rank=rank, world_size=world_size))
    handler.addFilter(RankFilter(rank=rank))
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.DEBUG if is_debugging() else logging.INFO)


def get_log_item(item: Any) -> Any:
    if isinstance(item, Tensor):
        return item.detach().cpu().item()
    return item
