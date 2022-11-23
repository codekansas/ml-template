import logging
from pathlib import Path

from omegaconf import DictConfig

from ml.utils.compiler import compile_training_loop

logger = logging.getLogger(__name__)


def compile_main(config: DictConfig) -> None:
    """Compiles the training loop into a single file."""  # noqa

    # Gets a unique output file path.
    out_path = Path.cwd() / "out" / "train.py"
    index = 1
    while out_path.exists():
        index += 1
        out_path = Path.cwd() / "out" / f"train_{index}.py"

    compile_training_loop(config, out_path)
    logger.info("Compiled to %s", out_path)
