import logging
import shlex
import sys
from typing import Callable, Dict

from omegaconf import DictConfig

from ml.core.env import add_global_tag
from ml.core.registry import Objects
from ml.scripts import compiler, mp_train, stage, train
from ml.utils.cli import parse_cli
from ml.utils.colors import colorize
from ml.utils.distributed import get_rank_optional, get_world_size_optional
from ml.utils.logging import configure_logging
from ml.utils.random import set_random_seed

logger = logging.getLogger(__name__)


def cli_main() -> None:
    configure_logging(rank=get_rank_optional(), world_size=get_world_size_optional())
    logger.info("Command: %s", shlex.join(sys.argv))

    set_random_seed()

    without_objects_scripts: Dict[str, Callable[[DictConfig], None]] = {
        "compile": compiler.compile_main,
        "mp_train": mp_train.mp_train_main,
        "stage": stage.stage_main,
    }

    with_objects_scripts: Dict[str, Callable[[Objects], None]] = {
        "train": train.train_main,
    }

    scripts: Dict[str, Callable[..., None]] = {**with_objects_scripts, **without_objects_scripts}

    def show_help() -> None:
        script_names = (colorize(script_name, "cyan") for script_name in scripts)
        print(f"Usage: ml < {' / '.join(script_names)} > ...\n", file=sys.stderr)
        for key, func in sorted(scripts.items()):
            if func.__doc__ is None:
                print(f"* {colorize(key, 'green')}", file=sys.stderr)
            else:
                print(f"* {colorize(key, 'green')}: {func.__doc__.strip()}", file=sys.stderr)
        print()
        sys.exit(1)

    # Parses the raw command line options.
    args = sys.argv[1:]
    if len(args) == 0:
        show_help()
    option, args = args[0], args[1:]

    # Adds a global tag with the currently-selected option.
    add_global_tag(option)

    # Parses the command-line arguments to a single DictConfig object.
    config = parse_cli(args)
    Objects.resolve_config(config)

    if option in without_objects_scripts:
        # Special handling for multi-processing; don't initialize anything since
        # everything will be initialized inside the child processes.
        without_objects_scripts[option](config)
    elif option in with_objects_scripts:
        # Converts the raw config to the objects they are pointing at.
        objs = Objects.parse_raw_config(config)
        with_objects_scripts[option](objs)
    else:
        print(f"Invalid option: {colorize(option, 'red')}\n", file=sys.stderr)
        show_help()


if __name__ == "__main__":
    cli_main()
