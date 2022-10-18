import logging
import shlex
import sys
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, cast

from omegaconf import DictConfig, OmegaConf

from ml.core.env import add_global_tag, get_global_tags, set_exp_name
from ml.core.registry import Objects
from ml.scripts import mp_train, stage, train
from ml.utils.colors import colorize
from ml.utils.distributed import get_rank_optional, get_world_size_optional
from ml.utils.logging import configure_logging
from ml.utils.random import set_random_seed

logger = logging.getLogger(__name__)

IGNORE_ARGS: Set[str] = {
    "trainer.exp_name",
    "trainer.log_dir_name",
    "trainer.base_run_dir",
    "trainer.run_id",
    "trainer.name",
}


def get_exp_name(prefix: Optional[str] = None, args: Optional[List[str]] = None) -> str:
    parts: List[str] = []
    if prefix is not None:
        parts += [prefix]
    if args is not None:
        parts += args
    if not parts:
        parts = ["run"]
    parts += get_global_tags()
    return ".".join(p for p in parts if p)


def parse_cli(args: List[str]) -> DictConfig:
    """Parses the remaining command-line arguments to a raw config.

    Args:
        args: The raw command-line arguments to parse

    Returns:
        The raw config, loaded from the provided arguments
    """

    def show_help() -> None:
        print("Usage: cmd <path/to/config.yaml> [<new_config.yaml>, ...] overrida.a=1 override.b=2", file=sys.stderr)
        sys.exit(1)

    if len(args) == 0 or "-h" in args or "--help" in args:
        show_help()

    # Builds the configs from the command-line arguments.
    config = DictConfig({})
    get_stem = lambda new_path: Path(new_path).stem
    argument_parts: List[str] = []
    paths: List[Path] = []

    # Parses all of the config paths.
    while len(args) > 0 and (args[0].endswith(".yaml") or args[0].endswith(".yml")):
        paths, new_stem, args = paths + [Path(args[0])], get_stem(args[0]), args[1:]
        argument_parts.append(new_stem)

    # Parses all of the additional config overrides.
    if len(args) > 0:
        split_args = [a.split("=") for a in args]
        assert all(len(a) == 2 for a in split_args), f"Got invalid arguments: {[a for a in split_args if len(a) != 2]}"
        argument_parts += [f"{k.split('.')[-1]}_{v}" for k, v in sorted(split_args) if k not in IGNORE_ARGS]

    # Registers an OmegaConf resolver with the job name.
    if not OmegaConf.has_resolver("exp_name"):
        OmegaConf.register_new_resolver("exp_name", partial(get_exp_name, args=argument_parts))
    set_exp_name(get_exp_name(args=argument_parts))

    # Finally, builds the config.
    try:
        for path in paths:
            config = cast(DictConfig, OmegaConf.merge(config, OmegaConf.load(path)))
        config = cast(DictConfig, OmegaConf.merge(config, OmegaConf.from_dotlist(args)))
    except Exception:
        logger.exception("Error while creating dotlist")
        show_help()

    return config


def main() -> None:
    configure_logging(rank=get_rank_optional(), world_size=get_world_size_optional())
    logger.info("Command: %s", shlex.join(sys.argv))

    set_random_seed()

    without_objects_scripts: Dict[str, Callable[[DictConfig], None]] = {
        "mp_train": mp_train.main,
        "stage": stage.main,
    }

    with_objects_scripts: Dict[str, Callable[[Objects], None]] = {
        "train": train.main,
    }

    scripts: Dict[str, Callable[..., None]] = {**with_objects_scripts, **without_objects_scripts}

    def show_help() -> None:
        script_names = (colorize(script_name, "cyan") for script_name in scripts)
        print(f"Usage: tr < {' / '.join(script_names)} > ...\n", file=sys.stderr)
        for key, func in scripts.items():
            if func.__doc__ is None:
                print(f"* {colorize(key, 'green')}\n", file=sys.stderr)
            else:
                print(f"* {colorize(key, 'green')}: {func.__doc__.strip()}\n", file=sys.stderr)
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
    main()
