import sys

from ml.core.env import add_global_tag
from ml.core.registry import Objects
from ml.scripts.train import train_main
from ml.utils.cli import parse_cli
from ml.utils.distributed import get_rank_optional, get_world_size_optional
from ml.utils.logging import configure_logging


def call_train() -> None:
    configure_logging(rank=get_rank_optional(), world_size=get_world_size_optional())

    def show_help() -> None:
        print("Usage: train /path/to/config.yaml", file=sys.stderr)
        sys.exit(1)

    # Parses the raw command line options.
    args = sys.argv[1:]
    if len(args) == 0:
        show_help()

    # Adds a global tag with the currently-selected option.
    add_global_tag("train")

    # Resolves the config to the correct objects.
    config = parse_cli(args)
    Objects.resolve_config(config)
    objs = Objects.parse_raw_config(config)

    # Calls main training loop.
    train_main(objs)


if __name__ == "__main__":
    call_train()
