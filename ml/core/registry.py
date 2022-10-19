from __future__ import annotations

import datetime
import functools
import hashlib
import importlib.util
import inspect
import json
import logging
import os
import shutil
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    Iterator,
    List,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)
from uuid import uuid4

from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.basecontainer import BaseContainer

from ml.core.config import BaseConfig, BaseObject, BaseObjectWithPointers
from ml.core.env import get_stage_dir
from ml.utils.colors import colorize
from ml.utils.paths import is_relative_to
from ml.utils.timer import Timer

if TYPE_CHECKING:
    from ml.loggers.base import BaseLogger, BaseLoggerConfig
    from ml.lr_schedulers.base import BaseLRScheduler, BaseLRSchedulerConfig
    from ml.models.base import BaseModel, BaseModelConfig
    from ml.optimizers.base import BaseOptimizer, BaseOptimizerConfig
    from ml.tasks.base import BaseTask, BaseTaskConfig
    from ml.trainers.base import BaseTrainer, BaseTrainerConfig

logger = logging.getLogger(__name__)

Entry = TypeVar("Entry", bound=BaseObject)
SpecificEntry = TypeVar("SpecificEntry")
Config = TypeVar("Config", bound=BaseConfig)

# Special key in the config, cooresponding to the reserved keyword in the
# BaseConfig, which is used to reference the object to construct.
NAME_KEY = "name"

# This points to the root directory location for the package.
ROOT_DIR = Path(__file__).parent.parent.parent.resolve()

# Date format for staging environments.
DATE_FORMAT = "%Y-%m-%d"

# Maximum number of days to keep a staging directory around. This should
# correspond to the maximum number of days that an experiment could run.
MAX_STAGING_DAYS = 31


def get_name(key: str, config: BaseContainer) -> str:
    if not isinstance(config, DictConfig):
        raise ValueError(f"Expected {key} config to be a dictionary, got {type(config)}")
    if NAME_KEY not in config:
        raise ValueError(f"Malformed {key} config; missing expected key {NAME_KEY}")
    name = config[NAME_KEY]
    if not isinstance(name, str):
        raise ValueError(f"Expected {key} name to be a string, got {name}")
    return name


def stage_environment() -> Path:
    """Stages the current environment to a root directory.

    Returns:
        The stage environment path
    """

    stage_dir = get_stage_dir()

    with Timer("getting files to stage"):
        fpaths: List[Path] = []
        for module in sys.modules.values():
            if (fpath_str := getattr(module, "__file__", None)) is None:
                continue
            if not is_relative_to(fpath := Path(fpath_str).resolve(), ROOT_DIR):
                continue
            fpaths.append(fpath)

    assert fpaths, "Couldn't find any file paths to stage!"

    with Timer("computing hash of current environment"):
        hashobj = hashlib.md5()
        for fpath in fpaths:
            with open(fpath, "rb") as f:
                while data := f.read(65536):
                    hashobj.update(data)
        hashval = hashobj.hexdigest()

    date_str = datetime.datetime.now().strftime(DATE_FORMAT)
    out_dir = stage_dir / f"{date_str}-{hashval[:10]}"
    if not out_dir.exists():
        with Timer("copying files to staging directory"):
            tmp_dir = stage_dir / ".tmp" / str(uuid4())
            if tmp_dir.parent.exists():
                shutil.rmtree(tmp_dir.parent)
            tmp_dir.mkdir(exist_ok=False, parents=True)
            for fpath in fpaths:
                new_fpath = tmp_dir / fpath.relative_to(ROOT_DIR)
                new_fpath.parent.mkdir(exist_ok=True, parents=True)
                shutil.copyfile(fpath, new_fpath)
            tmp_dir.rename(out_dir)
            tmp_dir.parent.rmdir()

    with Timer("removing old directories"):
        cur_time = datetime.datetime.now()
        for dpath in stage_dir.iterdir():
            dir_age = cur_time - datetime.datetime.fromtimestamp(os.stat(dpath).st_mtime)
            if dir_age > datetime.timedelta(days=14):
                logger.info("Removing old staging directory %s", dpath)
                shutil.rmtree(dpath)

    return out_dir


class register_base(ABC, Generic[Entry, Config]):  # pylint: disable=invalid-name
    """Defines the base registry type.

    Usage:

    ```
    @register("my_thing", dataclass=MyThingConfig)
    class MyThing(BaseThing):
        def __init__(self, config: MyThingConfig):
            super().__init__(config)
    ```
    """

    REGISTRY: Dict[str, Tuple[Type[Entry], Type[Config]]] = {}
    REGISTRY_LOCATIONS: Dict[str, Path] = {}

    @classmethod
    @abstractmethod
    def search_directory(cls) -> Path:
        """Returns the directory to search for entries."""

    @classmethod
    @abstractmethod
    def config_key(cls) -> str:
        """Returns the key for the current item from the config."""

    @classmethod
    def registry_path(cls) -> Path:
        return Path(__file__).parent.resolve() / ".cache" / f"{cls.config_key()}.json"

    @classmethod
    @functools.lru_cache(None)
    def load_registry_locations(cls) -> None:
        registry_path = cls.registry_path()
        if not registry_path.exists():
            return
        with open(registry_path, "r", encoding="utf-8") as f:
            try:
                cached_registry_locations = json.load(f)
            except json.decoder.JSONDecodeError:
                return
        new_locations = {
            key: Path(reg_loc)
            for key, reg_loc in cached_registry_locations.items()
            if key not in cls.REGISTRY_LOCATIONS and Path(reg_loc).is_file()
        }
        cls.REGISTRY_LOCATIONS.update(new_locations)

    @classmethod
    def save_registry_locations(cls) -> None:
        registry_path = cls.registry_path()
        registry_path.parent.mkdir(exist_ok=True, parents=True)
        registry_locations = {k: str(v.resolve()) for k, v in cls.REGISTRY_LOCATIONS.items() if v.is_file()}
        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(registry_locations, f, indent=2)

    @classmethod
    @functools.lru_cache(None)
    def manual_import(cls, path: Path) -> None:
        with Timer(f"importing '{path}'"):
            try:
                rel_path = path.relative_to(ROOT_DIR)
                module_name = ".".join(list(rel_path.parts[:-1]) + [rel_path.stem])
                if module_name not in sys.modules:
                    spec = importlib.util.spec_from_file_location(module_name, str(path))
                    assert spec is not None
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    loader = spec.loader
                    assert loader is not None
                    loader.exec_module(module)
            except Exception:
                logger.exception("Caught exception while importing %s", path)

    @classmethod
    def populate_registry(cls, name: str) -> None:
        """Populates the registry until it has the requested name available.

        Args:
            name: The name of the registry item to get
        """

        lower_name = name.lower()

        # Check in the existing registry locations.
        if name in cls.REGISTRY_LOCATIONS:
            cls.manual_import(cls.REGISTRY_LOCATIONS[name])
        if name in cls.REGISTRY:
            return

        # First do a quick sweep over the cached registry locations to see if
        # one happens to match the name being imported, since this is likely
        # to be the one we want and it will avoid having to import every file
        # by hand.
        for reg_name, path in cls.REGISTRY_LOCATIONS.items():
            if reg_name.lower().startswith(lower_name):
                cls.manual_import(path)
            if name in cls.REGISTRY:
                return

        # This gets populated the first time we walk the directories, so that
        # the second time we can just iterate through it again.
        subfiles: List[Path] = []

        def iter_directory(curdir: Path) -> Iterator[Path]:
            for subpath in curdir.iterdir():
                if subpath.stem.startswith("__"):
                    continue
                if subpath.is_file() and subpath.suffix == ".py":
                    subfile = subpath.resolve()
                    subfiles.append(subfile)
                    yield subfile
                elif subpath.is_dir():
                    yield from iter_directory(subpath)

        # Next sweep over the search directory and check for prefix matches.
        for path in iter_directory(ROOT_DIR / cls.search_directory()):
            if path.stem.lower().startswith(lower_name) or lower_name.startswith(path.stem.lower()):
                cls.manual_import(path)
                if name in cls.REGISTRY:
                    return

        # Finally, try loading files from the requested import path until
        # we've imported the name that we're looking for.
        for path in subfiles:
            cls.manual_import(path)
            if name in cls.REGISTRY:
                return

    @classmethod
    @functools.lru_cache(None)
    def lookup(cls, name: str) -> Tuple[Type[Entry], Type[Config]]:
        # Just loads the entry, if it already exists.
        if name in cls.REGISTRY:
            return cls.REGISTRY[name]

        # If not found, populates the registry. If still not found, then
        # we're out of luck and should throw an error
        with Timer(f"looking up {name}"):
            cls.load_registry_locations()
            cls.populate_registry(name)
            cls.save_registry_locations()
        if name not in cls.REGISTRY:
            options = "\n".join(f" - {k}" for k in sorted(cls.REGISTRY.keys()))
            logger.error("Couldn't locate %s '%s' in:\n%s", cls.config_key(), name, options)
            raise KeyError(f"Couldn't locate {cls.config_key()} '{name}' from {len(cls.REGISTRY)} options")
        return cls.REGISTRY[name]

    @classmethod
    def build_entry(cls, raw_config: DictConfig) -> Entry | None:
        if cls.config_key() not in raw_config:
            return None
        reg_cfg = raw_config[cls.config_key()]
        reg_name = get_name(cls.config_key(), reg_cfg)
        reg_cls, _ = cls.lookup(reg_name)
        with Timer(f"building {reg_name} from '{cls.REGISTRY_LOCATIONS[reg_name]}'"):
            reg_obj = reg_cls(reg_cfg)
        if isinstance(reg_obj, BaseObjectWithPointers):
            reg_obj.set_raw_config(raw_config)
        return reg_obj

    @classmethod
    def update_config(cls, raw_config: DictConfig) -> None:
        if cls.config_key() not in raw_config:
            return
        reg_cfg = raw_config[cls.config_key()]
        reg_name = get_name(cls.config_key(), reg_cfg)
        _, reg_cfg_cls = cls.lookup(reg_name)
        reg_cfg = OmegaConf.merge(OmegaConf.structured(reg_cfg_cls), reg_cfg)
        raw_config[cls.config_key()] = reg_cfg

    @classmethod
    def resolve_config(cls, raw_config: DictConfig) -> None:
        if cls.config_key() not in raw_config:
            return
        reg_cfg = raw_config[cls.config_key()]
        reg_name = get_name(cls.config_key(), reg_cfg)
        _, reg_cfg_cls = cls.lookup(reg_name)
        reg_cfg_cls.resolve(reg_cfg)
        raw_config[cls.config_key()] = reg_cfg

    def __init__(self, name: str, config: Type[Config]) -> None:
        self.name = name
        self.config = config

    def __call__(self, entry: SpecificEntry) -> SpecificEntry:
        if self.name in self.REGISTRY:
            raise RuntimeError(f"Found duplicate names: {self.name}")

        registry_location = Path(inspect.getfile(cast(Type[Entry], entry)))

        # Adds the registry entry and the entry's location to their respective
        # dictionaries. We overwrite any outdated cache entries.
        self.REGISTRY[self.name] = cast(Tuple[Type[Entry], Type[Config]], (entry, self.config))
        self.REGISTRY_LOCATIONS[self.name] = registry_location

        # Adds all default configurations as well.
        for key, default_cfg in self.config.get_defaults().items():
            self.REGISTRY[key] = cast(Tuple[Type[Entry], Type[Config]], (entry, default_cfg))
            self.REGISTRY_LOCATIONS[key] = registry_location

        return entry


class multi_register_base(register_base[Entry, Config], Generic[Entry, Config]):  # pylint: disable=invalid-name
    """Defines a registry which produces multiple objects."""

    @classmethod
    def update_config(cls, raw_config: DictConfig) -> None:
        if cls.config_key() not in raw_config:
            return
        reg_cfg = raw_config[cls.config_key()]
        if isinstance(reg_cfg, DictConfig):
            reg_cfgs = ListConfig([reg_cfg])
        elif isinstance(reg_cfg, ListConfig):
            reg_cfgs = reg_cfg
        else:
            raise NotImplementedError(f"Invalid logger config type: {type(reg_cfg)}")
        for i, reg_cfg in enumerate(reg_cfgs):
            reg_name = get_name(cls.config_key(), reg_cfg)
            _, reg_cfg_cls = cls.lookup(reg_name)
            reg_cfgs[i] = OmegaConf.merge(OmegaConf.structured(reg_cfg_cls), reg_cfg)
        raw_config[cls.config_key()] = reg_cfgs

    @classmethod
    def resolve_config(cls, raw_config: DictConfig) -> None:
        if cls.config_key() not in raw_config:
            return
        reg_cfgs = cast(ListConfig, raw_config[cls.config_key()])
        for i, reg_cfg in enumerate(reg_cfgs):
            reg_name = get_name(cls.config_key(), reg_cfg)
            _, reg_cfg_cls = cls.lookup(reg_name)
            reg_cfg_cls.resolve(reg_cfg)
            reg_cfgs[i] = reg_cfg
        raw_config[cls.config_key()] = reg_cfgs

    @classmethod
    def build_entry(cls, raw_config: DictConfig) -> List[Entry] | None:  # type: ignore
        if cls.config_key() not in raw_config:
            return None
        reg_cfgs = cast(ListConfig, raw_config[cls.config_key()])
        reg_objs = []
        for reg_cfg in reg_cfgs:
            reg_name = get_name(cls.config_key(), reg_cfg)
            reg_cls, _ = cls.lookup(reg_name)
            with Timer(f"building {reg_name} from '{cls.REGISTRY_LOCATIONS[reg_name]}'"):
                reg_obj = reg_cls(reg_cfg)
            reg_obj.set_raw_config(raw_config)
            reg_objs.append(reg_obj)
        return reg_objs


class register_model(register_base["BaseModel", "BaseModelConfig"]):  # pylint: disable=invalid-name
    """Defines a registry for holding modules."""

    REGISTRY: Dict[str, Tuple[Type["BaseModel"], Type["BaseModelConfig"]]] = {}
    REGISTRY_LOCATIONS: Dict[str, Path] = {}

    @classmethod
    def search_directory(cls) -> Path:
        return Path("ml/models")

    @classmethod
    def config_key(cls) -> str:
        return "model"


class register_task(register_base["BaseTask", "BaseTaskConfig"]):  # pylint: disable=invalid-name
    """Defines a registry for holding tasks."""

    REGISTRY: Dict[str, Tuple[Type["BaseTask"], Type["BaseTaskConfig"]]] = {}
    REGISTRY_LOCATIONS: Dict[str, Path] = {}

    @classmethod
    def search_directory(cls) -> Path:
        return Path("ml/tasks")

    @classmethod
    def config_key(cls) -> str:
        return "task"


class register_trainer(register_base["BaseTrainer", "BaseTrainerConfig"]):  # pylint: disable=invalid-name
    """Defines a registry for holding trainers."""

    REGISTRY: Dict[str, Tuple[Type["BaseTrainer"], Type["BaseTrainerConfig"]]] = {}
    REGISTRY_LOCATIONS: Dict[str, Path] = {}

    @classmethod
    def search_directory(cls) -> Path:
        return Path("ml/trainers")

    @classmethod
    def config_key(cls) -> str:
        return "trainer"


class register_optimizer(register_base["BaseOptimizer", "BaseOptimizerConfig"]):  # pylint: disable=invalid-name
    """Defines a registry for holding optimizers."""

    REGISTRY: Dict[str, Tuple[Type["BaseOptimizer"], Type["BaseOptimizerConfig"]]] = {}
    REGISTRY_LOCATIONS: Dict[str, Path] = {}

    @classmethod
    def search_directory(cls) -> Path:
        return Path("ml/optimizers")

    @classmethod
    def config_key(cls) -> str:
        return "optimizer"


class register_lr_scheduler(register_base["BaseLRScheduler", "BaseLRSchedulerConfig"]):  # pylint: disable=invalid-name
    """Defines a registry for holding learning rate schedulers."""

    REGISTRY: Dict[str, Tuple[Type["BaseLRScheduler"], Type["BaseLRSchedulerConfig"]]] = {}
    REGISTRY_LOCATIONS: Dict[str, Path] = {}

    @classmethod
    def search_directory(cls) -> Path:
        return Path("ml/lr_schedulers")

    @classmethod
    def config_key(cls) -> str:
        return "lr_scheduler"


class register_logger(multi_register_base["BaseLogger", "BaseLoggerConfig"]):  # pylint: disable=invalid-name
    """Defines a registry for holding loggers."""

    REGISTRY: Dict[str, Tuple[Type["BaseLogger"], Type["BaseLoggerConfig"]]] = {}
    REGISTRY_LOCATIONS: Dict[str, Path] = {}

    @classmethod
    def search_directory(cls) -> Path:
        return Path("ml/loggers")

    @classmethod
    def config_key(cls) -> str:
        return "logger"


@dataclass(frozen=True)
class Objects:
    raw_config: DictConfig
    model: "BaseModel | None" = None
    task: "BaseTask | None" = None
    trainer: "BaseTrainer | None" = None
    optimizer: "BaseOptimizer | None" = None
    lr_scheduler: "BaseLRScheduler | None" = None
    logger: "List[BaseLogger] | None" = None

    def __post_init__(self) -> None:
        # After initializing the object container, we add a pointer to the
        # current container to each of the constructed objects.
        if self.task is not None:
            self.task.set_objects(self)
        if self.trainer is not None:
            self.trainer.set_objects(self)
            if self.logger is not None:
                self.trainer.add_loggers(self.logger)
        if self.optimizer is not None:
            self.optimizer.set_objects(self)
        if self.lr_scheduler is not None:
            self.lr_scheduler.set_objects(self)
        if self.logger is not None:
            for sublogger in self.logger:
                sublogger.set_objects(self)

    def summarize(self) -> str:
        parts: Dict[str, str] = {}
        if self.model is not None:
            parts["Model"] = inspect.getfile(self.model.__class__)
        if self.task is not None:
            parts["Task"] = inspect.getfile(self.task.__class__)
        if self.trainer is not None:
            parts["Trainer"] = inspect.getfile(self.trainer.__class__)
        if self.optimizer is not None:
            parts["Optimizer"] = inspect.getfile(self.optimizer.__class__)
        if self.lr_scheduler is not None:
            parts["LR Scheduler"] = inspect.getfile(self.lr_scheduler.__class__)
        return "Components:" + "".join(f"\n ↪ {colorize(k, 'green')}: {v}" for k, v in parts.items())

    @classmethod
    def resolve_config(cls, config: DictConfig) -> None:
        """Resolves the config in-place.

        Args:
            config: The config to resolve
        """

        with Timer("building config"):
            # Pre-builds the config using the structured configs.
            register_model.update_config(config)
            register_task.update_config(config)
            register_trainer.update_config(config)
            register_optimizer.update_config(config)
            register_lr_scheduler.update_config(config)
            register_logger.update_config(config)

        with Timer("resolving configs"):
            # Resolves the final config once all structured configs have been merged.
            OmegaConf.resolve(config)

            # Runs object-specific resolutions.
            register_model.resolve_config(config)
            register_task.resolve_config(config)
            register_trainer.resolve_config(config)
            register_optimizer.resolve_config(config)
            register_lr_scheduler.resolve_config(config)
            register_logger.resolve_config(config)

    @classmethod
    def parse_raw_config(cls, config: DictConfig) -> "Objects":
        """Parses a raw config to the objects it contains.

        Args:
            config: The raw DictConfig to parse

        Returns:
            The parsed Objects dataclass
        """

        objs = Objects(
            raw_config=config,
            model=register_model.build_entry(config),
            task=register_task.build_entry(config),
            trainer=register_trainer.build_entry(config),
            optimizer=register_optimizer.build_entry(config),
            lr_scheduler=register_lr_scheduler.build_entry(config),
            logger=register_logger.build_entry(config),
        )

        logger.info("%s", objs.summarize())

        return objs

    @classmethod
    def from_config_file(cls, config_path: Union[str, Path], **overrides: Any) -> "Objects":
        config = OmegaConf.load(config_path)
        config = OmegaConf.merge(config, DictConfig(overrides))
        return cls.parse_raw_config(cast(DictConfig, config))
