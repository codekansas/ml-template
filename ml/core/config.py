from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Generic, Optional, Type, TypeVar

import torch
from omegaconf import MISSING, DictConfig

if TYPE_CHECKING:
    from ml.core.registry import Objects

FieldType = Any


def conf_field(
    value: FieldType,
    *,
    help: Optional[str] = None,  # pylint: disable=redefined-builtin
    short: Optional[str] = None,
) -> FieldType:
    """Gets a field for a given value.

    Args:
        value: The default value for the current field
        help: An optional metadata field, which may be parsed to a command
            line argument in some CLIs
        short: An optional metadata field, which may be parsed to a command
            line argument in some CLIs

    Returns:
        The correctly constructed dataclass field
    """

    metadata: Dict[str, Any] = {}
    if help is not None:
        metadata["help"] = help
    if short is not None:
        metadata["short"] = short

    if hasattr(value, "__call__"):
        return field(default_factory=value, metadata=metadata)
    else:
        return field(default=value, metadata=metadata)


BaseConfigT = TypeVar("BaseConfigT", bound="BaseConfig")


@dataclass
class BaseConfig:
    """Defines the base class for all configs."""

    name: str = conf_field(MISSING, short="n", help="The referenced name of the object to construct")

    @classmethod
    def get_defaults(cls: Type[BaseConfigT]) -> Dict[str, BaseConfigT]:
        """Returns default configurations.

        Returns:
            A dictionary of default configurations for the current config
        """

        return {}

    @classmethod
    def resolve(cls: Type[BaseConfigT], config: BaseConfigT) -> None:
        """Runs post-construction config resolution.

        Args:
            config: The config to resolve
        """


class BaseObject(Generic[BaseConfigT]):
    """Defines the base class for all objects."""

    def __init__(self, config: BaseConfigT) -> None:
        self.config: BaseConfigT = config


class BaseObjectWithPointers(BaseObject[BaseConfigT], Generic[BaseConfigT]):
    """Defines the base class for all objects with pointers to other objects."""

    def __init__(self, config: BaseConfigT) -> None:
        super().__init__(config)

        self._raw_config: Optional[DictConfig] = None
        self._objects: Optional["Objects"] = None

    @property
    @torch.jit.unused
    def raw_config(self) -> DictConfig:
        if self._raw_config is None:
            raise RuntimeError("Cannot access raw config yet; it has yet to be assigned")
        return self._raw_config

    def set_raw_config(self, raw_config: DictConfig) -> None:
        if self._raw_config is not None:
            raise RuntimeError("The raw config object was already written")
        self._raw_config = raw_config

    @property
    @torch.jit.unused
    def objects(self) -> "Objects":
        if self._objects is None:
            raise RuntimeError("Cannot access objects yet; it has yet to be assigned")
        return self._objects

    def set_objects(self, objects: "Objects") -> None:
        if self._objects is not None:
            raise RuntimeError("The objects field has already been written")
        self._objects = objects
