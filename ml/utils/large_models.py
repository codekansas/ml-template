"""Utility functions for dealing with large models."""

from contextlib import contextmanager
from typing import Callable, Iterator, Optional

import torch
from torch import Tensor, nn


@contextmanager
def init_empty_weights(include_buffers: bool = False) -> Iterator[None]:
    """Avoid instantiating weights when initializing a model.

    A context manager under which models are initialized with all parameters on
    the meta device, therefore creating an empty model. Useful when just
    initializing the model would blow the available RAM.

    Args:
        include_buffers: Whether or not to also put all buffers on the meta
            device while initializing.

    Yields:
        An empty context manager
    """

    old_register_parameter = nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = nn.Module.register_buffer

    def register_empty_parameter(module: nn.Module, name: str, param: Optional[nn.Parameter]) -> None:
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            meta_param = module._parameters[name].to(torch.device("meta"))  # type: ignore
            module._parameters[name] = param_cls(meta_param, **kwargs)  # type: ignore

    def register_empty_buffer(module: nn.Module, name: str, buffer: Optional[Tensor]) -> None:
        old_register_buffer(module, name, buffer)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(torch.device("meta"))  # type: ignore

    try:
        nn.Module.register_parameter = register_empty_parameter  # type: ignore
        if include_buffers:
            nn.Module.register_buffer = register_empty_buffer  # type: ignore
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter  # type: ignore
        if include_buffers:
            nn.Module.register_buffer = old_register_buffer  # type: ignore


def meta_to_empty_func(device: torch.device, dtype: Optional[torch.dtype]) -> Callable[[Tensor], Tensor]:
    def _func(t: Tensor) -> Tensor:
        if not t.is_meta:
            return t
        if t.is_floating_point() and dtype is not None:
            return torch.empty(t.shape, device=device, dtype=dtype)
        return torch.empty(t.shape, device=device)

    return _func
