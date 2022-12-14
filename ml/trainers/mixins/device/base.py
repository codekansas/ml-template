import contextlib
import functools
from abc import ABC, abstractmethod
from dataclasses import is_dataclass
from typing import (
    Any,
    Callable,
    ContextManager,
    Iterable,
    Iterator,
    List,
    Mapping,
    Sequence,
    TypeVar,
)

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data.dataloader import (
    DataLoader,
    _BaseDataLoaderIter,
    _MultiProcessingDataLoaderIter,
)

from ml.core.types import Batch
from ml.utils.timer import Timer

DeviceBatchT = TypeVar("DeviceBatchT", bound=Batch)  # pylint: disable=invalid-name


def get_tasks_outstanding(dataloader_iter: _BaseDataLoaderIter) -> int:
    if isinstance(dataloader_iter, _MultiProcessingDataLoaderIter):
        try:
            return dataloader_iter._worker_result_queue.qsize()
        except NotImplementedError:
            return -2
    return -1


class Prefetcher(Iterable[DeviceBatchT]):
    """Helper class for pre-loading samples into device memory."""

    def __init__(self, to_device_func: Callable[[Any], Any], dataloader: DataLoader) -> None:
        super().__init__()

        self.to_device_func = to_device_func
        self.dataloader = dataloader
        self.dataloader_iter = iter(self.dataloader)
        self.next_sample = None
        self.get_batch_time = -1.0
        self.num_queued_samples = -1

    def get_next_sample(self) -> Any:
        sample = self.to_device_func(next(self.dataloader_iter))
        self.num_queued_samples = get_tasks_outstanding(self.dataloader_iter)
        return sample

    def prefetch(self) -> None:
        try:
            self.next_sample = self.get_next_sample()
        except StopIteration:
            self.next_sample = None

    def recursive_chunk(self, item: Any, chunks: int) -> List[Any]:
        """Applies a function recursively to tensors in an item.

        Args:
            item: The item to apply the function to
            chunks: The number of output chunks

        Returns:
            The item, split into the requested number of chunks
        """

        if isinstance(item, (str, int, float)):
            return [item] * chunks
        if isinstance(item, np.ndarray):
            item = torch.from_numpy(item)
        if isinstance(item, Tensor):
            item_chunk_list = list(item.chunk(chunks, dim=0))
            assert len(item_chunk_list) == chunks, f"{len(item_chunk_list)=} != {chunks=}"
            return item_chunk_list
        if is_dataclass(item):
            item_chunk_dict = {k: self.recursive_chunk(v, chunks) for k, v in item.__dict__.items()}
            return [item.__class__(**{k: v[i] for k, v in item_chunk_dict.items()}) for i in range(chunks)]
        if isinstance(item, Mapping):
            item_chunk_dict = {k: self.recursive_chunk(v, chunks) for k, v in item.items()}
            return [{k: v[i] for k, v in item_chunk_dict.items()} for i in range(chunks)]
        if isinstance(item, Sequence):
            item_chunk_lists = [self.recursive_chunk(i, chunks) for i in item]
            return [[k[i] for k in item_chunk_lists] for i in range(chunks)]
        return item

    @classmethod
    def recursive_apply(cls, item: Any, func: Callable[[Tensor], Tensor]) -> Any:
        """Applies a function recursively to tensors in an item.

        Args:
            item: The item to apply the function to
            func: The function to apply (for the tensor)

        Returns:
            The same item, with the function applied
        """

        if isinstance(item, (str, int, float)):
            return item
        if isinstance(item, np.ndarray):
            item = torch.from_numpy(item)
        if isinstance(item, Tensor):
            return func(item)
        if is_dataclass(item):
            return item.__class__(**{k: cls.recursive_apply(v, func) for k, v in item.__dict__.items()})
        if isinstance(item, Mapping):
            return {k: cls.recursive_apply(v, func) for k, v in item.items()}
        if isinstance(item, Sequence):
            return [cls.recursive_apply(i, func) for i in item]
        return item

    def __iter__(self) -> Iterator[DeviceBatchT]:
        self.prefetch()

        try:
            while True:
                if self.next_sample is None:
                    raise StopIteration
                with Timer("getting batch") as timer:
                    sample = self.next_sample
                    self.prefetch()
                self.get_batch_time = timer.elapsed_time
                yield sample

        except StopIteration:
            # Resets the dataloader if the iteration has completed.
            self.dataloader_iter = iter(self.dataloader)
            raise


class InfinitePrefetcher(Iterable[DeviceBatchT]):
    def __init__(self, prefetcher: Prefetcher[DeviceBatchT]) -> None:
        self.prefetcher = prefetcher

    def __iter__(self) -> Iterator[DeviceBatchT]:
        while True:
            for batch in self.prefetcher:
                yield batch


class BaseDevice(ABC):
    """Base mixin for different trainer device types."""

    @classmethod
    @abstractmethod
    def has_device(cls) -> bool:
        """Detects whether or not the device is available.

        Returns:
            If the device is available
        """

    @classmethod
    @abstractmethod
    def get_device(cls) -> torch.device:
        """Returns the device, for instantiating new tensors.

        Returns:
            The device
        """

    @classmethod
    @abstractmethod
    def get_floating_point_type(cls) -> torch.dtype:
        """Returns the default floating point type to use.

        Returns:
            The dtype
        """

    @classmethod
    def sample_to_device(cls, sample: Any) -> Any:
        device = cls.get_device()
        dtype_fp = cls.get_floating_point_type()
        return Prefetcher.recursive_apply(
            sample,
            lambda t: t.to(
                device,
                dtype_fp if t.is_floating_point() else t.dtype,
                non_blocking=True,
            ),
        )

    @classmethod
    def get_prefetcher(cls, dataloader: DataLoader) -> Prefetcher:
        return Prefetcher(functools.partial(cls.sample_to_device), dataloader)

    @classmethod
    def module_to(cls, module: nn.Module) -> None:
        module.to(cls.get_device(), cls.get_floating_point_type())

    @classmethod
    def tensor_to(cls, tensor: Tensor) -> Tensor:
        device = cls.get_device()
        if tensor.is_floating_point():
            return tensor.to(device, cls.get_floating_point_type())
        return tensor.to(device)

    @classmethod
    def autocast_context(cls, enabled: bool = True) -> ContextManager:
        device_type = cls.get_device().type
        if device_type == "mps":
            device_type = "cpu"
        if device_type not in ("cpu", "cuda"):
            return contextlib.nullcontext()
        return torch.autocast(device_type, enabled=enabled)
