import functools
from typing import Any, List, cast

import torch
import torch.distributed as dist
from torch import Tensor

from ml.trainers.mixins.device.auto import AutoDevice


@functools.lru_cache
def get_device() -> torch.device:
    return AutoDevice.detect_device().get_device()


class Meter:
    def __init__(self) -> None:
        self._min_val: Tensor | None = None
        self._max_val: Tensor | None = None
        self._total_val: Tensor | None = None
        self._num_seen = torch.zeros((0,), dtype=torch.int64)
        self.has_reduce_been_called = False

    def add(self, value: int | float) -> None:
        device = get_device()
        if self._min_val is None or self._max_val is None or self._total_val is None:
            self._min_val = torch.tensor(value, dtype=torch.float64, device=device)
            self._max_val = torch.tensor(value, dtype=torch.float64, device=device)
            self._total_val = torch.tensor(value, dtype=torch.float64, device=device)
        else:
            self._min_val.clamp_max_(value)
            self._max_val.clamp_min_(value)
            self._total_val.add_(value)
        self._num_seen.add_(1)

    def reduce(self) -> List[Any]:
        if self.has_reduce_been_called:
            raise RuntimeError("`reduce` should only be called once, otherwise you will end up with incorrect values")
        self.has_reduce_been_called = True

        # These are actually the work handles, they just don't have proper
        # type support yet.
        works: List[Any] = []

        if self._min_val is not None:
            works.append(dist.all_reduce(self._min_val, dist.ReduceOp.MIN, async_op=True))
        if self._max_val is not None:
            works.append(dist.all_reduce(self._max_val, dist.ReduceOp.MAX, async_op=True))
        if self._total_val is not None:
            works.append(dist.all_reduce(self._total_val, dist.ReduceOp.SUM, async_op=True))
        works.append(dist.all_reduce(self._num_seen, dist.ReduceOp.SUM, async_op=True))
        return works

    @property
    def num_seen(self) -> int:
        return cast(int, self._num_seen.item())

    @property
    def min_val(self) -> float | None:
        return None if self._min_val is None else self._min_val.item()

    @property
    def max_val(self) -> float | None:
        return None if self._max_val is None else self._max_val.item()

    @property
    def mean_val(self) -> float | None:
        return None if self._total_val is None else (self._total_val / self._num_seen).item()
