import random
from dataclasses import dataclass
from typing import Generic, Iterable, Iterator, List, TypeVar

import numpy as np
from torch.utils.data.dataset import IterableDataset

T = TypeVar("T")


@dataclass
class DatasetInfo(Generic[T]):
    dataset: IterableDataset[T]
    sampling_rate: float = 1.0


class MultiIterDataset(IterableDataset[T]):
    def __init__(
        self,
        datasets: Iterable[DatasetInfo[T]],
        *,
        until_all_empty: bool = False,
        iterate_forever: bool = False,
    ) -> None:
        """Defines a dataset for iterating from multiple iterable datasets.

        Args:
            datasets: The information about the datasets to iterate from and
                how to iterate them; specifically,
            until_all_empty: If set, iterates until all datasets are empty,
                otherwise only iterate until any dataset is empty
            iterate_forever: If set, iterate child dataset forever
        """
        super().__init__()

        self.datasets = list(datasets)
        assert all(i.sampling_rate > 0 for i in self.datasets)

        self.until_all_empty = until_all_empty
        self.iterate_forever = iterate_forever

    iterators: List[Iterator[T]]
    rate_cumsum: np.ndarray

    def __iter__(self) -> Iterator[T]:
        self.rate_cumsum = np.concatenate([np.array([0]), np.cumsum([i.sampling_rate for i in self.datasets])])
        self.iterators = [i.dataset.__iter__() for i in self.datasets]
        return self

    def __next__(self) -> T:
        while True:
            val = random.random() * self.rate_cumsum[-1]
            idx = np.searchsorted(self.rate_cumsum, val, side="right") - 1
            iterator = self.iterators[idx]

            try:
                return iterator.__next__()

            except StopIteration:
                if not (self.until_all_empty or self.iterate_forever) or len(self.iterators) == 1:
                    raise

                if self.iterate_forever:
                    # Restart iterator.
                    self.iterators[idx] = self.datasets[idx].dataset.__iter__()
                else:
                    self.iterators.pop(idx)
                    lhs, rhs = self.rate_cumsum[:idx], self.rate_cumsum[idx + 1 :] - self.rate_cumsum[idx]
                    self.rate_cumsum = np.concatenate([lhs, rhs])
