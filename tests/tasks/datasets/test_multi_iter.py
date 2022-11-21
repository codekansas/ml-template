import itertools
import random
from typing import Iterator

from torch.utils.data.dataset import IterableDataset

from ml.tasks.datasets.multi_iter import DatasetInfo, MultiIterDataset


class DummyDataset(IterableDataset[int]):
    def __init__(self, length: int) -> None:
        super().__init__()

        self.length = length

    def __iter__(self) -> Iterator[int]:
        for i in range(self.length):
            yield i


def test_multi_iter_any_empty() -> None:
    """Tests MultiIterDataset when `use_all_empty` is False."""

    datasets = [DummyDataset(i) for i in range(1, 5)]
    ds = MultiIterDataset([DatasetInfo(d, random.random()) for d in datasets], until_all_empty=False)
    assert sum(val for val in ds) < sum(sum(val for val in d) for d in datasets)


def test_multi_iter_all_empty() -> None:
    """Tests MultiIterDataset when `use_all_empty` is True."""

    datasets = [DummyDataset(i) for i in range(1, 5)]
    ds = MultiIterDataset([DatasetInfo(d, random.random()) for d in datasets], until_all_empty=True)
    assert sum(val for val in ds) == sum(sum(val for val in d) for d in datasets)


def test_iter_forever() -> None:
    """Tests iterating forever."""

    datasets = [DummyDataset(i) for i in range(1, 5)]
    ds = MultiIterDataset([DatasetInfo(d, random.random()) for d in datasets], iterate_forever=True)
    assert sum(itertools.islice(ds, 100)) > sum(sum(val for val in d) for d in datasets)
