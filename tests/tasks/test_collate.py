import pytest
import torch

from ml.tasks.base import BaseTask
from ml.tasks.datasets.collate import pad_sequence


def test_collate_list() -> None:
    items = [[0], [1], [2], [3]]
    collated = BaseTask.collate_fn(items)
    assert collated is not None
    assert len(collated) == 1
    assert collated[0].shape == (4, 1)


def test_collate_tuple() -> None:
    items = [(0,), (1,), (2,), (3,)]
    collated = BaseTask.collate_fn(items)
    assert collated is not None
    assert len(collated) == 1
    assert collated[0].shape == (4, 1)


def test_collate_dict() -> None:
    items = [{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}]
    collated = BaseTask.collate_fn(items)
    assert collated is not None
    assert len(collated) == 1
    assert collated["a"].shape == (4, 1)


@pytest.mark.parametrize("max_length", [None, 5])
@pytest.mark.parametrize("left_pad", [True, False])
@pytest.mark.parametrize("left_truncate", [True, False])
def test_pad_sequence(max_length: int | None, left_pad: bool, left_truncate: bool) -> None:
    items = [torch.randn(1, ndims, 3) for ndims in [3, 6, 5]]
    sequence = pad_sequence(items, max_length=max_length, left_pad=left_pad, left_truncate=left_truncate, dim=1)
    tensor = BaseTask.collate_fn(sequence, mode="concat")
    assert tensor is not None
    assert tensor.shape == (3, 6 if max_length is None else max_length, 3)
