import functools
import random

import numpy as np
import pytest
import torch
from _pytest.python import Function


@pytest.fixture(autouse=True)
def set_random_seed() -> None:
    random.seed(1337)
    np.random.seed(1337)
    torch.manual_seed(1337)


@functools.lru_cache()
def has_gpu() -> bool:
    return torch.cuda.is_available()


def pytest_runtest_setup(item: Function) -> None:
    for mark in item.iter_markers():
        if mark.name == "has_gpu" and not has_gpu():
            pytest.skip("Skipping because this test requires a GPU and none is available")
