import asyncio
from typing import AsyncIterator

import torch
from torch.utils.data.dataloader import DataLoader

from ml.tasks.datasets.async_iterable import AsyncIterableDataset


def test_dataset() -> None:
    """Tests functioning of async iterable dataset."""

    class DummyDataset(AsyncIterableDataset[int]):
        def __init__(self) -> None:
            super().__init__()

            self.count = 0

        def __aiter__(self) -> AsyncIterator[int]:
            return self

        async def __anext__(self) -> int:
            self.count += 1
            await asyncio.sleep(1e-2)
            return self.count

    ds = DummyDataset()
    dl = DataLoader(ds, batch_size=4)

    for frame in dl:
        assert (frame == torch.tensor([1, 2, 3, 4])).all().item()
        break
