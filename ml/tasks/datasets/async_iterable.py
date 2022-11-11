from __future__ import annotations

import asyncio
import logging
import queue
import threading
from typing import AsyncIterator, Iterator, TypeVar

from torch.utils.data.dataset import IterableDataset

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def add_to_queue(async_iter: AsyncIterator[T], q: "queue.Queue[T | None]") -> None:
    try:
        async for item in async_iter:
            assert item is not None, "Item should not be None"
            q.put(item)
    finally:
        q.put(None)


def thread_worker(async_iter: AsyncIterator[T], q: "queue.Queue[T | None]") -> None:
    asyncio.run(add_to_queue(async_iter, q))


def thread_async_iter(async_iter: AsyncIterator[T], max_queue_size: int) -> Iterator[T]:
    q: "queue.Queue[T | None]" = queue.Queue(maxsize=max_queue_size)
    thread = threading.Thread(target=thread_worker, args=(async_iter, q), daemon=True)
    thread.start()
    while True:
        item = q.get(block=True)
        if item is None:
            break
        yield item
    thread.join()


class AsyncIterableDataset(IterableDataset[T]):
    def __init__(self, max_async_queue_size: int = 2) -> None:
        super().__init__()

        # The async iterator blocks on the queue if it has more than this many
        # elements, in order to avoid having the queue get too large.
        self.max_async_queue_size = max_async_queue_size

    def __aiter__(self) -> AsyncIterator[T]:
        raise NotImplementedError

    def __iter__(self) -> Iterator[T]:
        return thread_async_iter(self.__aiter__(), self.max_async_queue_size)
