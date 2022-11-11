import asyncio
import queue
import threading
from typing import AsyncIterator, Iterator, Optional, TypeVar

from torch.utils.data.dataset import IterableDataset

T = TypeVar("T")


def iter_async_items(
    async_iter: AsyncIterator[T],
    loop: asyncio.AbstractEventLoop,
    max_queue_size: int,
) -> Iterator[T]:
    q: "queue.Queue[Optional[T]]" = queue.Queue(maxsize=max_queue_size)

    async def async_iter_to_queue() -> None:
        try:
            async for item in async_iter:
                q.put(item)
        finally:
            q.put(None)

    async_result = asyncio.run_coroutine_threadsafe(async_iter_to_queue(), loop)

    def yield_queue_items() -> Iterator[T]:
        while True:
            next_item = q.get()
            if next_item is None:
                break
            yield next_item
        async_result.result()

    return yield_queue_items()


class AsyncIterableDataset(IterableDataset[T]):
    def __init__(self, max_async_queue_size: int = 2) -> None:
        super().__init__()

        # The async iterator blocks on the queue if it has more than this many
        # elements, in order to avoid having the queue get too large.
        self.max_async_queue_size = max_async_queue_size

        # Placeholders for the async loop and thread.
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None

    def __aiter__(self) -> AsyncIterator[T]:
        raise NotImplementedError

    def __iter__(self) -> Iterator[T]:
        if self.loop is None:
            self.loop = asyncio.get_event_loop()
        if self.thread is None:
            self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
            self.thread.start()
        return iter_async_items(self.__aiter__(), self.loop, self.max_async_queue_size)
