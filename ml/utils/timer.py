import errno
import functools
import logging
import os
import signal
import time
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

TimeoutFunc = TypeVar("TimeoutFunc", bound=Callable[..., Any])


class Timer:
    """Defines a simple timer for logging an event."""

    def __init__(self, description: str, min_seconds_to_print: float = 1.0) -> None:
        self.description = description
        self.min_seconds_to_print = min_seconds_to_print
        self._start_time: Optional[float] = None
        self._elapsed_time: Optional[float] = None

    @property
    def elapsed_time(self) -> float:
        assert (elapsed_time := self._elapsed_time) is not None
        return elapsed_time

    def __enter__(self) -> "Timer":
        self._start_time = time.time()
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        assert self._start_time is not None
        self._elapsed_time = time.time() - self._start_time
        if self._elapsed_time > self.min_seconds_to_print:
            logger.warning("Finished %s in %.3g seconds", self.description, self._elapsed_time)


def timeout(seconds: int, error_message: str = os.strerror(errno.ETIME)) -> Callable[[TimeoutFunc], TimeoutFunc]:
    """Decorator for timing out long-running functions.

    Note that this function won't work on Windows.

    Usage:
        try:
            @timeout(5)
            def long_running_function():
                ...
        except TimeoutError:
            handle_timeout()

    Args:
        seconds: Timeout after this many seconds
        error_message: Error message to pass to TimeoutError

    Returns:
        Decorator function
    """

    def decorator(func: TimeoutFunc) -> TimeoutFunc:
        def _handle_timeout(*_: Any) -> None:
            raise TimeoutError(error_message)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapper  # type: ignore

    return decorator
