# pylint: disable=redefined-builtin
"""Defines some utility functions for atomic file operations."""


import os
import tempfile as tmp
from contextlib import contextmanager
from pathlib import Path
from typing import (
    IO,
    Any,
    BinaryIO,
    Callable,
    ContextManager,
    Iterator,
    Literal,
    TextIO,
    Union,
    cast,
    overload,
)


def fsync_directory(path: Path) -> None:
    fid = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fid)
    finally:
        if fid >= 0:
            os.close(fid)


def atomic_save(
    save_func: Callable[[Path], None],
    save_path: str | Path,
    durable: bool = False,
) -> None:
    """Performs an atomic save using a temporary file.

    Args:
        save_func: The function to call, that saves the file
        save_path: Where to save the file
        durable: If set, make the write durable
    """

    save_path = Path(save_path)
    tmp_file = save_path.parent / f".tmp_{save_path.name}"
    renamed = False
    try:
        save_func(tmp_file)
        tmp_file.rename(save_path)
        renamed = True
        if durable:
            fsync_directory(save_path.parent)
    finally:
        if not renamed:
            tmp_file.unlink(missing_ok=True)


@contextmanager
def tempfile(suffix: str = "", dir: str | Path | None = None) -> Iterator[str]:
    tf = tmp.NamedTemporaryFile(delete=False, suffix=suffix, dir=dir)
    tf.file.close()
    try:
        yield tf.name
    finally:
        try:
            os.remove(tf.name)
        except OSError as e:
            if e.errno != 2:
                raise


TextModes = Literal["r", "w", "a"]
BinaryModes = Literal["rb", "wb", "ab"]


@overload
def open_atomic(
    filepath: str | Path,
    mode: BinaryModes,
    *,
    encoding: str = "utf-8",
    fsync: bool = False,
) -> ContextManager[BinaryIO]:
    ...


@overload
def open_atomic(
    filepath: str | Path,
    mode: TextModes,
    *,
    encoding: str = "utf-8",
    fsync: bool = False,
) -> ContextManager[TextIO]:
    ...


@contextmanager  # type: ignore
def open_atomic(
    filepath: str | Path,
    mode: str,
    *,
    encoding: str = "utf-8",
    fsync: bool = False,
) -> Iterator[IO[Any]]:
    with tempfile(dir=os.path.dirname(os.path.abspath(filepath))) as tmppath:
        with open(tmppath, mode=mode, encoding=encoding) as file:
            try:
                yield cast(Union[TextIO, BinaryIO], file)
            finally:
                if fsync:
                    file.flush()
                    os.fsync(file.fileno())
        os.rename(tmppath, filepath)
