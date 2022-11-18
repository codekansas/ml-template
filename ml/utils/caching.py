import functools
import json
import logging
import pickle
from typing import (
    Any,
    Callable,
    Generic,
    List,
    Literal,
    Mapping,
    Sequence,
    Tuple,
    TypeVar,
    get_args,
)

import numpy as np

from ml.core.env import get_cache_dir

logger = logging.getLogger(__name__)

Object = TypeVar("Object", bound=Any)
Tk = TypeVar("Tk")
Tv = TypeVar("Tv")

CacheType = Literal["pkl", "json"]


class cached_object:  # pylint: disable=invalid-name
    def __init__(self, cache_key: str, ext: CacheType = "pkl", ignore: bool = False, cache_obj: bool = True) -> None:
        """Defines a wrapper for caching function calls to a file location.

        This is just a convenient way of caching heavy operations to disk,
        using a specific key.

        Usage:

            ```
            @cached_object("video-dataset-clips")
            def get_clip_paths(dataset_name: str) -> List[str]:
                root_path = dataset_root / dataset_name
                return [str(path.resolve()) for path in root_path.glob("**/*.mov")]
            ```

        Args:
            cache_key: The key to use for caching the file
            ext: The caching type to use (JSON or pickling)
            ignore: Should the cache be ignored?
            cache_obj: If set, keep the object around to avoid deserializing it
                when it is accessed again
        """

        self.cache_key = cache_key
        self.ext = ext
        self.obj = None
        self.ignore = ignore
        self.cache_obj = cache_obj

        assert ext in get_args(CacheType), f"Unexpected extension: {ext}"

    def __call__(self, func: Callable[..., Object]) -> Callable[..., Object]:
        """Returns a wrapped function that caches the return value.

        Args:
            func: The function to cache, which returns the object to load

        Returns:
            A cached version of the same function
        """

        @functools.wraps(func)
        def call_function_cached(*args: Any, **kwargs: Any) -> Object:
            if self.obj is not None:
                return self.obj

            keys: List[str] = []
            for arg in args:
                keys += [str(arg)]
            for key, val in sorted(kwargs.items()):
                keys += [f"{key}_{val}"]
            key = ".".join(keys)

            fpath = get_cache_dir() / self.cache_key / f"{key}.{self.ext}"

            if fpath.is_file() and not self.ignore:
                logger.debug("Loading cached object from %s", fpath)
                if self.ext == "json":
                    with open(fpath, "r", encoding="utf-8") as f:
                        return json.load(f)
                if self.ext == "pkl":
                    with open(fpath, "rb") as fb:
                        return pickle.load(fb)
                raise NotImplementedError(f"Can't load extension {self.ext}")

            obj = func(*args, **kwargs)
            if self.cache_obj:
                self.obj = obj

            logger.debug("Saving cached object to %s", fpath)
            fpath.parent.mkdir(exist_ok=True, parents=True)
            if self.ext == "json":
                with open(fpath, "w", encoding="utf-8") as f:
                    json.dump(obj, f)
                    return obj
            if self.ext == "pkl":
                with open(fpath, "wb") as fb:
                    pickle.dump(obj, fb)
                    return obj
            raise NotImplementedError(f"Can't save extension {self.ext}")

        return call_function_cached


class Index(Generic[Tk, Tv]):
    def __init__(self, items: Mapping[Tk, Sequence[Tv]]):
        """Indexes a dictionary with values that are lists.

        This lazily indexes all the values in the provided dictionary, flattens
        them out and allows them to be looked up by a specific index. This is
        analogous to PyTorch's ConcatDataset.

        Args:
            items: The dictionary to index
        """

        self.items = items

    @functools.cached_property
    def _item_list(self) -> List[Tuple[Tk, List[Tv]]]:
        return [(k, list(v)) for k, v in self.items.items()]

    @functools.cached_property
    def _indices(self) -> np.ndarray:
        return np.concatenate([np.array([0]), np.cumsum(np.array([len(i) for _, i in self._item_list]))])

    def __getitem__(self, index: int) -> Tuple[Tk, Tv]:
        a = np.searchsorted(self._indices, index, side="right") - 1
        b = index - self._indices[a]
        key, values = self._item_list[a]
        value = values[b]
        return key, value

    def __len__(self) -> int:
        return self._indices[-1].item()
