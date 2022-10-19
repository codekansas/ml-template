from __future__ import annotations

import functools
from dataclasses import is_dataclass
from typing import Any, Callable, List, Literal

import numpy as np
import torch
import torchvision.transforms.functional as V
from PIL.Image import Image
from torch import Tensor

CollateMode = Literal["stack", "concat"]


def is_named_tuple(obj: Any) -> bool:
    return isinstance(obj, tuple) and hasattr(obj, "_asdict") and hasattr(obj, "_fields")


def collate(items: List[Any], *, mode: CollateMode | Callable[[List[Tensor]], Tensor] = "stack") -> Any | None:
    """Defines a general-purpose collating function.

    Args:
        items: The list of items to collate
        mode: Either `stack`, `concat`, or a custom function which is called on
            a list of tensors and returns a single tensor

    Returns:
        The collated item, or None if the item list was empty

    Raises:
        NotImplementedError: If the mode is invalid
    """

    if len(items) == 0:
        return None
    item = items[0]

    # Any None items should be filtered out.
    if item is None:
        return None

    # All Numpy arrays are converted to tensors.
    if isinstance(item, np.ndarray):
        return collate([torch.from_numpy(i) for i in items], mode=mode)

    # All images are converted to tensors.
    if isinstance(item, Image):
        return collate([V.to_tensor(i) for i in items], mode=mode)

    # Numbers are converted to a list of tensors.
    if isinstance(item, bool):
        return collate([torch.BoolTensor([i]) for i in items], mode=mode)
    if isinstance(item, int):
        return collate([torch.IntTensor([i]) for i in items], mode=mode)
    if isinstance(item, float):
        return collate([torch.FloatTensor([i]) for i in items], mode=mode)

    # Tensors are either concatenated or stacked.
    if isinstance(item, Tensor):
        if callable(mode):
            return mode(items)
        if isinstance(mode, str):
            if mode == "stack":
                return torch.stack(items, dim=0)
            if mode == "concat":
                return torch.cat(items, dim=0)
            raise NotImplementedError(f"Invalid collate mode: {mode}")
        raise NotImplementedError(f"Invalid mode type: {type(mode)}")

    # Collate dictionaries if they have the same keys.
    if isinstance(item, dict) and all(set(i.keys()) == set(item.keys()) for i in items):
        output_dict = {}
        item_keys = set(item.keys())
        for key in item_keys:
            output_dict[key] = collate([i[key] for i in items], mode=mode)
        return output_dict

    # Collate lists and tuples if they have the same lengths.
    if isinstance(item, (list, tuple)) and all(len(i) == len(item) for i in items):
        output_list = []
        for j in range(len(item)):
            output_list.append(collate([i[j] for i in items], mode=mode))
        if is_named_tuple(item):
            return type(item)(*output_list)  # type: ignore
        if isinstance(item, tuple):
            return tuple(output_list)
        return output_list

    # Handles dataclasses.
    if is_dataclass(item):
        output_dict = {}
        item_keys = item.__dict__.keys()
        for key in item_keys:
            output_dict[key] = collate([getattr(i, key) for i in items], mode=mode)
        return item.__class__(**output_dict)

    # By default, don't do anything.
    return items


def collate_non_null(items: List[Any], *, mode: CollateMode | Callable[[List[Tensor]], Tensor] = "stack") -> Any:
    collated = collate(items, mode=mode)
    assert collated is not None
    return collated


def pad_sequence(
    tensors: List[Tensor],
    *,
    dim: int = 0,
    max_length: int | None = None,
    left_pad: bool = False,
    left_truncate: bool = False,
    pad_value: int | float | bool = 0,
) -> List[Tensor]:
    """Pads or truncates a sequence of tensors to the same length.

    Args:
        tensors: The tensors to pad or truncate
        dim: The dimension to pad or truncate
        max_length: The maximum tensor length
        left_pad: If set, pad on the left side, otherwise pad the right side
        left_truncate: If set, truncate on the left side, otherwise truncate
            on the right side
        pad_value: The padding value to use

    Returns:
        The padded or truncated tensors

    Raises:
        ValueError: If the tensor dimensions are invalid
    """

    if not tensors:
        return tensors

    num_dims = tensors[0].dim()
    if num_dims == 0:
        raise ValueError("Tensor dimensions must be greater than zero")
    if not all(t.dim() == num_dims for t in tensors):
        tensor_dims = {t.dim() for t in tensors}
        raise ValueError(f"All tensors should have the same number of dimensions; got {tensor_dims}")

    dim = dim if dim >= 0 else num_dims + dim
    target_length = int(max(t.size(dim) for t in tensors))
    if max_length is not None:
        target_length = min(target_length, max_length)

    def pad_tensor(t: Tensor) -> Tensor:
        length = t.size(dim)
        if length > target_length:
            t = torch.narrow(t, dim, length - target_length if left_truncate else 0, target_length)
        elif length < target_length:
            padding_shape = [target_length - s if i == dim else s for i, s in enumerate(t.shape)]
            padding = t.new_full(padding_shape, fill_value=pad_value)
            t = torch.cat((padding, t) if left_pad else (t, padding), dim=dim)
        return t

    return list(map(pad_tensor, tensors))


def get_collate_fn(*, mode: str | Callable[[List[Tensor]], Tensor] = "stack") -> Callable[[List[Any]], Any]:
    return functools.partial(collate, mode=mode)
