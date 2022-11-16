"""Defines general-purpose helper functions for initializing norm layers.

Some pointers:

- For networks which need to run efficiently at inference time, batch norm
  is usually a good idea, since it can be fused with the weights of the
  convolutional neural network. Similarly, weight norm also achieves this
  (see `nn.utils.weight_norm`).

This documentation should be updated with a better explanation of different
types of normalization functions. My usual approach is to just throw everything
at the wall and see what sticks.
"""


from typing import Literal, Optional, cast, get_args

import torch
from torch import Tensor, nn

NormType = Literal[
    "no_norm",
    "batch",
    "batch_affine",
    "instance",
    "instance_affine",
    "group",
    "group_affine",
    "layer",
    "layer_affine",
]


def cast_norm_type(s: str) -> NormType:
    args = get_args(NormType)
    assert s in args, f"Invalid norm type: '{s}' Valid options are {args}"
    return cast(NormType, s)


class LastBatchNorm(nn.Module):
    """Applies batch norm along final dimension without transposing the tensor.

    The normalization is pretty simple, it basically just tracks the running
    mean and variance for each channel, then normalizes each channel to have
    a unit normal distribution.

    Input:
        x: Tensor with shape (..., N)

    Output:
        The tensor, normalized by the running mean and variance
    """

    __constants__ = ["channels", "momentum", "affine", "eps"]

    mean: Tensor
    var: Tensor

    def __init__(
        self,
        channels: int,
        momentum: float = 0.99,
        affine: bool = True,
        eps: float = 1e-4,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        self.channels = channels
        self.momentum = momentum
        self.affine = affine
        self.eps = eps

        if dtype is None:
            mean_tensor = torch.zeros(channels, device=device)
            var_tensor = torch.ones(channels, device=device)
        else:
            mean_tensor = torch.zeros(channels, device=device, dtype=dtype)
            var_tensor = torch.ones(channels, device=device, dtype=dtype)
        self.register_buffer("mean", mean_tensor)
        self.register_buffer("var", var_tensor)

        if self.affine:
            self.affine_transform = nn.Linear(channels, channels, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        if self.affine:
            x = self.affine_transform(x)
        if self.training:
            x_flat = x.flatten(0, -2)
            mean, var = x_flat.mean(dim=0).detach(), x_flat.var(dim=0).detach()
            new_mean = mean * (1 - self.momentum) + self.mean * self.momentum
            new_var = var * (1 - self.momentum) + self.var * self.momentum
            x_out = (x - new_mean.expand_as(x)) / (new_var.expand_as(x) + self.eps)
            self.mean.copy_(new_mean, non_blocking=True)
            self.var.copy_(new_var, non_blocking=True)
        else:
            x_out = (x - self.mean.expand_as(x)) / (self.var.expand_as(x) + self.eps)
        return x_out


class ConvLayerNorm(nn.Module):
    __constants__ = ["channels", "eps", "elementwise_affine", "static_shape"]

    def __init__(
        self,
        channels: int,
        *,
        dims: int | None = None,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.channels = channels
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            if dtype is None:
                self.weight = nn.Parameter(torch.empty(self.channels, device=device))
                self.bias = nn.Parameter(torch.empty(self.channels, device=device))
            else:
                self.weight = nn.Parameter(torch.empty(self.channels, device=device, dtype=dtype))
                self.bias = nn.Parameter(torch.empty(self.channels, device=device, dtype=dtype))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.static_shape = None if dims is None else (1, -1) + (1,) * dims

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        mean = inputs.mean(dim=1, keepdim=True)
        var = torch.square(inputs - mean).mean(dim=1, keepdim=True)
        normalized_inputs = (inputs - mean) / (var + self.eps).sqrt()
        if self.elementwise_affine:
            if self.static_shape is None:
                weight = self.weight.unflatten(0, (-1,) + (1,) * (len(inputs.shape) - 2))
                bias = self.bias.unflatten(0, (-1,) + (1,) * (len(inputs.shape) - 2))
            else:
                weight = self.weight.view(self.static_shape)
                bias = self.bias.view(self.static_shape)
            normalized_inputs = normalized_inputs * weight + bias
        return normalized_inputs


def get_norm_1d(
    norm: NormType,
    *,
    dim: Optional[int] = None,
    groups: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> nn.Module:
    if norm == "no_norm":
        return nn.Identity()
    if norm == "batch":
        if dim is None:
            return nn.LazyBatchNorm1d(affine=False, device=device, dtype=dtype)
        return nn.BatchNorm1d(dim, affine=False, device=device, dtype=dtype)
    if norm == "batch_affine":
        if dim is None:
            return nn.LazyBatchNorm1d(affine=True, device=device, dtype=dtype)
        return nn.BatchNorm1d(dim, affine=True, device=device, dtype=dtype)
    if norm == "instance":
        if dim is None:
            return nn.LazyInstanceNorm1d(affine=False, device=device, dtype=dtype)
        return nn.InstanceNorm1d(dim, affine=False, device=device, dtype=dtype)
    if norm == "instance_affine":
        if dim is None:
            return nn.LazyInstanceNorm1d(affine=True, device=device, dtype=dtype)
        return nn.InstanceNorm1d(dim, affine=True, device=device, dtype=dtype)
    if norm in ("group", "group_affine"):
        assert dim is not None, "`dim` is required for group norm"
        assert groups is not None, "`groups` is required for group norm"
        return nn.GroupNorm(groups, dim, affine=norm == "group_affine", device=device, dtype=dtype)
    if norm in ("layer", "layer_affine"):
        assert dim is not None
        return ConvLayerNorm(dim, dims=1, elementwise_affine=norm == "layer_affine", device=device, dtype=dtype)
    raise NotImplementedError(f"Invalid 1D norm type: {norm}")


def get_norm_2d(
    norm: NormType,
    *,
    dim: Optional[int] = None,
    groups: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> nn.Module:
    if norm == "no_norm":
        return nn.Identity()
    if norm == "batch":
        if dim is None:
            return nn.LazyBatchNorm2d(affine=False, device=device, dtype=dtype)
        return nn.BatchNorm2d(dim, affine=False, device=device, dtype=dtype)
    if norm == "batch_affine":
        if dim is None:
            return nn.LazyBatchNorm2d(affine=True, device=device, dtype=dtype)
        return nn.BatchNorm2d(dim, affine=True, device=device, dtype=dtype)
    if norm == "instance":
        if dim is None:
            return nn.LazyInstanceNorm2d(affine=False, device=device, dtype=dtype)
        return nn.InstanceNorm2d(dim, affine=False, device=device, dtype=dtype)
    if norm == "instance_affine":
        if dim is None:
            return nn.LazyInstanceNorm2d(affine=True, device=device, dtype=dtype)
        return nn.InstanceNorm2d(dim, affine=True, device=device, dtype=dtype)
    if norm in ("group", "group_affine"):
        assert dim is not None, "`dim` is required for group norm"
        assert groups is not None, "`groups` is required for group norm"
        return nn.GroupNorm(groups, dim, affine=norm == "group_affine", device=device, dtype=dtype)
    if norm in ("layer", "layer_affine"):
        assert dim is not None
        return ConvLayerNorm(dim, dims=2, elementwise_affine=norm == "layer_affine", device=device, dtype=dtype)
    raise NotImplementedError(f"Invalid 2D norm type: {norm}")


def get_norm_3d(
    norm: NormType,
    *,
    dim: Optional[int] = None,
    groups: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> nn.Module:
    if norm == "no_norm":
        return nn.Identity()
    if norm == "batch":
        if dim is None:
            return nn.LazyBatchNorm3d(affine=False, device=device, dtype=dtype)
        return nn.BatchNorm3d(dim, affine=False, device=device, dtype=dtype)
    if norm == "batch_affine":
        if dim is None:
            return nn.LazyBatchNorm3d(affine=True, device=device, dtype=dtype)
        return nn.BatchNorm3d(dim, affine=True, device=device, dtype=dtype)
    if norm == "instance":
        if dim is None:
            return nn.LazyInstanceNorm3d(affine=False, device=device, dtype=dtype)
        return nn.InstanceNorm3d(dim, affine=False, device=device, dtype=dtype)
    if norm == "instance_affine":
        if dim is None:
            return nn.LazyInstanceNorm3d(affine=True, device=device, dtype=dtype)
        return nn.InstanceNorm3d(dim, affine=True, device=device, dtype=dtype)
    if norm in ("group", "group_affine"):
        assert dim is not None, "`dim` is required for group norm"
        assert groups is not None, "`groups` is required for group norm"
        return nn.GroupNorm(groups, dim, affine=norm == "group_affine", device=device, dtype=dtype)
    if norm in ("layer", "layer_affine"):
        assert dim is not None
        return ConvLayerNorm(dim, dims=1, elementwise_affine=norm == "layer_affine", device=device, dtype=dtype)
    raise NotImplementedError(f"Invalid 3D norm type: {norm}")


def get_norm_linear(
    norm: NormType,
    *,
    dim: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> nn.Module:
    if norm == "no_norm":
        return nn.Identity()
    if norm == "batch":
        assert dim is not None, "`dim` is required for batch norm"
        return LastBatchNorm(dim, affine=False, device=device, dtype=dtype)
    if norm == "batch_affine":
        assert dim is not None, "`dim` is required for batch norm"
        return LastBatchNorm(dim, affine=True, device=device, dtype=dtype)
    if norm == "layer":
        assert dim is not None, "`dim` is required for layer norm"
        return nn.LayerNorm(dim, elementwise_affine=False, device=device, dtype=dtype)
    if norm == "layer_affine":
        assert dim is not None, "`dim` is required for layer norm"
        return nn.LayerNorm(dim, elementwise_affine=True, device=device, dtype=dtype)
    raise NotImplementedError(f"Invalid linear norm type: {norm}")
