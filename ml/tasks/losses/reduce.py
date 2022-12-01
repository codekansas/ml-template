from typing import Literal, cast, get_args

from torch import Tensor, nn

ReduceType = Literal["mean", "sum", None]


def cast_reduce_type(s: str | None) -> ReduceType:
    args = get_args(ReduceType)
    assert s in args, f"Invalid reduce type: '{s}' Valid options are {args}"
    return cast(ReduceType, s)


def reduce(t: Tensor, reduce_type: ReduceType, dim: int) -> Tensor:
    if reduce_type is None:
        return t
    if reduce_type == "mean":
        return t.mean(dim)
    if reduce_type == "sum":
        return t.sum(dim)
    raise NotImplementedError(f"Unexpected sample reduction type: {reduce_type}")


class SampleReduce(nn.Module):
    def __init__(self, reduce_type: ReduceType) -> None:
        super().__init__()

        self.reduce_type = reduce_type

    def forward(self, loss: Tensor) -> Tensor:
        loss_flat = loss.flatten(1)
        return reduce(loss_flat, self.reduce_type, dim=1)
