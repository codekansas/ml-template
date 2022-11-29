from dataclasses import dataclass

from torch.utils.data.dataloader import get_worker_info as _get_worker_info_base


@dataclass
class WorkerInfo:
    worker_id: int
    num_workers: int
    in_worker: bool


def get_worker_info() -> WorkerInfo:
    """Gets a typed worker info object which always returns a value.

    Returns:
        The typed worker info object
    """

    if (worker_info := _get_worker_info_base()) is None:
        return WorkerInfo(
            worker_id=0,
            num_workers=1,
            in_worker=False,
        )

    return WorkerInfo(
        worker_id=worker_info.id,
        num_workers=worker_info.num_workers,
        in_worker=True,
    )
