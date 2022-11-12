import logging
import os

import torch

from ml.trainers.mixins.device.base import BaseDevice

logger = logging.getLogger(__name__)


def get_env_bool(key: str) -> bool:
    val = int(os.environ.get(key, 0))
    assert val in (0, 1), f"Invalid value for {key}: {val}"
    return val == 1


class GPUDevice(BaseDevice):
    """Mixin to support single-GPU training."""

    @classmethod
    def has_device(cls) -> bool:
        return torch.cuda.is_available() and torch.cuda.device_count() > 0

    @classmethod
    def get_device(cls) -> torch.device:
        return torch.device("cuda", 0)

    @classmethod
    def get_floating_point_type(cls) -> torch.dtype:
        use_fp32 = get_env_bool("USE_FP32")
        use_fp64 = get_env_bool("USE_FP64")
        if use_fp64:
            logger.info("Using FP64")
            return torch.float64
        elif use_fp32:
            logger.info("Using FP32")
            return torch.float32
        else:
            return torch.float16
            # return torch.bfloat16
