import random
from typing import Optional

import numpy as np
import torch

from ml.core.env import get_env_random_seed


def set_random_seed(seed: Optional[int] = None, offset: int = 0) -> None:
    if seed is None:
        seed = get_env_random_seed()
    seed += offset
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
