"""Defines an API for the most commonly used components.

This lets you just have a single import like `from ml import api` then use,
for example, `api.conf_field(...)` to access various components.

There wasn't an explicit dictating which functions to include here, it was just
whichever functions seemed like they would be generally useful to have on
hand quickly.
"""

# flake8: noqa
# pylint: disable=unused-import

from ml.core.config import conf_field
from ml.core.env import (
    get_cache_dir,
    get_data_dir,
    get_eval_dir,
    get_exp_name,
    get_log_dir,
    get_model_dir,
)
from ml.core.registry import (
    register_logger,
    register_lr_scheduler,
    register_model,
    register_optimizer,
    register_task,
    register_trainer,
)
from ml.core.state import Phase, State
from ml.core.types import Batch, Loss, Output
from ml.lr_schedulers.base import BaseLRScheduler, BaseLRSchedulerConfig
from ml.models.activations import cast_activation_type, get_activation
from ml.models.base import BaseModel, BaseModelConfig
from ml.models.init import cast_init_type, init_
from ml.models.norms import (
    cast_norm_type,
    get_norm_1d,
    get_norm_2d,
    get_norm_3d,
    get_norm_linear,
)
from ml.optimizers.base import BaseOptimizer, BaseOptimizerConfig
from ml.tasks.base import BaseTask, BaseTaskConfig
from ml.trainers.base import BaseTrainer, BaseTrainerConfig
from ml.utils.argparse import from_args, get_args, get_type_from_string
from ml.utils.atomic import atomic_save, open_atomic
from ml.utils.augmentation import get_image_mask
from ml.utils.caching import cached_object
from ml.utils.checks import assert_no_nans
from ml.utils.colors import colorize
from ml.utils.datetime import format_timedelta
from ml.utils.distributed import (
    get_master_addr,
    get_master_port,
    get_rank,
    get_rank_optional,
    get_world_size,
    get_world_size_optional,
    is_distributed,
    is_master,
)
from ml.utils.large_models import init_empty_weights, meta_to_empty_func
from ml.utils.timer import Timer
from ml.utils.video import READERS as VIDEO_READERS, WRITERS as VIDEO_WRITERS
