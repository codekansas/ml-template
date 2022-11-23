import logging

from omegaconf import DictConfig

from ml.core.registry import register_trainer

logger = logging.getLogger(__name__)


def mp_train_main(config: DictConfig) -> None:
    """Runs the training loop in a subprocess."""  # noqa

    trainer = register_trainer.build_entry(config)
    assert trainer is not None, "Trainer is required to launch multiprocessing jobs"
    trainer.launch()
