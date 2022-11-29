import itertools
import logging
import time

import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, IterableDataset

from ml.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def test_dataset(ds: Dataset | IterableDataset | DataLoader, max_samples: int = 3) -> None:
    """Iterates through a dataset.

    Args:
        ds: The dataset to iterate through
        max_samples: Maximum number of samples to loop through
    """

    configure_logging(use_tqdm=True)
    start_time = time.time()

    if isinstance(ds, (IterableDataset, DataLoader)):
        logger.info("Iterating samples in %s", "dataloader" if isinstance(ds, DataLoader) else "dataset")
        for i, _ in enumerate(itertools.islice(ds, max_samples)):
            if i % 10 == 0:
                logger.info("Sample %d in %.2g seconds", i, time.time() - start_time)
    else:
        samples = len(ds)  # type: ignore
        logger.info("Dataset has %d items", samples)
        for i in tqdm.trange(min(samples, max_samples)):
            _ = ds[i]
            if i % 10 == 0:
                logger.info("Sample %d in %.2g seconds", i, time.time() - start_time)
