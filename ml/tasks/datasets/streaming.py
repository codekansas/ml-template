import logging
import random
from typing import Collection, Dict, Generic, Iterator, List, Optional, Tuple, TypeVar

from torch.utils.data.dataloader import get_worker_info
from torch.utils.data.dataset import IterableDataset

logger = logging.getLogger(__name__)

Batch = TypeVar("Batch")


class StreamingDataset(IterableDataset[Tuple[int, Batch]], Generic[Batch]):
    def __init__(self, datasets: Collection[IterableDataset[Batch]], max_simultaneous: int) -> None:
        """Defines a dataset which combines many streaming datasets.

        This dataset takes a set of child iterable datasets and iterates from
        them infinitely. When a child dataset is exhausted, it is returned to
        the reservoir and restarted, while another dataset is chosen.

        An example usage for this dataset is to get samples from many videos,
        where each sub-dataset yields video samples. This way the child dataset
        can be used to run inference on a single video, while the parent
        streaming dataset can be used to train on a mixture of videos. The
        child dataset can then be optimized to make video loading times fast.

        Args:
            datasets: The sub-datasets to iterate from
            max_simultaneous: The maximum number of simultaneous datasets to
                iterate from. Increasing this number increases the dataset
                diversity but also increases memory usage as samples need to be
                stored in memory

        Raises:
            ValueError: If no datasets are provided
        """

        super().__init__()

        if len(datasets) == 0:
            raise ValueError("Must provide at least one dataset")

        self.datasets = list(datasets)
        self.max_simultaneous = max_simultaneous

    worker_datasets: Dict[int, IterableDataset[Batch]]
    iterators: Dict[int, Iterator[Batch]]
    reservoir: List[int]
    reservoir_pointer: int

    def __iter__(self) -> Iterator[Tuple[int, Batch]]:
        worker_info = get_worker_info()
        dataset_ids = list(range(len(self.datasets)))
        if worker_info is None:
            worker_id = 0
        else:
            worker_id = worker_info.id
            dataset_ids = dataset_ids[worker_id :: worker_info.num_workers]

        # Gets the subset of worker dataset for this iterator.
        self.worker_datasets = {i: self.datasets[i] for i in dataset_ids}
        if len(self.worker_datasets) == 0:
            raise ValueError(f"Worker {worker_id} doesn't have any datasets; consider reducing the worker count")

        # Creates a reservoir of available IDs, and a dict of active iterators.
        self.iterators = {}
        self.reservoir = list(dataset_ids)
        random.shuffle(self.reservoir)
        self.reservoir_pointer = 0

        return self

    def swap_reservoir(self, a: int, b: int) -> None:
        self.reservoir[a], self.reservoir[b] = self.reservoir[b], self.reservoir[a]

    def fill_reservoir(self) -> None:
        while self.reservoir_pointer < min(self.max_simultaneous, len(self.reservoir)):
            new_iter_id = random.randint(self.reservoir_pointer, len(self.reservoir) - 1)
            self.swap_reservoir(new_iter_id, self.reservoir_pointer)
            self.reservoir_pointer += 1

    def sample_reservoir_id(self) -> int:
        return random.randint(0, self.reservoir_pointer - 1)

    def return_dataset(self, reservoir_id: int) -> None:
        assert reservoir_id < self.reservoir_pointer
        dataset_id = self.reservoir[reservoir_id]
        if dataset_id in self.iterators:
            self.iterators.pop(dataset_id)
        self.swap_reservoir(reservoir_id, self.reservoir_pointer - 1)
        self.reservoir_pointer -= 1

    def __next__(self) -> Tuple[int, Batch]:
        dataset_id: Optional[int] = None
        sample: Optional[Batch] = None
        while dataset_id is None or sample is None:
            self.fill_reservoir()
            reservoir_id = self.sample_reservoir_id()
            dataset_id = self.reservoir[reservoir_id]
            if dataset_id not in self.iterators:
                self.iterators[dataset_id] = iter(self.worker_datasets[dataset_id])
            try:
                sample = next(self.iterators[dataset_id])
            except StopIteration:
                logger.debug("Finished one iteration for dataset %d", dataset_id)
                self.return_dataset(reservoir_id)
        return dataset_id, sample
