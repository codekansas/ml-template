from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch import Tensor
from torch.utils.data.dataset import IterableDataset

from ml.utils.video import READERS, Reader, VideoProps


class VideoFileDataset(IterableDataset[Tensor]):
    def __init__(self, file_path: str | Path, reader: Reader = "ffmpeg") -> None:
        """Defines a dataset which iterates through frames in a video file.

        Args:
            file_path: The path to the video file to iterate through
            reader: The video reader to use
        """

        super().__init__()

        assert reader in READERS, f"Unsupported {reader=}"

        self.file_path = str(file_path)
        self.reader = reader

    video_props: VideoProps
    video_stream: Iterator[np.ndarray]

    def __iter__(self) -> Iterator[Tensor]:
        self.video_props = VideoProps.from_file_ffmpeg(self.file_path)
        self.video_stream = READERS[self.reader](self.file_path)
        return self

    def __next__(self) -> Tensor:
        buffer = next(self.video_stream)
        return torch.from_numpy(buffer).permute(2, 0, 1)  # HWC -> CHW
