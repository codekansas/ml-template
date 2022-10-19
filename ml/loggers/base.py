from __future__ import annotations

import datetime
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Generic, List, Tuple, TypeVar

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ml.core.config import BaseConfig, BaseObjectWithPointers, conf_field
from ml.core.state import Phase, State

LogT = TypeVar("LogT")

DEFAULT_NAMESPACE = "value"
VALID_CHANNEL_COUNTS = {1, 3}
TARGET_FPS = 12


def get_scalar(scalar: int | float | Tensor) -> int | float:
    if isinstance(scalar, Tensor):
        return scalar.detach().cpu().item()
    return scalar


@dataclass
class BaseLoggerConfig(BaseConfig):
    write_every_n_seconds: int = conf_field(1, help="Only write a log line every N seconds")


LoggerConfigT = TypeVar("LoggerConfigT", bound=BaseLoggerConfig)


class BaseLogger(BaseObjectWithPointers[LoggerConfigT], Generic[LoggerConfigT], ABC):
    """Defines the base logger."""

    log_directory: Path

    def __init__(self, config: LoggerConfigT) -> None:
        super().__init__(config)

        self.start_time = datetime.datetime.now()
        self.last_write_time: Dict[Phase, datetime.datetime] = {}

    def initialize(self, log_directory: Path) -> None:
        self.log_directory = log_directory

    def get_cpu_array(self, value: np.ndarray | Tensor) -> Tensor:
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value)
        return value.detach().cpu()

    def log_scalar(self, key: str, value: int | float | Tensor, state: State, namespace: str) -> None:
        """Logs a scalar value.

        Args:
            key: The key to log
            value: The value to log
            state: The current log state
            namespace: The namespace to be logged
        """

    def log_string(self, key: str, value: str, state: State, namespace: str) -> None:
        """Logs a string value.

        Args:
            key: The key to log
            value: The value to log
            state: The current log state
            namespace: The namespace to be logged
        """

    def log_image(self, key: str, value: Tensor, state: State, namespace: str) -> None:
        """Logs a normalized image, with shape (C, H, W).

        Args:
            key: The key to log
            value: The value to log
            state: The current log state
            namespace: The namespace to be logged
        """

    def log_video(self, key: str, value: Tensor, state: State, namespace: str) -> None:
        """Logs a normalized video, with shape (T, C, H, W).

        Args:
            key: The key to log
            value: The value to log
            state: The current log state
            namespace: The namespace to be logged
        """

    def log_histogram(self, key: str, value: Tensor, state: State, namespace: str) -> None:
        """Logs a histogram, with any shape.

        Args:
            key: The key to log
            value: The value to log
            state: The current log state
            namespace: The namespace to be logged
        """

    def log_point_cloud(self, key: str, value: Tensor, state: State, namespace: str) -> None:
        """Logs a normalized point cloud, with shape (B, N, 3).

        Args:
            key: The key to log
            value: The value to log
            state: The current log state
            namespace: The namespace to be logged
        """

    def should_write(self, state: State) -> bool:
        """Returns whether or not the current state should be written.

        This function checks that the last time the current phase was written
        was greater than some interval in the past, to avoid writing tons of
        values when the iteration time is extremely small.

        Args:
            state: The state to check

        Returns:
            If the logger should write values for the current state
        """

        current_time = datetime.datetime.now()
        min_write_time_diff = datetime.timedelta(seconds=self.config.write_every_n_seconds)

        if state.phase not in self.last_write_time:
            self.last_write_time[state.phase] = current_time
            return True
        elif current_time - self.last_write_time[state.phase] < min_write_time_diff:
            return False
        else:
            self.last_write_time[state.phase] = current_time
            return True

    @abstractmethod
    def write(self, state: State) -> None:
        """Writes the logs.

        Args:
            state: The current log state
        """

    @abstractmethod
    def clear(self, state: State) -> None:
        """Clears the logs.

        Args:
            state: The current log state
        """


def _aminmax(t: Tensor) -> Tuple[Tensor, Tensor]:
    # `aminmax` isn't supported for MPS tensors, fall back to separate calls.
    minv, maxv = (t.min(), t.max()) if t.is_mps else tuple(t.aminmax())
    return minv, maxv


def standardize_image(image: Tensor, *, log_key: str | None = None, normalize: bool = True) -> Tensor:
    """Converts an arbitrary image to shape (C, H, W).

    Args:
        image: The image tensor to log
        log_key: An optional logging key to use in the exception message
        normalize: Normalize images to (0, 1)

    Returns:
        The normalized image, with shape (C, H, W)

    Raises:
        ValueError: If the image shape is invalid
    """

    if normalize and image.is_floating_point():
        minv, maxv = _aminmax(image)
        maxv.clamp_min_(1.0)
        minv.clamp_max_(0.0)
        image = torch.clamp((image.detach() - minv) / (maxv - minv), 0.0, 1.0)

    if image.ndim == 2:
        return image.unsqueeze(0)
    if image.ndim == 3:
        if image.shape[0] in VALID_CHANNEL_COUNTS:
            return image
        if image.shape[2] in VALID_CHANNEL_COUNTS:
            return image.permute(2, 0, 1)
    raise ValueError(f"Invalid image shape{'' if log_key is None else f' for {log_key}'}: {image.shape}")


def standardize_images(
    images: Tensor,
    *,
    max_images: int | None = None,
    log_key: str | None = None,
    normalize: bool = True,
) -> Tensor:
    """Converts an arbitrary set of images to shape (B, C, H, W).

    Args:
        images: The image tensor to log
        max_images: Maximum number of images to select
        log_key: An optional logging key to use in the exception message
        normalize: Normalize images to (0, 1)

    Returns:
        The normalized image, with shape (B, C, H, W)

    Raises:
        ValueError: If the image shape is invalid
    """

    if normalize and images.is_floating_point():
        minv, maxv = _aminmax(images)
        maxv.clamp_min_(1.0)
        minv.clamp_max_(0.0)
        images = torch.clamp((images.detach() - minv) / (maxv - minv), 0.0, 1.0)

    if images.ndim == 3:
        return images.unsqueeze(1)
    if images.ndim == 4:
        if images.shape[1] in VALID_CHANNEL_COUNTS:
            return images if max_images is None else images[:max_images]
        if images.shape[3] in VALID_CHANNEL_COUNTS:
            images = images.permute(0, 3, 1, 2)
            return images if max_images is None else images[:max_images]
    raise ValueError(f"Invalid image shape{'' if log_key is None else f' for {log_key}'}: {images.shape}")


def standardize_video(video: Tensor, *, log_key: str | None = None, normalize: bool = True) -> Tensor:
    """Converts an arbitrary video to shape (T, C, H, W).

    Args:
        video: The video tensor to log
        log_key: An optional logging key to use in the exception message
        normalize: Normalize images to (0, 1)

    Returns:
        The normalized video, with shape (T, C, H, W)

    Raises:
        ValueError: If the video shape is invalid
    """

    if normalize and video.is_floating_point():
        minv, maxv = _aminmax(video[-1])
        maxv.clamp_min_(1.0)
        minv.clamp_max_(0.0)
        video = torch.clamp((video.detach() - minv) / (maxv - minv), 0.0, 1.0)

    if video.ndim == 3:
        return video.unsqueeze(1)
    if video.ndim == 4:
        if video.shape[1] in VALID_CHANNEL_COUNTS:
            return video
        if video.shape[3] in VALID_CHANNEL_COUNTS:
            return video.permute(0, 3, 1, 2)
    raise ValueError(f"Invalid video shape{'' if log_key is None else f' for {log_key}'}: {video.shape}")


def standardize_videos(
    videos: Tensor,
    *,
    max_videos: int | None = None,
    log_key: str | None = None,
    normalize: bool = True,
) -> Tensor:
    """Converts an arbitrary video to shape (B, T, C, H, W).

    Args:
        videos: The video tensor to log
        max_videos: Maximum number of images to select
        log_key: An optional logging key to use in the exception message
        normalize: Normalize images to (0, 1)

    Returns:
        The normalized video, with shape (B, T, C, H, W)

    Raises:
        ValueError: If the video shape is invalid
    """

    if normalize and videos.is_floating_point():
        minv, maxv = _aminmax(videos[:, -1])
        maxv.clamp_min_(1.0)
        minv.clamp_max_(0.0)
        videos = torch.clamp((videos.detach() - minv) / (maxv - minv), 0.0, 1.0)

    if videos.ndim == 4:
        return videos.unsqueeze(2)
    if videos.ndim == 5:
        if videos.shape[2] in VALID_CHANNEL_COUNTS:
            return videos if max_videos is None else videos[:max_videos]
        if videos.shape[4] in VALID_CHANNEL_COUNTS:
            videos = videos.permute(0, 3, 1, 2)
            return videos if max_videos is None else videos[:max_videos]
    raise ValueError(f"Invalid video shape{'' if log_key is None else f' for {log_key}'}: {videos.shape}")


def normalize_fps(
    video: Tensor | List[Tensor],
    fps: int | None,
    length: int | None,
    stack_dim: int = 0,
    target_fps: int = TARGET_FPS,
) -> Tensor:
    """Normalizes a video to have a particular FPS.

    Args:
        video: The video to normalize, with shape (T, C, H, W)
        fps: The desired frames per second
        length: The desired video length, in seconds, at the target FPS
        target_fps: The target frames per second for the logger
        stack_dim: Which dimension to stack along, for lists

    Returns:
        The normalized video
    """

    if fps is None and length is None:
        return torch.stack(video, dim=stack_dim) if isinstance(video, list) else video

    pre_frames = len(video) if isinstance(video, list) else video.size(0)
    if fps is None:
        assert length is not None  # Not used, just for type checker
        fps = int(pre_frames / length)

    post_frames = int(pre_frames * (target_fps / fps))

    if isinstance(video, list):
        frame_ids = torch.linspace(0, pre_frames - 1, post_frames).long()
        return torch.stack([video[i] for i in frame_ids], dim=stack_dim)

    frame_ids = torch.linspace(0, pre_frames - 1, post_frames, device=video.device).long()
    return video[frame_ids]


def standardize_point_cloud(value: Tensor, max_points: int, *, log_key: str | None) -> Tensor:
    for i in range(0, value.ndim - 1):
        if value.shape[i] == 3:
            value = value.transpose(i, -1)
            break
    if value.shape[-1] != 3:
        raise ValueError(f"Invalid point cloud shape{'' if log_key is None else f' for {log_key}'}: {value.shape}")
    if value.ndim == 2:
        value = value.unsqueeze(0)
    elif value.ndim > 3:
        value = value.flatten(1, -2)
    if value.shape[1] > max_points:
        indices = torch.multinomial(torch.ones(value.shape[1], device=value.device), max_points)
        value = value[:, indices]
    return value


def make_square_image_or_video(
    images_or_videos: Tensor,
    *,
    sep: int = 0,
    squareness_weight: float = 1.0,
    emptiness_weight: float = 1.0,
) -> Tensor:
    """Makes a square image by concatenating all the child images.

    This does a simple ternary search to minimize a squareness penalty and an
    emptiness penalty (i.e., the resulting image should be mostly filled in
    and also approximately square).

    Args:
        images_or_videos: The images tensor, with shape (B, C, H, W) or
            (B, T, C, H, W)
        sep: Some optional padding around the images
        squareness_weight: Weight for number of non-square pixels in penalty
        emptiness_weight: Weight for number of empty pixels in penalty

    Returns:
        The square image, with shape (C, H', W') or (T, C, H', W')
    """

    assert images_or_videos.dim() in (4, 5)

    def ternary_search_optimal_side_counts(height: int, width: int, count: int) -> Tuple[int, int]:
        lo, hi = 1, count

        def squareness_penalty(val: int) -> float:
            h, w = val * height, ((count + val - 1) // val) * width
            return (h * w) - min(h, w) ** 2

        def emptiness_penalty(val: int) -> float:
            h, w = val * height, ((count + val - 1) // val) * width
            return (h * w) - (height * width * count)

        def penalty(val: int) -> float:
            return squareness_penalty(val) * squareness_weight + emptiness_penalty(val) * emptiness_weight

        # Runs ternary search to minimize penalty.
        while lo < hi - 2:
            lmid, rmid = (lo * 2 + hi) // 3, (lo + hi * 2) // 3
            if penalty(lmid) > penalty(rmid):
                lo = lmid
            else:
                hi = rmid

        # Returns the lowest-penalty configuration.
        mid = (lo + hi) // 2
        plo, pmid, phi = penalty(lo), penalty(mid), penalty(hi)
        if pmid <= plo and pmid <= phi:
            return mid, (count + mid - 1) // mid
        elif plo <= phi:
            return lo, (count + lo - 1) // lo
        else:
            return hi, (count + hi - 1) // hi

    height, width = images_or_videos.shape[-2:]
    image_list = list(torch.unbind(images_or_videos, dim=0))
    hside, wside = ternary_search_optimal_side_counts(height, width, len(image_list))

    image_list = image_list + [torch.zeros_like(images_or_videos[0])] * (hside * wside - len(image_list))
    a, b = sep // 2, (sep + 1) // 2
    image_list = [F.pad(image, (a, b, a, b)) for image in image_list]
    wconcat = [torch.cat(image_list[i : i + wside], dim=-1) for i in range(0, len(image_list), wside)]
    new_image = torch.cat(wconcat, dim=-2)
    return new_image[..., a : new_image.shape[-2] - b, a : new_image.shape[-1] - b]


class MultiLogger:
    """Defines an intermediate container which holds values to log somewhere else."""

    def __init__(self, default_namespace: str = DEFAULT_NAMESPACE) -> None:
        self.scalars: Dict[str, Dict[str, int | float | Tensor]] = defaultdict(dict)
        self.strings: Dict[str, Dict[str, str]] = defaultdict(dict)
        self.images: Dict[str, Dict[str, Tensor]] = defaultdict(dict)
        self.videos: Dict[str, Dict[str, Tensor]] = defaultdict(dict)
        self.histograms: Dict[str, Dict[str, Tensor]] = defaultdict(dict)
        self.point_clouds: Dict[str, Dict[str, Tensor]] = defaultdict(dict)
        self.default_namespace = default_namespace

    def resolve_namespace(self, namespace: str | None = None) -> str:
        return self.default_namespace if namespace is None else namespace

    def log_scalar(self, key: str, value: int | float | Tensor, *, namespace: str | None = None) -> None:
        assert isinstance(value, (int, float, Tensor))
        namespace = self.resolve_namespace(namespace)
        self.scalars[namespace][key] = value

    def log_string(
        self,
        key: str,
        value: str,
        *,
        namespace: str | None = None,
    ) -> None:
        assert isinstance(value, str)
        namespace = self.resolve_namespace(namespace)
        self.strings[namespace][key] = value

    def log_image(self, key: str, value: Tensor, *, namespace: str | None = None) -> None:
        namespace = self.resolve_namespace(namespace)
        assert isinstance(value, Tensor)
        self.images[namespace][key] = standardize_image(value, log_key=f"{namespace}/{key}")

    def log_images(
        self,
        key: str,
        value: Tensor,
        *,
        namespace: str | None = None,
        max_images: int | None = None,
        sep: int = 0,
    ) -> None:
        namespace = self.resolve_namespace(namespace)
        value = standardize_images(value, max_images=max_images, log_key=f"{namespace}/{key}")
        image = make_square_image_or_video(value, sep=sep)
        self.log_image(key, image, namespace=namespace)

    def log_video(
        self,
        key: str,
        value: Tensor,
        *,
        namespace: str | None = None,
        fps: int | None = None,
        length: int | None = None,
    ) -> None:
        namespace = self.resolve_namespace(namespace)
        assert isinstance(value, Tensor)
        video = standardize_video(value, log_key=f"{namespace}/{key}")
        video = normalize_fps(video, fps, length)
        self.videos[namespace][key] = video

    def log_videos(
        self,
        key: str,
        value: Tensor | List[Tensor],
        *,
        namespace: str | None = None,
        max_videos: int | None = None,
        sep: int = 0,
        fps: int | None = None,
        length: int | None = None,
    ) -> None:
        namespace = self.resolve_namespace(namespace)
        video = normalize_fps(value, fps, length, stack_dim=1)
        video = standardize_videos(video, max_videos=max_videos, log_key=f"{namespace}/{key}")
        video = make_square_image_or_video(video, sep=sep)
        self.log_video(key, video, namespace=namespace)

    def log_point_cloud(
        self,
        key: str,
        value: Tensor,
        *,
        namespace: str | None = None,
        max_points: int = 1000,
    ) -> None:
        namespace = self.resolve_namespace(namespace)
        self.point_clouds[namespace][key] = standardize_point_cloud(value, max_points, log_key=f"{namespace}/{key}")

    def log_histogram(
        self,
        key: str,
        value: Tensor,
        *,
        namespace: str | None = None,
    ) -> None:
        namespace = self.resolve_namespace(namespace)
        self.histograms[namespace][key] = value

    def write_dict(
        self,
        loggers: List[BaseLogger],
        values: Dict[str, Dict[str, LogT]],
        state: State,
        func: Callable[[BaseLogger], Callable[[str, LogT, State, str], None]],
    ) -> None:
        for logger in loggers:
            for namespace, value in values.items():
                for key, log_value in value.items():
                    func(logger)(key, log_value, state, namespace)
        values.clear()

    def write(self, loggers: List[BaseLogger], state: State) -> None:
        self.write_dict(loggers, self.scalars, state, lambda logger: logger.log_scalar)
        self.write_dict(loggers, self.strings, state, lambda logger: logger.log_string)
        self.write_dict(loggers, self.images, state, lambda logger: logger.log_image)
        self.write_dict(loggers, self.videos, state, lambda logger: logger.log_video)
        self.write_dict(loggers, self.histograms, state, lambda logger: logger.log_histogram)
        self.write_dict(loggers, self.point_clouds, state, lambda logger: logger.log_point_cloud)
