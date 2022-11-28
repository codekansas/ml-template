import datetime
import functools
import math
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as V
from omegaconf import DictConfig, OmegaConf, SCMode
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor
from torchvision.transforms import InterpolationMode

from ml.core.config import BaseConfig, BaseObjectWithPointers, conf_field
from ml.core.state import Phase, State

LogT = TypeVar("LogT")
Number = Union[int, float, Tensor]

DEFAULT_NAMESPACE = "value"
VALID_CHANNEL_COUNTS = {1, 3}
TARGET_FPS = 12


def flatten(d: Dict[str, Any] | List[Any] | Tuple[Any, ...]) -> Dict[str, int | float | str | bool]:
    out_dict: Dict[str, int | float | str | bool] = {}

    # Flattens dictionaries.
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, (int, float, str, bool)):
                out_dict[k] = v
            elif isinstance(v, (dict, list, tuple)):
                for kk, vv in flatten(v).items():
                    out_dict[f"{k}.{kk}"] = vv
            elif v is None:
                pass
            else:
                raise NotImplementedError(f"Unsupported value type {type(v)}")

    # Flattens lists and tuples.
    elif isinstance(d, (list, tuple)):
        for i, v in enumerate(d):
            if isinstance(v, (int, float, str, bool)):
                out_dict[f"{i}"] = v
            elif isinstance(v, (dict, list, tuple)):
                for kk, vv in flatten(v).items():
                    out_dict[f"{i}.{kk}"] = vv
            elif v is None:
                pass
            else:
                raise NotImplementedError(f"Unsupported value type {type(v)}")

    else:
        raise NotImplementedError(f"Unsupported top-level type: {type(v)}")

    return out_dict


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

    def log_scalar(self, key: str, value: Callable[[], Number], state: State, namespace: str) -> None:
        """Logs a scalar value.

        Args:
            key: The key to log
            value: The value to log
            state: The current log state
            namespace: The namespace to be logged
        """

    def log_string(self, key: str, value: Callable[[], str], state: State, namespace: str) -> None:
        """Logs a string value.

        Args:
            key: The key to log
            value: The value to log
            state: The current log state
            namespace: The namespace to be logged
        """

    def log_image(self, key: str, value: Callable[[], Tensor], state: State, namespace: str) -> None:
        """Logs a normalized image, with shape (C, H, W).

        Args:
            key: The key to log
            value: The value to log
            state: The current log state
            namespace: The namespace to be logged
        """

    def log_video(self, key: str, value: Callable[[], Tensor], state: State, namespace: str) -> None:
        """Logs a normalized video, with shape (T, C, H, W).

        Args:
            key: The key to log
            value: The value to log
            state: The current log state
            namespace: The namespace to be logged
        """

    def log_histogram(self, key: str, value: Callable[[], Tensor], state: State, namespace: str) -> None:
        """Logs a histogram, with any shape.

        Args:
            key: The key to log
            value: The value to log
            state: The current log state
            namespace: The namespace to be logged
        """

    def log_point_cloud(self, key: str, value: Callable[[], Tensor], state: State, namespace: str) -> None:
        """Logs a normalized point cloud, with shape (B, N, 3).

        Args:
            key: The key to log
            value: The value to log
            state: The current log state
            namespace: The namespace to be logged
        """

    def log_config(self, config: Dict[str, int | float | str | bool], metrics: Dict[str, int | float]) -> None:
        """Logs a set of metrics and configuration.

        This is only called once, when metrics are computed for a whole dataset.

        Args:
            config: The run config
            metrics: Dictionary of metrics
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


def _chunk_lines(text: str, max_length: int) -> Iterator[str]:
    for i in range(0, len(text), max_length):
        yield text[i : i + max_length]


def standardize_text(text: str, max_line_length: int | None = None, remove_non_ascii: bool = False) -> List[str]:
    """Standardizes a text string to a list of lines.

    Args:
        text: The text to standardize
        max_line_length: If set, truncate lines to this length
        remove_non_ascii: Remove non-ASCII characters if present

    Returns:
        The standardized text lines
    """

    if remove_non_ascii:
        text = "".join(char for char in text if ord(char) < 128)
    lines = [re.sub(r"\s+", " ", line) for line in re.split(r"[\n\r]+", text.strip())]
    if max_line_length is not None:
        lines = [subline for line in lines for subline in _chunk_lines(line, max_line_length)]
    return lines


def make_human_viewable_resolution(
    image: Tensor,
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    trg_res: Tuple[int, int] = (250, 250),
) -> Tensor:
    """Resizes image to human-viewable resolution.

    Args:
        image: The image to resize, with shape (C, H, W)
        interpolation: Interpolation mode to use for image resizing
        trg_res: The target image resolution; the image will be reshaped to
            have approximately the same area as an image with this resolution

    Returns:
        The resized image
    """

    height, width = V.get_image_size(image)
    trg_height, trg_width = trg_res
    factor = math.sqrt((trg_height * trg_width) / (height * width))
    new_height, new_width = int(height * factor), int(width * factor)
    return V.resize(image, [new_height, new_width], interpolation)


def standardize_image(
    image: Tensor,
    *,
    log_key: str | None = None,
    normalize: bool = True,
    keep_resolution: bool = False,
) -> Tensor:
    """Converts an arbitrary image to shape (C, H, W).

    Args:
        image: The image tensor to log
        log_key: An optional logging key to use in the exception message
        normalize: Normalize images to (0, 1)
        keep_resolution: If set, preserve original image resolution, otherwise
            change image resolution to human-viewable

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
        image = image.unsqueeze(0)
    elif image.ndim == 3:
        if image.shape[0] in VALID_CHANNEL_COUNTS:
            pass
        elif image.shape[2] in VALID_CHANNEL_COUNTS:
            image = image.permute(2, 0, 1)
        else:
            raise ValueError(f"Invalid channel count{'' if log_key is None else f' for {log_key}'}: {image.shape}")
    else:
        raise ValueError(f"Invalid image shape{'' if log_key is None else f' for {log_key}'}: {image.shape}")

    if not keep_resolution:
        image = make_human_viewable_resolution(image)
    return image


def standardize_images(
    images: Tensor,
    *,
    max_images: int | None = None,
    log_key: str | None = None,
    normalize: bool = True,
    keep_resolution: bool = False,
) -> Tensor:
    """Converts an arbitrary set of images to shape (B, C, H, W).

    Args:
        images: The image tensor to log
        max_images: Maximum number of images to select
        log_key: An optional logging key to use in the exception message
        normalize: Normalize images to (0, 1)
        keep_resolution: If set, preserve original image resolution, otherwise
            change image resolution to human-viewable

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
        images = images.unsqueeze(1)
    elif images.ndim == 4:
        if images.shape[1] in VALID_CHANNEL_COUNTS:
            images = images if max_images is None else images[:max_images]
        elif images.shape[3] in VALID_CHANNEL_COUNTS:
            images = images.permute(0, 3, 1, 2)
            images = images if max_images is None else images[:max_images]
        else:
            raise ValueError(f"Invalid channel count{'' if log_key is None else f' for {log_key}'}: {images.shape}")
    else:
        raise ValueError(f"Invalid image shape{'' if log_key is None else f' for {log_key}'}: {images.shape}")

    if not keep_resolution:
        images = torch.stack([make_human_viewable_resolution(image) for image in images.unbind(0)], 0)
    return images


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


def image_with_text(
    image: Tensor,
    text: List[str],
    max_num_lines: int | None = None,
    line_spacing: int = 4,
    centered: bool = True,
) -> Tensor:
    """Adds a text label to an image.

    Args:
        image: The image to label, with shape (C, H, W)
        text: The text label for the image
        max_num_lines: The number of lines of spacing to add to the bottom
            of the image
        line_spacing: The spacing between adjacent lines
        centered: If set, center the text labels, otherwise align to the left

    Returns:
        The image with a text label
    """

    if not text:
        return image
    if max_num_lines is None:
        max_num_lines = len(text)
    else:
        text = text[:max_num_lines]
    pil_image = V.to_pil_image(image)
    width, height = pil_image.size
    font: ImageFont = ImageFont.load_default()
    _, _, _, line_height = font.getbbox(text[0])
    new_width, new_height = width, height + line_spacing + max_num_lines * (line_height + line_spacing)
    padded_image = Image.new(pil_image.mode, (new_width, new_height), (255, 255, 255))
    padded_image.paste(pil_image, (0, 0))
    drawer = ImageDraw.Draw(padded_image)
    for i, text_line in enumerate(text):
        text_line_top = height + line_spacing + i * (line_height + line_spacing)
        if centered:
            _, _, line_width, _ = font.getbbox(text_line)
            text_line_left = (width - line_width) / 2
            drawer.text((text_line_left, text_line_top), text_line, font=font, fill=(0, 0, 0))
        else:
            drawer.text((line_spacing, text_line_top), text_line, font=font, fill=(0, 0, 0))
    return V.pil_to_tensor(padded_image)


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


def cast_fp32(value: Any) -> Any:
    if isinstance(value, Tensor) and value.is_floating_point():
        value = value.detach().float().cpu()
    return value


class MultiLogger:
    """Defines an intermediate container which holds values to log somewhere else."""

    def __init__(self, default_namespace: str = DEFAULT_NAMESPACE) -> None:
        self.config: Tuple[DictConfig, Dict[str, int | float]] | None = None
        self.has_logged_config = False
        self.scalars: Dict[str, Dict[str, Callable[[], Number]]] = defaultdict(dict)
        self.strings: Dict[str, Dict[str, Callable[[], str]]] = defaultdict(dict)
        self.images: Dict[str, Dict[str, Callable[[], Tensor]]] = defaultdict(dict)
        self.videos: Dict[str, Dict[str, Callable[[], Tensor]]] = defaultdict(dict)
        self.histograms: Dict[str, Dict[str, Callable[[], Tensor]]] = defaultdict(dict)
        self.point_clouds: Dict[str, Dict[str, Callable[[], Tensor]]] = defaultdict(dict)
        self.poses: Dict[str, Dict[str, Callable[[], Tensor]]] = defaultdict(dict)
        self.default_namespace = default_namespace

    def resolve_namespace(self, namespace: str | None = None) -> str:
        return self.default_namespace if namespace is None else namespace

    def log_scalar(self, key: str, value: Callable[[], Number] | Number, *, namespace: str | None = None) -> None:
        """Logs a scalar value.

        Args:
            key: The key being logged
            value: The scalar value being logged
            namespace: An optional logging namespace
        """

        namespace = self.resolve_namespace(namespace)

        @functools.lru_cache
        def scalar_future() -> Number:
            value_concrete = value() if callable(value) else value
            assert isinstance(value_concrete, (int, float, Tensor))
            value_concrete = cast_fp32(value_concrete)
            return value_concrete

        self.scalars[namespace][key] = scalar_future

    def log_string(self, key: str, value: Callable[[], str] | str, *, namespace: str | None = None) -> None:
        """Logs a string value.

        Args:
            key: The key being logged
            value: The string value being logged
            namespace: An optional logging namespace
        """

        namespace = self.resolve_namespace(namespace)

        @functools.lru_cache
        def value_future() -> str:
            value_concrete = value() if callable(value) else value
            assert isinstance(value_concrete, str)
            return value_concrete

        self.strings[namespace][key] = value_future

    def log_image(
        self,
        key: str,
        value: Callable[[], Tensor] | Tensor,
        *,
        namespace: str | None = None,
        keep_resolution: bool = False,
    ) -> None:
        """Logs an image.

        Args:
            key: The key being logged
            value: The image being logged; can be (C, H, W), (H, W, C) or (H, W)
                as an RGB (3 channel) or grayscale (1 channel) image
            namespace: An optional logging namespace
            keep_resolution: If set, keep the image resolution the same,
                otherwise upscale or downscale the image to a standard
                resolution
        """

        namespace = self.resolve_namespace(namespace)

        @functools.lru_cache
        def image_future() -> Tensor:
            value_concrete = value() if callable(value) else value
            assert isinstance(value_concrete, Tensor)
            value_concrete = cast_fp32(value_concrete)
            return standardize_image(value_concrete, log_key=f"{namespace}/{key}", keep_resolution=keep_resolution)

        self.images[namespace][key] = image_future

    def log_labeled_image(
        self,
        key: str,
        value: Callable[[], Tuple[Tensor, str]] | Tuple[Tensor, str],
        *,
        namespace: str | None = None,
        max_line_length: int | None = None,
        keep_resolution: bool = False,
        centered: bool = True,
    ) -> None:
        """Logs an image with a label.

        Args:
            key: The key being logged
            value: The image and label being logged; the image can be (C, H, W),
                (H, W, C) or (H, W) as an RGB (3 channel) or grayscale
                (1 channel) image
            namespace: An optional logging namespace
            max_line_length: Labels longer than this length are wrapped around
            keep_resolution: If set, keep the image resolution the same,
                otherwise upscale or downscale the image to a standard
                resolution
            centered: If set, center the text labels, otherwise align to the
                left
        """

        namespace = self.resolve_namespace(namespace)

        @functools.lru_cache
        def labeled_image_future() -> Tensor:
            image, text = value() if callable(value) else value
            assert isinstance(image, Tensor)
            assert isinstance(text, str)
            image = standardize_image(image, log_key=f"{namespace}/{key}", keep_resolution=keep_resolution)
            text = standardize_text(text, max_line_length=max_line_length, remove_non_ascii=True)
            image = cast_fp32(image)
            return image_with_text(image, text, centered=centered)

        self.images[namespace][key] = labeled_image_future

    def log_images(
        self,
        key: str,
        value: Callable[[], Tensor] | Tensor,
        *,
        namespace: str | None = None,
        keep_resolution: bool = False,
        max_images: int | None = None,
        sep: int = 0,
    ) -> None:
        """Logs a set of images.

        The images are tiled to be nearly-square.

        Args:
            key: The key being logged
            value: The images being logged; can be (B, C, H, W), (B, H, W, C)
                or (B H, W) as an RGB (3 channel) or grayscale (1 channel) image
            namespace: An optional logging namespace
            keep_resolution: If set, keep the image resolution the same,
                otherwise upscale or downscale the image to a standard
                resolution
            max_images: The maximum number of images to show; extra images
                are clipped
            sep: An optional separation amount between adjacent images
        """

        namespace = self.resolve_namespace(namespace)

        @functools.lru_cache
        def images_future() -> Tensor:
            value_concrete = value() if callable(value) else value
            assert isinstance(value_concrete, Tensor)
            value_concrete = standardize_images(
                value_concrete,
                max_images=max_images,
                log_key=f"{namespace}/{key}",
                keep_resolution=keep_resolution,
            )
            value_concrete = cast_fp32(value_concrete)
            return make_square_image_or_video(value_concrete, sep=sep)

        self.images[namespace][key] = images_future

    def log_labeled_images(
        self,
        key: str,
        value: Callable[[], Tuple[Tensor, Sequence[str]]] | Tuple[Tensor, Sequence[str]],
        *,
        namespace: str | None = None,
        max_line_length: int | None = None,
        keep_resolution: bool = False,
        max_images: int | None = None,
        sep: int = 0,
        centered: bool = True,
    ) -> None:
        """Logs a set of images with labels.

        The images are tiled to be nearly-square.

        Args:
            key: The key being logged
            value: The images and labels being logged; images can be
                (B, C, H, W), (B, H, W, C) or (B H, W) as an RGB (3 channel)
                or grayscale (1 channel) image, with exactly B labels
            namespace: An optional logging namespace
            max_line_length: Labels longer than this length are wrapped around
            keep_resolution: If set, keep the image resolution the same,
                otherwise upscale or downscale the image to a standard
                resolution
            max_images: The maximum number of images to show; extra images
                are clipped
            sep: An optional separation amount between adjacent images
            centered: If set, center the text labels, otherwise align to the
                left
        """

        namespace = self.resolve_namespace(namespace)

        @functools.lru_cache
        def labeled_images_future() -> Tensor:
            images, texts = value() if callable(value) else value
            assert isinstance(images, Tensor)
            assert images.shape[0] == len(texts)
            num_images = len(images)
            images = standardize_images(
                images,
                max_images=max_images,
                log_key=f"{namespace}/{key}",
                keep_resolution=keep_resolution,
            )
            text_lists = [standardize_text(text, max_line_length, remove_non_ascii=True) for text in texts]
            max_num_lines = max(len(text_list) for text_list in text_lists)
            labeled_images = torch.stack(
                [
                    image_with_text(images[i], text_lists[i], max_num_lines=max_num_lines, centered=centered)
                    for i in range(num_images)
                ],
                dim=0,
            )
            return make_square_image_or_video(labeled_images, sep=sep)

        self.images[namespace][key] = labeled_images_future

    def log_video(
        self,
        key: str,
        value: Callable[[], Tensor] | Tensor,
        *,
        namespace: str | None = None,
        fps: int | None = None,
        length: int | None = None,
    ) -> None:
        """Logs a video.

        Args:
            key: The key being logged
            value: The video being logged; the video can be (T, C, H, W),
                (T, H, W, C) or (T, H, W) as an RGB (3 channel) or grayscale
                (1 channel) video
            namespace: An optional logging namespace
            fps: The video frames per second
            length: The desired video length, in seconds, at the target FPS
        """

        namespace = self.resolve_namespace(namespace)

        @functools.lru_cache
        def video_future() -> Tensor:
            value_concrete = value() if callable(value) else value
            assert isinstance(value_concrete, Tensor)
            video = standardize_video(value_concrete, log_key=f"{namespace}/{key}")
            value_concrete = cast_fp32(value_concrete)
            return normalize_fps(video, fps, length)

        self.videos[namespace][key] = video_future

    def log_videos(
        self,
        key: str,
        value: Callable[[], Tensor | List[Tensor]] | Tensor | List[Tensor],
        *,
        namespace: str | None = None,
        max_videos: int | None = None,
        sep: int = 0,
        fps: int | None = None,
        length: int | None = None,
    ) -> None:
        """Logs a set of video.

        Args:
            key: The key being logged
            value: The videos being logged; the video can be (B, T, C, H, W),
                (B, T, H, W, C) or (B T, H, W) as an RGB (3 channel) or
                grayscale (1 channel) video
            namespace: An optional logging namespace
            max_videos: The maximum number of videos to show; extra images
                are clipped
            sep: An optional separation amount between adjacent videos
            fps: The video frames per second
            length: The desired video length, in seconds, at the target FPS
        """

        namespace = self.resolve_namespace(namespace)

        @functools.lru_cache
        def videos_future() -> Tensor:
            value_concrete = value() if callable(value) else value
            assert isinstance(value_concrete, (Tensor, list))
            video = normalize_fps(value_concrete, fps, length, stack_dim=1)
            video = standardize_videos(video, max_videos=max_videos, log_key=f"{namespace}/{key}")
            value_concrete = cast_fp32(value_concrete)
            return make_square_image_or_video(video, sep=sep)

        self.videos[namespace][key] = videos_future

    def log_histogram(self, key: str, value: Callable[[], Tensor] | Tensor, *, namespace: str | None = None) -> None:
        """Logs a histogram.

        Args:
            key: The key being logged
            value: The values to create a histogram from, with arbitrary shape
            namespace: An optional logging namespace
        """

        namespace = self.resolve_namespace(namespace)

        @functools.lru_cache
        def histogram_future() -> Tensor:
            value_concrete = value() if callable(value) else value
            assert isinstance(value_concrete, Tensor)
            value_concrete = cast_fp32(value_concrete)
            return value_concrete

        self.histograms[namespace][key] = histogram_future

    def log_point_cloud(
        self,
        key: str,
        value: Callable[[], Tensor] | Tensor,
        *,
        namespace: str | None = None,
        max_points: int = 1000,
    ) -> None:
        """Logs a point cloud.

        Args:
            key: The key being logged
            value: The point cloud values, with shape (N, 3) or (B, ..., 3);
                can pass multiple batches in order to show multiple point
                clouds
            namespace: An optional logging namespace
            max_points: An optional maximum number of points in the point cloud
        """

        namespace = self.resolve_namespace(namespace)

        @functools.lru_cache
        def point_cloud_future() -> Tensor:
            value_concrete = value() if callable(value) else value
            assert isinstance(value_concrete, Tensor)
            value_concrete = cast_fp32(value_concrete)
            return standardize_point_cloud(value_concrete, max_points, log_key=f"{namespace}/{key}")

        self.point_clouds[namespace][key] = point_cloud_future

    def write_dict(
        self,
        loggers: List[BaseLogger],
        values: Dict[str, Dict[str, Callable[[], LogT]]],
        state: State,
        func: Callable[[BaseLogger], Callable[[str, Callable[[], LogT], State, str], None]],
    ) -> None:
        for logger in loggers:
            for namespace, value in values.items():
                for key, log_value in value.items():
                    func(logger)(key, log_value, state, namespace)
        values.clear()

    def log_config(self, config: DictConfig, metrics: Dict[str, int | float]) -> None:
        if self.config is not None:
            raise RuntimeError("Config has already been logged; don't log it twice")
        self.config = (config, metrics)

    def write(self, loggers: List[BaseLogger], state: State) -> None:
        self.write_dict(loggers, self.scalars, state, lambda logger: logger.log_scalar)
        self.write_dict(loggers, self.strings, state, lambda logger: logger.log_string)
        self.write_dict(loggers, self.images, state, lambda logger: logger.log_image)
        self.write_dict(loggers, self.videos, state, lambda logger: logger.log_video)
        self.write_dict(loggers, self.histograms, state, lambda logger: logger.log_histogram)
        self.write_dict(loggers, self.point_clouds, state, lambda logger: logger.log_point_cloud)

        # Only writes the config once.
        if self.config is not None and not self.has_logged_config:
            raw_config, metrics = self.config
            raw_config_dict = OmegaConf.to_container(
                cfg=raw_config,
                resolve=True,
                throw_on_missing=False,
                enum_to_str=True,
                structured_config_mode=SCMode.DICT,
            )
            config_dict = flatten(raw_config_dict)  # type: ignore
            for logger in loggers:
                logger.log_config(config_dict, metrics)
            self.has_logged_config = True
