"""Image checks, resize, train-only normalization, and train-only augmentation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from src.config import parse_image_size
from src.splits import LeakageError

# A missing-pixel fraction above this value fails closed. It does not warn.
MAX_MISSING_FRACTION = 0.30


@dataclass(frozen=True)
class NormStats:
    mean: float
    std: float
    source_split: str


def as_channel_first(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        return array[None, ...]
    if array.ndim != 3:
        raise ValueError(f"expected a 2D or 3D image, got {array.shape}")
    if array.shape[0] <= 4 and array.shape[1] > 4 and array.shape[2] > 4:
        return array
    if array.shape[-1] <= 4:
        return np.moveaxis(array, -1, 0)
    raise ValueError(f"cannot infer the channel axis for shape {array.shape}")


def missing_fraction(image: np.ndarray) -> float:
    return float((~np.isfinite(image)).mean())


def fit_norm_stats(images: list[np.ndarray], source_split: str) -> NormStats:
    """Fit mean and std. Refuses anything other than the train split."""
    if source_split != "train":
        raise LeakageError("normalization stats may be fit on the train split only")
    if not images:
        raise ValueError("need at least one train image")
    values = [np.asarray(image, dtype=np.float32) for image in images]
    finite = np.concatenate([image[np.isfinite(image)].ravel() for image in values])
    if finite.size == 0:
        raise ValueError("train images have no finite pixels")
    std = float(finite.std())
    if std < 1e-6:
        std = 1.0
    return NormStats(mean=float(finite.mean()), std=std, source_split=source_split)


def stack_prepared(images: list[np.ndarray]) -> np.ndarray:
    """Stack chips that have already been prepared.

    A native ``image_size`` does not resize, so every chip in the stack
    must already share a height and width.
    """
    if not images:
        raise ValueError("no images to stack")
    try:
        return np.stack(images)
    except ValueError as exc:
        shapes = sorted({tuple(np.shape(image)) for image in images})
        raise ValueError(
            "prepared chips do not share a height and width "
            f"{shapes}. A native image_size does not resize them onto one grid."
        ) from exc


def _resize(image: np.ndarray, size: int) -> np.ndarray:
    channels = []
    for channel in image:
        picture = Image.fromarray(channel.astype(np.float32), mode="F")
        picture = picture.resize((size, size), Image.Resampling.BILINEAR)
        channels.append(np.asarray(picture, dtype=np.float32))
    return np.stack(channels, axis=0)


def prepare_image(
    image: np.ndarray,
    image_size: int | str | None,
    stats: NormStats | None,
    max_missing_fraction: float = MAX_MISSING_FRACTION,
) -> np.ndarray:
    """Return ``(C, H, W)`` float32. Does not invent brightness temperature.

    ``image_size`` ``None`` or ``native`` keeps the chip's height and width.
    An integer resizes to that square. ``max_missing_fraction`` is
    ``data.max_missing_fraction`` when the caller has a config.
    A missing-pixel fraction above ``max_missing_fraction`` raises. Raw
    digital numbers stay pixel values; this function does not name them Tb.
    """
    array = as_channel_first(image)
    fraction = missing_fraction(array)
    if fraction > max_missing_fraction:
        raise ValueError(
            f"missing fraction {fraction:.2f} exceeds {max_missing_fraction:.2f}"
        )
    filled = array.copy()
    for index in range(filled.shape[0]):
        finite = np.isfinite(filled[index])
        fill = float(np.median(filled[index][finite])) if finite.any() else 0.0
        filled[index][~finite] = fill
    side = parse_image_size(image_size)
    output = filled if side is None else _resize(filled, side)
    if stats is not None:
        output = (output - stats.mean) / stats.std
    return output


def dn_to_brightness_temperature(dn: np.ndarray, gain: float, offset: float) -> np.ndarray:
    """Apply a caller-supplied calibration. This does not look up a sensor table."""
    return np.asarray(dn, dtype=np.float32) * float(gain) + float(offset)


def augment_image(image: np.ndarray, split: str, rng: np.random.Generator) -> np.ndarray:
    """Flip or rotate a train image. Refuses every other split.

    Classical features do not call this. A later training loop may, and only
    on rows whose split is ``train``.
    """
    if split != "train":
        raise LeakageError("augmentation is allowed on the train split only")
    array = as_channel_first(image)
    choice = int(rng.integers(0, 3))
    if choice == 0:
        return np.flip(array, axis=-1).copy()
    if choice == 1:
        return np.rot90(array, k=1, axes=(-2, -1)).copy()
    return array.copy()
