"""Image checks, resize, and train-only normalization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from cloudie.splits import LeakageError


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


def _resize(image: np.ndarray, size: int) -> np.ndarray:
    channels = []
    for channel in image:
        picture = Image.fromarray(channel.astype(np.float32), mode="F")
        picture = picture.resize((size, size), Image.BILINEAR)
        channels.append(np.asarray(picture, dtype=np.float32))
    return np.stack(channels, axis=0)


def prepare_image(
    image: np.ndarray,
    image_size: int,
    stats: NormStats | None,
    max_missing_fraction: float = 0.30,
) -> np.ndarray:
    """Return ``(C, H, W)`` float32. Does not invent brightness temperature."""
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
    resized = _resize(filled, image_size)
    if stats is not None:
        resized = (resized - stats.mean) / stats.std
    return resized


def dn_to_brightness_temperature(
    dn: np.ndarray, gain: float, offset: float
) -> np.ndarray:
    """Apply a caller-supplied calibration. This does not look up a sensor table."""
    return np.asarray(dn, dtype=np.float32) * float(gain) + float(offset)


def leave_unflipped(image: np.ndarray, hemisphere: str | None = None) -> np.ndarray:
    """Use the frame as stored.

    Southern Hemisphere TCIR imagery is not mirrored. ``hemisphere`` records
    the caller's label and does not change the array.
    """
    del hemisphere
    return np.asarray(image)
