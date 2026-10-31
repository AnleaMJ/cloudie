"""Synthetic frames for the infrared gate. These are not satellite data."""

from __future__ import annotations

import numpy as np


def spiral_frame(height: int, width: int | None = None) -> np.ndarray:
    """Centered eye, eyewall, and spiral bands in normalized coordinates.

    Width defaults to ``height``. A non-square frame uses separate x and y
    scales, which is not the square accept case.
    """
    width = height if width is None else width
    yy, xx = np.mgrid[0:height, 0:width]
    dy = (yy - (height - 1) / 2) / (height / 2)
    dx = (xx - (width - 1) / 2) / (width / 2)
    radius = np.hypot(dy, dx)
    theta = np.arctan2(dy, dx)
    eye = np.exp(-(radius**2) / (2 * 0.07**2))
    wall = np.exp(-((radius - 0.20) ** 2) / (2 * 0.045**2))
    bands = (0.5 + 0.5 * np.sin(4 * theta - 10 * radius)) * np.exp(
        -((radius - 0.48) ** 2) / (2 * 0.18**2)
    )
    image = -1.0 * wall - 0.55 * bands * (radius < 0.9) + 0.75 * eye + 0.12 * (radius > 0.8)
    return image.astype(np.float32)


def uniform_noise(height: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, (height, height)).astype(np.float32)


def clear_sky(height: int) -> np.ndarray:
    row = np.linspace(-0.2, 0.8, height, dtype=np.float32)
    return np.tile(row, (height, 1))


def rgb_photo(height: int, seed: int = 1) -> np.ndarray:
    """Three channels with different means. Not a greyscale infrared field."""
    rng = np.random.default_rng(seed)
    image = rng.normal(0.0, 0.25, (height, height, 3)).astype(np.float32)
    image[..., 0] += 0.9
    image[..., 1] -= 0.4
    image[..., 2] += 0.2
    return image


def colour_enhanced(height: int, *, alpha: bool = False) -> np.ndarray:
    """False-colour rendering of a spiral, the way a display product is painted.

    The underlying pattern is organized. The channels are not copies, so the
    gate must refuse it as colour-enhanced rather than score the cyclone shape.
    ``alpha`` adds a fourth plane of ones, as in an RGBA file.
    """
    gray = spiral_frame(height)
    span = float(gray.max() - gray.min())
    unit = (gray - float(gray.min())) / max(span, 1e-6)
    red = np.clip(1.0 - unit, 0.0, 1.0)
    green = np.clip(np.sin(np.pi * unit), 0.0, 1.0)
    blue = np.clip(unit, 0.0, 1.0)
    image = np.stack([red, green, blue], axis=-1).astype(np.float32)
    if alpha:
        ones = np.ones(gray.shape, dtype=np.float32)
        image = np.concatenate([image, ones[..., None]], axis=-1)
    return image


def stratiform(height: int, seed: int = 2) -> np.ndarray:
    """Smooth cloud texture without an eye or a cold core."""
    rng = np.random.default_rng(seed)
    raw = rng.normal(0.0, 1.0, (height, height))
    kernel = 13
    pad = kernel // 2
    padded = np.pad(raw, pad, mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kernel, kernel))
    blurred = windows.mean(axis=(-1, -2))
    blurred = blurred / (float(blurred.std()) + 1e-6)
    return blurred.astype(np.float32)


def cold_count(image: np.ndarray) -> int:
    """Pixels below zero. The count grows with the frame; the fraction does not."""
    return int((np.asarray(image) < 0).sum())
