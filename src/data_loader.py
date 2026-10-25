"""Manifest loading. Image paths are relative to the manifest directory."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.preprocessing import (
    MAX_MISSING_FRACTION,
    NormStats,
    prepare_image,
    stack_prepared,
)
from src.schema import require_valid


def load_manifest(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return require_valid(frame)


def resolve_image(manifest_path: str | Path, image_path: str) -> Path:
    candidate = Path(image_path)
    if candidate.is_absolute():
        return candidate
    return Path(manifest_path).parent / candidate


def load_image(path: str | Path) -> np.ndarray:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    if file_path.suffix == ".npy":
        return np.load(file_path)
    return np.asarray(Image.open(file_path), dtype=np.float32)


def prepare_split(
    frame: pd.DataFrame,
    manifest_path: str | Path,
    split: str,
    image_size: int | str | None,
    stats: NormStats | None,
    max_missing_fraction: float = MAX_MISSING_FRACTION,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one split. ``stats`` must already have been fit on train.

    ``image_size`` comes from ``data.image_size``. ``native`` keeps each
    chip's height and width, and those chips must already share a shape.
    """
    rows = frame[frame["split"] == split]
    images = []
    winds = []
    for row in rows.itertuples(index=False):
        path = resolve_image(manifest_path, row.image_path)
        images.append(
            prepare_image(load_image(path), image_size, stats, max_missing_fraction)
        )
        winds.append(float(row.wind_speed_kt))
    if not images:
        raise ValueError(f"split {split} has no rows")
    return stack_prepared(images), np.asarray(winds, dtype=np.float32)
