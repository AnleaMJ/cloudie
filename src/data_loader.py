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


def apply_frame_cache(
    frame: pd.DataFrame, dataset: str, cache_root: str | Path | None
) -> pd.DataFrame:
    """Point rows at cached fp16 ``.npy`` files when that dataset's index exists.

    A missing cache leaves ``image_path`` unchanged. A sample that is not
    in the index, or whose file is gone, keeps its source path.
    """
    if cache_root in (None, "", "null"):
        return frame
    index = Path(cache_root) / str(dataset).strip().lower() / "index.csv"
    if not index.is_file():
        return frame
    cached = pd.read_csv(index)
    if "sample_id" not in cached.columns or "cache_path" not in cached.columns:
        raise ValueError(f"frame cache index {index} needs sample_id and cache_path")
    mapping = {
        str(sample): str(path)
        for sample, path in zip(cached["sample_id"], cached["cache_path"], strict=False)
    }
    out = frame.copy()
    paths: list[str] = []
    replaced = 0
    for sample_id, image_path in zip(out["sample_id"].astype(str), out["image_path"].astype(str)):
        candidate = mapping.get(sample_id)
        if candidate and Path(candidate).is_file():
            paths.append(str(Path(candidate).resolve()))
            if paths[-1] != str(Path(image_path).resolve()):
                replaced += 1
        else:
            paths.append(image_path)
    out["image_path"] = paths
    if replaced:
        print(f"{dataset} frame cache supplied {replaced} fp16 frames from {index}")
    return out


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
