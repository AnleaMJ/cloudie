"""Four-frame windows inside one split.

The image tensor is infrared only. Wind, R34, and radius of maximum wind
may be read as targets for the last frame. They are not channels.
A previous-frame link that leaves the split cannot bring that frame in.
Southern Hemisphere frames are not flipped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cloudie.data_loader import load_image, resolve_image
from cloudie.preprocessing import NormStats, leave_unflipped, prepare_image

N_FRAMES = 4
SIZE_LABELS = ("r34_nmi", "rmw_nmi")


def size_targets_present(frame: pd.DataFrame) -> bool:
    """True when both size columns exist and one row has a finite pair.

    All-null columns are not labels. The size head stays off until then.
    """
    if any(column not in frame.columns for column in SIZE_LABELS):
        return False
    values = frame.loc[:, list(SIZE_LABELS)].apply(pd.to_numeric, errors="coerce")
    finite_pair = np.isfinite(values.to_numpy(dtype=np.float64)).all(axis=1)
    return bool(finite_pair.any())


def _prev_id(value: object) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "<na>", "nat"}:
        return None
    return text


def _window_stays_in_split(
    sample_ids: list[str], prev_ids: list[str | None], in_split: set[str]
) -> bool:
    """Reject a window whose previous-frame link leaves the split.

    A link that was cleared is ignored. A link that is still set must name
    the previous frame in this window, except at the first frame, where a
    same-split predecessor may sit just outside a sliding window.
    """
    for offset, _sample_id in enumerate(sample_ids):
        prev = prev_ids[offset]
        if prev is None:
            continue
        if prev not in in_split:
            return False
        if offset > 0 and prev != sample_ids[offset - 1]:
            return False
    return True


def frame_windows(
    frame: pd.DataFrame, split: str, n_frames: int = N_FRAMES
) -> list[list[str]]:
    """Sample ids of consecutive frames, one storm, one split.

    ``n_frames`` is 4. A link that points outside this split rejects the
    window. A link that was cleared stays cleared and is not followed.
    """
    if n_frames != N_FRAMES:
        raise ValueError(f"ConvLSTM uses {N_FRAMES} frames")
    if "split" not in frame.columns:
        raise ValueError("frame is missing split")
    rows = frame[frame["split"] == split]
    if rows.empty:
        return []
    order = ["timestamp", "sample_id"] if "timestamp" in rows.columns else ["sample_id"]
    in_split = set(rows["sample_id"].astype(str))
    windows: list[list[str]] = []
    for _storm_id, group in rows.groupby("storm_id", sort=False):
        ordered = group.sort_values(order, kind="mergesort")
        sample_ids = [str(value) for value in ordered["sample_id"].tolist()]
        if "prev_sample_id" in ordered.columns:
            prev_ids = [_prev_id(value) for value in ordered["prev_sample_id"].tolist()]
        else:
            prev_ids = [None] * len(sample_ids)
        for start in range(0, len(sample_ids) - n_frames + 1):
            chosen = sample_ids[start : start + n_frames]
            prev = prev_ids[start : start + n_frames]
            if _window_stays_in_split(chosen, prev, in_split):
                windows.append(chosen)
    return windows


def _row(indexed: pd.DataFrame, sample_id: str) -> pd.Series:
    row = indexed.loc[sample_id]
    if isinstance(row, pd.DataFrame):
        return row.iloc[0]
    return row


def _pair(row: pd.Series, column: str) -> float:
    number = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
    if pd.isna(number):
        return float("nan")
    return float(number)


def stack_windows(
    frame: pd.DataFrame,
    manifest_path: str | Path,
    split_name: str,
    image_size: int,
    stats: NormStats | None,
    max_missing_fraction: float,
    n_frames: int = N_FRAMES,
) -> tuple[np.ndarray, np.ndarray, list[str | None], np.ndarray | None]:
    """Return ``(N, 4, C, H, W)`` images and last-frame targets.

    Size targets are returned only when :func:`size_targets_present` is true.
    """
    windows = frame_windows(frame, split_name, n_frames)
    if not windows:
        raise ValueError(f"split {split_name} has no {n_frames}-frame windows")
    indexed = frame.set_index(frame["sample_id"].astype(str), drop=False)
    use_size = size_targets_present(frame)
    images: list[np.ndarray] = []
    winds: list[float] = []
    labels: list[str | None] = []
    sizes: list[list[float]] = []
    for sample_ids in windows:
        frames: list[np.ndarray] = []
        for sample_id in sample_ids:
            row = _row(indexed, sample_id)
            array = load_image(resolve_image(manifest_path, str(row["image_path"])))
            hemisphere = None
            if "hemisphere" in frame.columns and not pd.isna(row["hemisphere"]):
                hemisphere = str(row["hemisphere"])
            frames.append(
                prepare_image(
                    leave_unflipped(array, hemisphere),
                    image_size,
                    stats,
                    max_missing_fraction,
                )
            )
        images.append(np.stack(frames, axis=0))
        last = _row(indexed, sample_ids[-1])
        winds.append(float(last["wind_speed_kt"]))
        if "_class" in frame.columns and not pd.isna(last["_class"]):
            labels.append(str(last["_class"]))
        else:
            labels.append(None)
        if use_size:
            sizes.append([_pair(last, "r34_nmi"), _pair(last, "rmw_nmi")])
    size_array = np.asarray(sizes, dtype=np.float32) if use_size else None
    return (
        np.stack(images, axis=0),
        np.asarray(winds, dtype=np.float32),
        labels,
        size_array,
    )
