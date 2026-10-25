"""Storm-safe splits. Random frame splits are rejected.

``storm_ids_by_split`` returns the disjoint train, val, and test storm ids.
A later residual-quantile coverage check can reuse that map from an assigned
frame without assigning rows again.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.schema import DEFAULT_NAMES, FORBIDDEN_FEATURES, wind_to_imd5

_CLASS_SEQUENCES = tuple(
    tuple(part for part in re.split(r"[^a-z0-9]+", name.lower()) if part)
    for name in DEFAULT_NAMES
)
_COMPACT_MARKERS = ("windspeed", "windkt", "r34", "rmw", "imd5", "intensity")
_LABEL_TOKENS = frozenset(
    {"wind", "windspeed", "windkt", "r34", "rmw", "imd5", "intensity", "class"}
)


class LeakageError(RuntimeError):
    """A split shares a storm across sides, or is a random frame split."""


def _name_tokens(name: str) -> list[str]:
    return [part for part in re.split(r"[^a-z0-9]+", str(name).lower()) if part]


def _contains_sequence(tokens: list[str], sequence: tuple[str, ...]) -> bool:
    width = len(sequence)
    if width == 0 or len(tokens) < width:
        return False
    return any(
        tuple(tokens[index : index + width]) == sequence
        for index in range(len(tokens) - width + 1)
    )


def is_forbidden_input(name: str) -> bool:
    """True for a label column or a name derived from wind, size, or class.

    Image features such as ``pixel_mean`` and ``delta_pixel_mean`` stay allowed.
    """
    text = str(name).strip().lower()
    if text in {item.lower() for item in FORBIDDEN_FEATURES}:
        return True
    if text in {item.lower() for item in DEFAULT_NAMES}:
        return True
    tokens = _name_tokens(text)
    if any(
        token in _LABEL_TOKENS or token.startswith(("windspeed", "windkt", "r34", "rmw", "imd5"))
        for token in tokens
    ):
        return True
    if any(_contains_sequence(tokens, sequence) for sequence in _CLASS_SEQUENCES):
        return True
    compact = "".join(tokens)
    return any(marker in compact for marker in _COMPACT_MARKERS)


def forbidden_input_names(names: Iterable[str]) -> list[str]:
    return sorted({str(name) for name in names if is_forbidden_input(str(name))})


def assert_no_forbidden_inputs(names: Iterable[str], detail: str) -> None:
    banned = forbidden_input_names(names)
    if banned:
        raise LeakageError(f"{detail}: {banned}")


def assert_no_group_leakage(*frames: pd.DataFrame, group_col: str = "storm_id") -> None:
    seen: dict[str, int] = {}
    for index, frame in enumerate(frames):
        if frame.empty:
            continue
        for storm_id in frame[group_col].astype(str).unique():
            if storm_id in seen:
                raise LeakageError(
                    f"{group_col}={storm_id} is in split index {seen[storm_id]} and {index}"
                )
            seen[storm_id] = index
    sample_ids: list[str] = []
    for frame in frames:
        if "sample_id" in frame.columns:
            sample_ids.extend(frame["sample_id"].astype(str).tolist())
    if len(sample_ids) != len(set(sample_ids)):
        raise LeakageError("sample_id repeated across splits")


def storm_ids_by_split(frame: pd.DataFrame) -> dict[str, frozenset[str]]:
    """Return disjoint storm ids for ``train``, ``val``, and ``test``.

    A later residual-quantile coverage check can take ``result["test"]``
    from an assigned frame and score those storms without assigning rows again.
    """
    if "storm_id" not in frame.columns or "split" not in frame.columns:
        raise LeakageError("storm_id and split are required")
    known = ("train", "val", "test")
    labels = set(frame["split"].astype(str).unique())
    unknown = labels - set(known)
    if unknown:
        raise LeakageError(f"unknown splits: {sorted(unknown)}")
    parts = [frame.loc[frame["split"].astype(str) == name] for name in known]
    assert_no_group_leakage(*parts)
    return {
        name: frozenset(part["storm_id"].astype(str))
        for name, part in zip(known, parts, strict=True)
    }


def _storm_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for storm_id, group in frame.groupby("storm_id", sort=False):
        if "year" in group.columns:
            years = pd.to_numeric(group["year"], errors="coerce").dropna().astype(int)
        else:
            years = pd.Series(dtype=int)
        basins = (
            sorted(set(group["basin"].dropna().astype(str)))
            if "basin" in group.columns
            else []
        )
        median_wind = float(pd.to_numeric(group["wind_speed_kt"]).median())
        rows.append(
            {
                "storm_id": str(storm_id),
                "year_min": int(years.min()) if len(years) else pd.NA,
                "year_max": int(years.max()) if len(years) else pd.NA,
                "basins": basins,
                "stratify": wind_to_imd5(median_wind) or "below",
            }
        )
    return pd.DataFrame(rows)


def _counts(n: int, test_fraction: float, val_fraction: float) -> tuple[int, int]:
    if n < 3:
        raise LeakageError("need at least 3 storms for train/val/test")
    n_test = max(1, int(round(n * test_fraction)))
    n_val = max(1, int(round((n - n_test) * val_fraction)))
    if n - n_test - n_val < 1:
        raise LeakageError("split fractions left no training storms")
    return n_test, n_val


def _ordered_storms(table: pd.DataFrame, seed: int) -> list[str]:
    """Interleave intensity bins so a slice is approximately stratified."""
    rng = np.random.default_rng(seed)
    groups = [
        list(rng.permutation(ids.to_numpy()))
        for _, ids in table.groupby("stratify")["storm_id"]
    ]
    ordered: list[str] = []
    while any(groups):
        for group in groups:
            if group:
                ordered.append(str(group.pop()))
    return ordered


def _assignment_from_ids(
    table: pd.DataFrame, test_ids: set[str], val_ids: set[str]
) -> dict[str, str]:
    return {
        sid: ("test" if sid in test_ids else "val" if sid in val_ids else "train")
        for sid in table["storm_id"].astype(str)
    }


def assign_storms(
    frame: pd.DataFrame,
    strategy: str,
    test_fraction: float = 0.2,
    val_fraction: float = 0.2,
    seed: int = 42,
    holdout_years: list[int] | None = None,
    holdout_basin: str | None = None,
) -> pd.DataFrame:
    """Return a copy with ``split`` in {train, val, test}. Storms are never cut."""
    strategy = strategy.lower().strip()
    if strategy in {"random", "frame", "shuffle", "random_frame"}:
        raise LeakageError(
            "random frame splits are forbidden; use storm_wise, chronological, or holdout_basin"
        )
    work = frame.copy()
    work["storm_id"] = work["storm_id"].astype(str)
    table = _storm_table(work)
    holdout_years = list(holdout_years or [])
    if strategy == "chronological":
        if table["year_min"].isna().any():
            missing = table.loc[table["year_min"].isna(), "storm_id"].tolist()
            raise LeakageError(f"chronological split needs a year for every storm: {missing}")
        if not holdout_years:
            years = sorted(int(year) for year in table["year_min"].astype(int).unique())
            n_hold = max(1, int(round(len(years) * test_fraction)))
            holdout_years = years[-n_hold:]
        held = {int(year) for year in holdout_years}

        def touches_year(row: pd.Series) -> bool:
            span = set(range(int(row.year_min), int(row.year_max) + 1))
            return bool(span & held)

        test_ids = set(table.loc[table.apply(touches_year, axis=1), "storm_id"].astype(str))
        pool = table[~table["storm_id"].isin(test_ids)]
        if pool.empty or not test_ids:
            raise LeakageError("chronological split produced an empty side")
        n_val = max(1, int(round(len(pool) * val_fraction)))
        if n_val >= len(pool):
            raise LeakageError("not enough non-holdout storms for train and val")
        val_ids = set(_ordered_storms(pool, seed)[:n_val])
        assignment = _assignment_from_ids(table, test_ids, val_ids)
    elif strategy == "holdout_basin":
        if not holdout_basin:
            raise LeakageError("holdout_basin strategy requires holdout_basin")
        test_ids = set(
            table.loc[
                table["basins"].map(lambda basins: holdout_basin in basins), "storm_id"
            ].astype(str)
        )
        pool = table[~table["storm_id"].isin(test_ids)]
        if pool.empty or not test_ids:
            raise LeakageError("holdout basin split produced an empty side")
        n_val = max(1, int(round(len(pool) * val_fraction)))
        if n_val >= len(pool):
            raise LeakageError("not enough storms outside the holdout basin for train and val")
        val_ids = set(_ordered_storms(pool, seed)[:n_val])
        assignment = _assignment_from_ids(table, test_ids, val_ids)
    elif strategy == "storm_wise":
        ordered = _ordered_storms(table, seed)
        n_test, n_val = _counts(len(ordered), test_fraction, val_fraction)
        assignment = _assignment_from_ids(
            table, set(ordered[:n_test]), set(ordered[n_test : n_test + n_val])
        )
    else:
        raise LeakageError(f"unknown split strategy: {strategy}")
    work["split"] = work["storm_id"].map(assignment)
    if work["split"].isna().any():
        raise LeakageError("some rows were not assigned a split")
    parts = [work[work["split"] == name] for name in ("train", "val", "test")]
    assert_no_group_leakage(*parts)
    return work


def add_previous_frame_id(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the previous sample inside the same storm, ordered by time.

    The link is not yet split-safe. Call ``null_cross_split_links`` before
    using it as a feature.
    """
    if "storm_id" not in frame.columns or "sample_id" not in frame.columns:
        raise LeakageError("storm_id and sample_id are required")
    out = frame.copy()
    out["storm_id"] = out["storm_id"].astype(str)
    out["sample_id"] = out["sample_id"].astype(str)
    if "timestamp" in out.columns:
        parsed = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
        if parsed.isna().any():
            order_key = out["timestamp"].astype(str)
        else:
            order_key = parsed
        out = out.assign(_order=order_key)
        out = out.sort_values(["storm_id", "_order"], kind="mergesort")
    else:
        out = out.sort_values(["storm_id"], kind="mergesort")
    out["prev_sample_id"] = out.groupby("storm_id", sort=False)["sample_id"].shift(1)
    out = out.drop(columns=["_order"], errors="ignore")
    return out.sort_index()


def _sample_split_lookup(frame: pd.DataFrame) -> pd.Series:
    """Map sample id strings to the split of the first row with that id."""
    temporary = frame.assign(_sid=frame["sample_id"].map(str))
    temporary = temporary.drop_duplicates("_sid")
    return temporary.set_index("_sid")["split"].astype(str)


def null_cross_split_links(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop previous-frame ids that point outside the row's split.

    Returns the frame and the number of links removed. Unknown previous ids
    are removed as well; they can hide a filtered-out frame.
    """
    if "split" not in frame.columns or "prev_sample_id" not in frame.columns:
        raise LeakageError("split and prev_sample_id are required")
    lookup = _sample_split_lookup(frame)
    out = frame.copy()
    dropped = 0
    kept: list[object] = []
    for prev, split in zip(out["prev_sample_id"], out["split"].astype(str), strict=True):
        if pd.isna(prev):
            kept.append(pd.NA)
            continue
        key = str(prev)
        if key not in lookup.index or lookup.loc[key] != split:
            kept.append(pd.NA)
            dropped += 1
        else:
            kept.append(key)
    out["prev_sample_id"] = kept
    return out, dropped
