"""Manifest schema, project intensity bins, and row-level quality gates."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

REQUIRED_COLUMNS = ("sample_id", "image_path", "storm_id", "wind_speed_kt")
FORBIDDEN_FEATURES = frozenset(
    {"wind_speed_kt", "r34_nmi", "rmw_nmi", "imd5_class", "intensity_bin"}
)

# Project metric labels, not an IMD warning. Five names group D+DD and
# ESCS+SuCS. The edges 17/34/48/64/91 kt are this project's own grouping.
# IMD operational winds are 3-minute averages and are not these bins.
# NASA HURDAT2 labels are 1-minute winds; config data.wind_averaging is
# 1min for that dataset. Keep these tuples equal to data.imd5_edges_kt
# and data.imd5_names. The July 2024 IMD 3-minute sheet starts Extremely
# Severe at 90 kt, so 90 kt is VSCS here.
DEFAULT_EDGES_KT = (17.0, 34.0, 48.0, 64.0, 91.0)
DEFAULT_NAMES = ("D_DD", "CS", "SCS", "VSCS", "ESCS_SUCS")


class SchemaError(ValueError):
    """A manifest failed a quality gate."""


@dataclass(frozen=True)
class QualityReport:
    n_rows: int
    n_storms: int
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def wind_to_imd5(
    wind_kt: float,
    edges: tuple[float, ...] = DEFAULT_EDGES_KT,
    names: tuple[str, ...] = DEFAULT_NAMES,
) -> str | None:
    """Return a project class, or None below the depression threshold.

    The names are this project's grouping. They are not IMD 3-minute categories.
    """
    if len(edges) != len(names):
        raise ValueError("edges and names must have the same length")
    if pd.isna(wind_kt) or float(wind_kt) < edges[0]:
        return None
    value = float(wind_kt)
    for lower, upper, name in zip(edges, edges[1:], names):
        if lower <= value < upper:
            return name
    return names[-1]


def validate_manifest(
    frame: pd.DataFrame,
    wind_min_kt: float = 0.0,
    wind_max_kt: float = 200.0,
) -> QualityReport:
    """Fail closed on missing keys, duplicate ids, or implausible winds."""
    errors: list[str] = []
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        return QualityReport(len(frame), 0, (f"missing required columns: {missing}",))

    if frame["sample_id"].isna().any() or (frame["sample_id"].astype(str) == "").any():
        errors.append("sample_id has nulls")
    if frame["sample_id"].duplicated().any():
        examples = frame.loc[frame["sample_id"].duplicated(), "sample_id"].head(5).tolist()
        errors.append(f"duplicate sample_id, examples: {examples}")
    if frame["storm_id"].isna().any() or (frame["storm_id"].astype(str).str.len() == 0).any():
        errors.append("storm_id has nulls")
    if frame["image_path"].isna().any():
        errors.append("image_path has nulls")

    winds = pd.to_numeric(frame["wind_speed_kt"], errors="coerce")
    if winds.isna().any():
        errors.append("wind_speed_kt has non-numeric values")
    else:
        bad = (winds < wind_min_kt) | (winds > wind_max_kt)
        if bad.any():
            errors.append(
                f"{int(bad.sum())} wind values outside [{wind_min_kt}, {wind_max_kt}] kt"
            )

    if "wind_averaging" in frame.columns:
        periods = set(frame["wind_averaging"].dropna().astype(str).unique())
        if len(periods) > 1:
            errors.append(
                f"mixed wind_averaging periods {sorted(periods)}; split or convert explicitly"
            )

    n_storms = int(frame["storm_id"].nunique())
    return QualityReport(len(frame), n_storms, tuple(errors))


def require_valid(frame: pd.DataFrame, **kwargs: float) -> pd.DataFrame:
    report = validate_manifest(frame, **kwargs)
    if not report.ok:
        raise SchemaError("; ".join(report.errors))
    out = frame.copy()
    out["storm_id"] = out["storm_id"].astype(str)
    out["sample_id"] = out["sample_id"].astype(str)
    out["wind_speed_kt"] = pd.to_numeric(out["wind_speed_kt"])
    out["imd5_class"] = out["wind_speed_kt"].map(wind_to_imd5)
    return out
