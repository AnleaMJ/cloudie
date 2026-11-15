"""IMD 3-minute wind target.

NASA and TCIR labels are 1-minute sustained winds. INCYDE labels are already
IMD 3-minute winds. The training target is ``wind_kt_3min``. Raw knots stay
on ``wind_speed_kt`` and ``wind_kt``.

The 1-minute to 3-minute factor is ``labels.wind_1min_to_3min`` (0.93 in the
YAML). It is a derived at-sea gust factor from WMO/TD-1555 Tables 1.1 and
1.2 (Harper, Kepert and Ginger 2010), not a WMO constant, and it is meant
for open water more than 20 km offshore. This module does not download a
coastline. A dataset with no land indicator is marked unknown.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.schema import wind_averaging_seconds

# Citation for the default factor. The numeric factor itself lives in the YAML.
WIND_CONVERSION_NOTE = (
    "0.93, derived from WMO/TD-1555 Tables 1.1 and 1.2, "
    "at-sea exposure (not a WMO constant)"
)
HELD_RAW_NOTE = "held raw: near land, at-sea factor not applied"
NEAR_LAND_POLICIES = ("flag", "raw")


def conversion_note(factor: float) -> str:
    """Provenance string for a row whose 1-minute wind was multiplied.

    The default factor of 0.93 uses the citation verbatim. Any other
    configured factor keeps the same source and names that factor.
    """
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("wind factor must be a positive finite number")
    if f"{float(factor):.2f}" == "0.93":
        return WIND_CONVERSION_NOTE
    return (
        f"{float(factor):.2f}, derived from WMO/TD-1555 Tables 1.1 and 1.2, "
        "at-sea exposure (not a WMO constant)"
    )


def _blank(value: object) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        return False
    return str(value).strip().lower() in {"", "nan", "none", "null", "<na>"}


def _flag(value: object) -> str:
    """``true``, ``false``, or ``unknown``."""
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if _blank(value):
        return "unknown"
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "t"}:
        return "true"
    if text in {"0", "false", "no", "n", "f"}:
        return "false"
    if text in {"unknown", "na", "n/a"}:
        return "unknown"
    raise ValueError(f"land indicator {value!r} is not true, false, or unknown")


def near_land_status(frame: pd.DataFrame, open_water_km: float) -> pd.Series:
    """Per-row near-land status: ``true``, ``false``, or ``unknown``.

    ``distance_to_land_km`` is the quantitative indicator. Open water is
    strictly farther than ``open_water_km`` (20 km in the YAML). A
    ``near_land`` or ``landfall`` column is used when there is no distance.
    Latitude and longitude are not a land indicator. With none of those
    columns the whole frame is unknown.
    """
    if open_water_km < 0 or not math.isfinite(open_water_km):
        raise ValueError("open_water_km must be a non-negative finite distance")
    if "distance_to_land_km" in frame.columns:
        distance = pd.to_numeric(frame["distance_to_land_km"], errors="coerce")
        status = np.where(
            distance.isna(),
            "unknown",
            np.where(distance > float(open_water_km), "false", "true"),
        )
        return pd.Series(status, index=frame.index, dtype=object)
    if "near_land" in frame.columns:
        return pd.Series(
            [_flag(value) for value in frame["near_land"].tolist()],
            index=frame.index,
            dtype=object,
        )
    if "landfall" in frame.columns:
        return pd.Series(
            [_flag(value) for value in frame["landfall"].tolist()],
            index=frame.index,
            dtype=object,
        )
    return pd.Series("unknown", index=frame.index, dtype=object)


def _raw_knots(frame: pd.DataFrame) -> pd.Series:
    if "wind_speed_kt" not in frame.columns and "wind_kt" not in frame.columns:
        raise ValueError("a wind row needs wind_speed_kt or wind_kt")
    if "wind_speed_kt" in frame.columns and "wind_kt" in frame.columns:
        left = pd.to_numeric(frame["wind_speed_kt"], errors="coerce")
        right = pd.to_numeric(frame["wind_kt"], errors="coerce")
        if not np.allclose(left.to_numpy(dtype=float), right.to_numpy(dtype=float), equal_nan=True):
            raise ValueError("wind_kt and wind_speed_kt disagree; refusing to convert either")
        return left
    column = "wind_speed_kt" if "wind_speed_kt" in frame.columns else "wind_kt"
    return pd.to_numeric(frame[column], errors="coerce")


def apply_wind_standard(
    frame: pd.DataFrame,
    *,
    factor: float,
    policy: str,
    open_water_km: float,
) -> pd.DataFrame:
    """Add ``wind_kt_3min`` without overwriting the raw label.

    3-minute rows, including INCYDE, pass through (``wind_kt_3min == wind_kt``).
    1-minute rows are multiplied by ``factor`` unless ``policy`` is ``raw``
    and the row is near land. ``flag`` (the default) still multiplies those
    rows and sets ``conversion_exposure_flag`` true. Unknown exposure is
    converted and the flag stays ``unknown``, so the at-sea factor is not
    recorded as an open-water measurement.
    """
    choice = str(policy).strip().lower()
    if choice not in NEAR_LAND_POLICIES:
        raise ValueError("labels.near_land_policy must be flag or raw")
    note = conversion_note(factor)
    if "wind_averaging" not in frame.columns:
        raise ValueError("wind_averaging is required before the 3-minute target is added")
    raw = _raw_knots(frame).to_numpy(dtype=float)
    status = near_land_status(frame, open_water_km).tolist()
    periods = [str(value).strip() for value in frame["wind_averaging"].tolist()]
    target = np.empty(len(frame), dtype=float)
    scale = np.empty(len(frame), dtype=float)
    notes: list[str] = []
    flags: list[str] = []
    seconds: list[int | None] = []
    for index, (period, land) in enumerate(zip(periods, status, strict=True)):
        duration = wind_averaging_seconds(period)
        seconds.append(duration)
        if duration == 60:
            hold = land == "true" and choice == "raw"
            if hold:
                target[index] = raw[index]
                scale[index] = 1.0
                notes.append(HELD_RAW_NOTE)
                flags.append("true")
            else:
                target[index] = raw[index] * float(factor)
                scale[index] = float(factor)
                notes.append(note)
                flags.append(land)
        else:
            target[index] = raw[index]
            scale[index] = 1.0
            notes.append("")
            flags.append("")
    out = frame.copy()
    out["wind_speed_kt"] = raw
    out["wind_kt"] = raw
    out["wind_kt_3min"] = target
    out["wind_kt_scale"] = scale
    out["wind_conversion"] = notes
    out["conversion_exposure_flag"] = flags
    out["near_land"] = status
    out["wind_averaging_s"] = pd.array(seconds, dtype="Int64")
    return out


def run_conversion_note(frame: pd.DataFrame) -> str | None:
    """The single conversion note on this frame, or null when nothing was scaled."""
    if "wind_conversion" not in frame.columns or frame.empty:
        return None
    notes = []
    for value in frame["wind_conversion"].tolist():
        if _blank(value):
            continue
        text = str(value).strip()
        if text.lower() in {"nan", "none", "null"}:
            continue
        notes.append(text)
    notes = sorted(set(notes))
    if not notes:
        return None
    if len(notes) == 1:
        return notes[0]
    return " | ".join(notes)


def back_converted_prediction(prediction: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Map a 3-minute prediction back to the raw label.

    Converted 1-minute rows use ``prediction / factor``. Pass-through rows
    have scale 1. Near-land rows held at the raw label also have scale 1.
    """
    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    weights = np.asarray(scale, dtype=np.float64).reshape(-1)
    if pred.shape != weights.shape:
        raise ValueError("prediction and wind_kt_scale must have the same length")
    if weights.size and (not np.isfinite(weights).all() or np.any(weights <= 0)):
        raise ValueError("wind_kt_scale must be a positive finite multiplier")
    return pred / weights
