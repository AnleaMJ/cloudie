"""Manifest schema and row-level quality gates.

Required columns are ``sample_id``, ``image_path``, ``storm_id``, and
``wind_speed_kt``. ``require_valid`` raises ``SchemaError`` on a failed
gate. It does not warn and continue.

``wind_averaging`` must record one period for the whole manifest, such as
``1-min``, ``3-min``, or ``10-min``. Mixed or blank periods are rejected.
``is_synthetic`` must be an explicit true or false on every row so a
synthetic sample cannot be read as real.

Image statistics are pixel stats. This schema has no brightness-temperature
columns. Raw digital numbers are not named ``tb``. Those names are used
only after a brightness-temperature calibration table exists.

The 17/34/48/64/91 kt bins are this project's grouping, not official IMD
classes, and they are not a warning product. ``imd5_class`` stores that
project group.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

REQUIRED_COLUMNS = ("sample_id", "image_path", "storm_id", "wind_speed_kt")
FORBIDDEN_FEATURES = frozenset(
    {"wind_speed_kt", "r34_nmi", "rmw_nmi", "imd5_class", "intensity_bin"}
)
# Project grouping for metrics. Not official IMD classes and not a warning.
DEFAULT_EDGES_KT = (17.0, 34.0, 48.0, 64.0, 91.0)
DEFAULT_NAMES = ("D_DD", "CS", "SCS", "VSCS", "ESCS_SUCS")
DEFAULT_WIND_MIN_KT = 0.0
DEFAULT_WIND_MAX_KT = 200.0

_TRUE_TEXT = frozenset({"1", "true", "yes", "y", "t"})
_FALSE_TEXT = frozenset({"0", "false", "no", "n", "f"})
_MISSING_TEXT = frozenset({"", "nan", "none", "<na>", "nat", "null"})


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
    """Return this project's intensity group, or None below the lowest edge.

    The edges are Cloudie's grouping. They are not official IMD classes
    and they are not a warning product.
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


def _unpack_scalar(value: object) -> object:
    """Return a Python scalar when ``value`` is a numpy scalar."""
    if isinstance(value, (str, bytes, bytearray, bool)):
        return value
    item = getattr(value, "item", None)
    if not callable(item):
        return value
    try:
        unpacked = item()
    except (TypeError, ValueError):
        return value
    if type(unpacked) is type(value):
        return value
    return unpacked


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if missing is True:
        return True
    return False


def _sample_key(value: object) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if _is_missing(value):
        return ""
    if isinstance(value, str):
        return value.strip()
    value = _unpack_scalar(value)
    if isinstance(value, bool):
        return "True" if value else "False"
    if _is_missing(value):
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if value != value:
            return ""
        if value.is_integer():
            return str(int(value))
    return str(value).strip()


def parse_is_synthetic(value: object) -> bool:
    """Return an explicit synthetic flag. Ambiguous values raise.

    Null, blank, and unrecognized values are not treated as real data.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE_TEXT:
            return True
        if text in _FALSE_TEXT:
            return False
        if text in _MISSING_TEXT:
            raise SchemaError("is_synthetic is null; synthetic rows must be explicit")
        try:
            number = float(text)
        except ValueError:
            raise SchemaError(
                f"is_synthetic value {value!r} is not an explicit true or false"
            ) from None
        return parse_is_synthetic(number)
    value = _unpack_scalar(value)
    if isinstance(value, bool):
        return value
    if _is_missing(value):
        raise SchemaError("is_synthetic is null; synthetic rows must be explicit")
    if isinstance(value, str):
        return parse_is_synthetic(value)
    if isinstance(value, int) and not isinstance(value, bool):
        if value in (0, 1):
            return bool(value)
        raise SchemaError(
            f"is_synthetic value {value!r} is not an explicit true or false"
        )
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise SchemaError("is_synthetic is null; synthetic rows must be explicit")
        if value == 1.0:
            return True
        if value == 0.0:
            return False
    raise SchemaError(f"is_synthetic value {value!r} is not an explicit true or false")


def recorded_wind_averaging(frame: pd.DataFrame) -> str:
    """Return the single averaging period recorded on this manifest.

    The column is required. A blank row or two different periods raises.
    """
    if "wind_averaging" not in frame.columns:
        raise SchemaError(
            "wind_averaging is required; record one period for the manifest "
            "(for example 1-min, 3-min, or 10-min)"
        )
    if len(frame) == 0:
        raise SchemaError(
            "wind_averaging is required; record one period for the manifest"
        )
    periods: list[str] = []
    for value in frame["wind_averaging"].tolist():
        if isinstance(value, str):
            text = value.strip()
        elif _is_missing(value):
            text = ""
        else:
            unpacked = _unpack_scalar(value)
            if _is_missing(unpacked):
                text = ""
            else:
                text = str(unpacked).strip()
        if text.lower() in _MISSING_TEXT:
            raise SchemaError(
                "wind_averaging has nulls; record one period for every row"
            )
        periods.append(text)
    unique = sorted(set(periods))
    if len(unique) != 1:
        raise SchemaError(
            f"mixed wind_averaging periods {unique}; split or convert explicitly"
        )
    return unique[0]


def _synthetic_flags(frame: pd.DataFrame) -> list[bool]:
    if "is_synthetic" not in frame.columns:
        raise SchemaError(
            "is_synthetic is required so a synthetic row cannot be read as real"
        )
    return [parse_is_synthetic(value) for value in frame["is_synthetic"].tolist()]


def validate_manifest(
    frame: pd.DataFrame,
    wind_min_kt: float = DEFAULT_WIND_MIN_KT,
    wind_max_kt: float = DEFAULT_WIND_MAX_KT,
) -> QualityReport:
    """Collect quality-gate failures. Callers that load data use ``require_valid``."""
    errors: list[str] = []
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        return QualityReport(len(frame), 0, (f"missing required columns: {missing}",))

    keys = frame["sample_id"].map(_sample_key)
    null_id = frame["sample_id"].isna() | keys.eq("") | keys.str.lower().isin(_MISSING_TEXT)
    if bool(null_id.any()):
        errors.append("sample_id has nulls")
    if bool(keys.duplicated().any()):
        examples = keys.loc[keys.duplicated()].head(5).tolist()
        errors.append(f"duplicate sample_id, examples: {examples}")
    blank_storm = frame["storm_id"].astype(str).str.strip().str.len() == 0
    if frame["storm_id"].isna().any() or blank_storm.any():
        errors.append("storm_id has nulls")
    if frame["image_path"].isna().any() or (
        frame["image_path"].astype(str).str.strip().str.len() == 0
    ).any():
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

    try:
        recorded_wind_averaging(frame)
    except SchemaError as exc:
        errors.append(str(exc))
    try:
        _synthetic_flags(frame)
    except SchemaError as exc:
        errors.append(str(exc))

    n_storms = int(frame["storm_id"].nunique(dropna=True))
    return QualityReport(len(frame), n_storms, tuple(errors))


def require_valid(frame: pd.DataFrame, **kwargs: float) -> pd.DataFrame:
    """Return a copy that passed every manifest gate. Raises on failure."""
    report = validate_manifest(frame, **kwargs)
    if not report.ok:
        raise SchemaError("; ".join(report.errors))
    out = frame.copy()
    out["storm_id"] = out["storm_id"].astype(str).str.strip()
    out["sample_id"] = out["sample_id"].map(_sample_key).astype(str)
    out["image_path"] = out["image_path"].astype(str).str.strip()
    out["wind_speed_kt"] = pd.to_numeric(out["wind_speed_kt"])
    out["wind_averaging"] = recorded_wind_averaging(out)
    out["is_synthetic"] = _synthetic_flags(out)
    # Project grouping stored on imd5_class. Not an official IMD class code.
    out["imd5_class"] = out["wind_speed_kt"].map(wind_to_imd5)
    return out
