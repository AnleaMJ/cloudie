"""Build a Cloudie manifest from the cropped INCYDE IR set.

Zenodo record 8015544, file ``Cropped_cyclone_data.zip``, MD5
``ed5dac254c689e5a327dee7580324347``. This module does not download the
zip and CI must not either. ``verify_zip_md5`` checks a local copy and
raises when the digest does not match.

Labels are the CSV ``MSWS kts`` values. They are raw IMD 3-minute
sustained winds, so every row is ``wind_averaging=3-min`` and
``wind_averaging_s=180``. Nothing converts those knots to a 1-minute wind.

Only original frames are kept. ``flip_h``, ``flip_v``, ``rot_90``,
``rot_180``, and ``rot_270`` copies are dropped.

``incyde/audit/exclusions.csv`` drops a frame only when ``default_exclude``
is true (the Sequence 42 wrong-date frames) unless ``drop_flagged`` is set.
``label_ne_imd_interp`` and ``outside_imd_track_span`` stay in the default
manifest.

The modeling split is ``proposed_split`` from ``incyde/audit/storm_split.csv``.
It is not redrawn. ``cv_group`` is ``cooccurrence_group`` so a later grouped
CV can keep concurrent storms in one fold. The INCYDE ``Set`` column is
stored as ``source_split`` and is not the modeling split.

``on_fix``, ``near_land``, and ``distance_to_land_km`` come from
``incyde/audit/frame_flags.csv``, joined on the original filename.
``on_fix`` stays the file's true or false. ``near_land`` and
``distance_to_land_km`` may be ``unknown``. Unknown is not turned into
false or zero.

Pixels are an 8-bit IR render, not calibrated brightness temperature.
There is no ``year_holdout_2019`` path in this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.schema import require_valid, wind_averaging_seconds

ZIP_MD5 = "ed5dac254c689e5a327dee7580324347"
ZENODO_RECORD = "8015544"
ZENODO_DOI = "10.5281/zenodo.8015544"
ZIP_NAME = "Cropped_cyclone_data.zip"
WIND_AVERAGING = "3-min"
WIND_AVERAGING_S = 180
LABEL_SOURCE = "incyde_csv_msws_imd_3min"
LABEL_SCOPE = "INSAT-only"
RAW_3MIN_METRICS = (
    "All metrics use raw 3-minute IMD labels. "
    "wind_averaging is 3-min and wind_averaging_s is 180 on every row. "
    "There is no 1-minute conversion."
)
COMPARABILITY = (
    "INSAT-3D/3DR only. Not comparable with NASA or TCIR tables. "
    "Not like-for-like with Agrawal et al. 2024. " + RAW_3MIN_METRICS
)
ON_FIX_DEFINITION = (
    "on_fix is joined from incyde/audit/frame_flags.csv on filename. "
    "It is true only when that file says true: the UTC timestamp equals "
    "an IMD best-track time_resolved for that storm. "
    "near_land and distance_to_land_km come from the same file. "
    "unknown stays unknown and is not defaulted to false or zero."
)
_SYNOPTIC_UTC_HOURS = frozenset({0, 3, 6, 9, 12, 15, 18, 21})
_MONTHS = {
    name: index
    for index, name in enumerate(
        ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"),
        start=1,
    )
}
_NAME_RE = re.compile(
    r"^(?P<day>\d{2})(?P<mon>[A-Za-z]{3})(?P<year>\d{4})_"
    r"(?P<hh>\d{2})(?P<mm>\d{2})_(?P<obj>\d+)"
    r"(?:_(?P<aug>flip_h|flip_v|rot_90|rot_180|rot_270))?\.jpe?g$",
    re.IGNORECASE,
)
_AUG_TYPES = frozenset({"flip_h", "flip_v", "rot_90", "rot_180", "rot_270"})
_TRUE_TEXT = frozenset({"1", "true", "yes", "y", "t"})
_SET_MAP = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
    "testing": "test",
}
AUDIT_DIR = Path(__file__).resolve().parent / "audit"


class IncydeDataError(ValueError):
    """The cropped INCYDE files cannot build a leak-free manifest."""


def verify_zip_md5(path: str | Path, expected: str = ZIP_MD5) -> str:
    """Return the MD5 hex digest. Raise when it does not match ``expected``."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    digest = hashlib.md5()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    found = digest.hexdigest()
    if found != expected:
        raise IncydeDataError(
            f"{file_path} md5 {found} does not match {expected}. "
            "The zip was not unpacked and no manifest was written."
        )
    return found


def parse_frame_name(filename: str) -> dict[str, object]:
    """Parse one INCYDE JPEG name. The clock is IST."""
    name = Path(filename).name
    match = _NAME_RE.match(name)
    if match is None:
        raise IncydeDataError(f"unrecognized INCYDE filename: {name}")
    month = _MONTHS.get(match.group("mon").upper())
    if month is None:
        raise IncydeDataError(f"unrecognized month in filename: {name}")
    ist = pd.Timestamp(
        year=int(match.group("year")),
        month=month,
        day=int(match.group("day")),
        hour=int(match.group("hh")),
        minute=int(match.group("mm")),
    )
    utc = ist - pd.Timedelta(hours=5, minutes=30)
    aug = (match.group("aug") or "").lower()
    if aug and aug not in _AUG_TYPES:
        raise IncydeDataError(f"unknown augmentation in filename: {name}")
    on_clock = utc.minute == 0 and utc.second == 0 and int(utc.hour) in _SYNOPTIC_UTC_HOURS
    return {
        "filename": name,
        "object_index": int(match.group("obj")),
        "aug_type": aug,
        "is_augmented": bool(aug),
        "timestamp": utc.strftime("%Y-%m-%dT%H:%M:00Z"),
        "filename_time_ist": ist.strftime("%Y-%m-%d %H:%M"),
        "on_fix_clock": on_clock,
    }


def _is_true(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in _TRUE_TEXT


def _labels_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.rename(columns={column: str(column).strip() for column in frame.columns})
    missing = {"Sequence", "MSWS kts", "filename"} - set(frame.columns)
    if missing:
        raise IncydeDataError(f"labels CSV missing {sorted(missing)}")
    return frame


def _storm_split_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = {"Sequence", "storm_id", "proposed_split", "cooccurrence_group"} - set(frame.columns)
    if missing:
        raise IncydeDataError(f"storm split missing {sorted(missing)}")
    frame = frame.copy()
    frame["Sequence"] = pd.to_numeric(frame["Sequence"], errors="raise").astype(int)
    unknown = set(frame["proposed_split"].astype(str)) - {"train", "val", "test"}
    if unknown:
        raise IncydeDataError(f"proposed_split has unknown values: {sorted(unknown)}")
    if frame["Sequence"].duplicated().any():
        raise IncydeDataError("storm split has a repeated Sequence")
    groups = frame.groupby("cooccurrence_group")["proposed_split"].nunique()
    crossed = groups[groups > 1]
    if not crossed.empty:
        raise IncydeDataError(
            "cooccurrence_group crosses proposed splits: "
            + ", ".join(str(item) for item in crossed.index.tolist()[:8])
        )
    return frame


def _exclusion_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = {"filename", "qc_flag", "default_exclude"} - set(frame.columns)
    if missing:
        raise IncydeDataError(f"exclusions missing {sorted(missing)}")
    out = frame.copy()
    out["filename"] = out["filename"].map(lambda value: Path(str(value)).name)
    if out["filename"].duplicated().any():
        raise IncydeDataError("exclusions.csv has a repeated filename")
    return out


def _flag_token(value: object, column: str, filename: str) -> str:
    """Return true, false, or unknown. Blank and any other text raise."""
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "t"}:
        return "true"
    if text in {"false", "0", "no", "n", "f"}:
        return "false"
    if text == "unknown":
        return "unknown"
    raise IncydeDataError(
        f"{filename}: {column} is {value!r}. Expected true, false, or unknown. "
        "Unknown is not defaulted."
    )


def read_frame_flags(path: str | Path) -> pd.DataFrame:
    """Read audit flags. Empty distance is unknown, not zero."""
    file_path = Path(path)
    if not file_path.is_file():
        raise IncydeDataError(f"missing frame flags {file_path}")
    raw = pd.read_csv(file_path, dtype=str, keep_default_na=False)
    missing = {"filename", "on_fix", "near_land", "distance_to_land_km"} - set(raw.columns)
    if missing:
        raise IncydeDataError(f"frame flags missing {sorted(missing)}")
    out = pd.DataFrame({"filename": raw["filename"].map(lambda value: Path(str(value)).name)})
    if out["filename"].duplicated().any() or out["filename"].eq("").any():
        raise IncydeDataError("frame flags have a blank or repeated filename")
    out["on_fix"] = [
        _flag_token(value, "on_fix", name)
        for value, name in zip(raw["on_fix"], out["filename"], strict=True)
    ]
    out["near_land"] = [
        _flag_token(value, "near_land", name)
        for value, name in zip(raw["near_land"], out["filename"], strict=True)
    ]
    distance = raw["distance_to_land_km"].astype(str).str.strip()
    blank = distance.eq("") | distance.str.lower().isin({"nan", "none", "<na>", "null"})
    out["distance_to_land_km"] = np.where(blank | distance.str.lower().eq("unknown"), "unknown", distance)
    return out


def _source_split(value: object) -> str:
    return _SET_MAP.get(str(value).strip().lower(), str(value).strip().lower())


def _image_dir(root: Path, labels_path: Path) -> Path:
    candidates = [
        root / "Cropped_cyclone_data" / "Cropped_cyclone_images",
        root / "Cropped_cyclone_images",
        labels_path.parent / "Cropped_cyclone_images",
        labels_path.parent,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise IncydeDataError(f"no image directory under {root}")


def _locate_labels(root: Path) -> Path:
    candidates = [
        root / "Cropped_cyclone_data" / "Cropped_cyclone_labels.csv",
        root / "Cropped_cyclone_labels.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    found = sorted(root.glob("**/Cropped_cyclone_labels.csv"))
    if len(found) == 1:
        return found[0]
    raise IncydeDataError(f"Cropped_cyclone_labels.csv not found under {root}")


def _record_image(path: Path) -> tuple[int, int, str, bool]:
    """Return height, width, mode, and whether the file is grey IR."""
    from PIL import Image

    with Image.open(path) as image:
        mode = image.mode
        width, height = image.size
        if mode in {"L", "I", "I;16", "F"}:
            return height, width, mode, True
        array = np.asarray(image)
    if array.ndim == 2:
        return height, width, mode, True
    if array.ndim == 3 and array.shape[-1] >= 3:
        grey = bool(
            np.array_equal(array[..., 0], array[..., 1])
            and np.array_equal(array[..., 0], array[..., 2])
        )
        return height, width, mode, grey
    raise IncydeDataError(f"unexpected image shape for {path}: {getattr(array, 'shape', None)}")


def build_manifest(
    root: str | Path,
    manifest_path: str | Path,
    *,
    storm_split_path: str | Path | None = None,
    exclusions_path: str | Path | None = None,
    labels_path: str | Path | None = None,
    drop_flagged: bool = False,
    check_images: bool = True,
    frame_flags_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Write a validated manifest of original frames and return it with a summary.

    ``image_path`` is absolute. The manifest is not committed.
    """
    root = Path(root)
    labels_file = Path(labels_path) if labels_path else _locate_labels(root)
    image_dir = _image_dir(root, labels_file)
    split_file = Path(storm_split_path) if storm_split_path else AUDIT_DIR / "storm_split.csv"
    exclusion_file = Path(exclusions_path) if exclusions_path else AUDIT_DIR / "exclusions.csv"
    flags_file = Path(frame_flags_path) if frame_flags_path else AUDIT_DIR / "frame_flags.csv"
    flags = read_frame_flags(flags_file).set_index("filename")
    storms = _storm_split_frame(split_file)
    exclusions = _exclusion_frame(exclusion_file)
    labels = _labels_frame(labels_file)

    parsed = [parse_frame_name(name) for name in labels["filename"].astype(str)]
    names = pd.DataFrame(parsed)
    work = labels.reset_index(drop=True).copy()
    for column in names.columns:
        work[column] = names[column].to_numpy()
    work["Sequence"] = pd.to_numeric(work["Sequence"], errors="raise").astype(int)
    work["wind_speed_kt"] = pd.to_numeric(work["MSWS kts"], errors="coerce")
    if work["wind_speed_kt"].isna().any():
        raise IncydeDataError("MSWS kts has non-numeric values")

    n_labels = int(len(work))
    augmented = work["is_augmented"].astype(bool)
    n_augmented = int(augmented.sum())
    originals = work.loc[~augmented].copy()
    if originals.empty:
        raise IncydeDataError("no original frames in the labels CSV")
    if originals["filename"].duplicated().any():
        raise IncydeDataError("duplicate original filename")

    exclusion_lookup = exclusions.set_index("filename")
    originals["qc_flag"] = originals["filename"].map(exclusion_lookup["qc_flag"]).fillna("").astype(str)
    default_mask = originals["filename"].map(exclusion_lookup["default_exclude"])
    originals["default_exclude"] = default_mask.map(
        lambda value: _is_true(value) if pd.notna(value) else False
    )
    n_default = int(originals["default_exclude"].sum())
    flagged = originals["qc_flag"].str.len() > 0
    n_flagged_kept = int((flagged & ~originals["default_exclude"]).sum())
    drop = flagged if drop_flagged else originals["default_exclude"].astype(bool)
    kept = originals.loc[~drop].copy()
    if kept.empty:
        raise IncydeDataError("every original frame was excluded")

    merged = kept.merge(storms, on="Sequence", how="left", validate="many_to_one", suffixes=("", "_storm"))
    missing_storm = merged["storm_id"].isna() | merged["proposed_split"].isna()
    if bool(missing_storm.any()):
        sequences = sorted(merged.loc[missing_storm, "Sequence"].unique().tolist())
        raise IncydeDataError(f"sequences missing from storm_split.csv: {sequences[:8]}")

    missing_flags = sorted(set(merged["filename"].astype(str)) - set(flags.index.astype(str)))
    if missing_flags:
        raise IncydeDataError(
            f"{len(missing_flags)} filenames are missing from frame flags, first: {missing_flags[:3]}. "
            "Missing on_fix is not defaulted."
        )
    on_fix_token = merged["filename"].map(flags["on_fix"]).astype(str)
    near_land = merged["filename"].map(flags["near_land"]).astype(str)
    distance = merged["filename"].map(flags["distance_to_land_km"]).astype(str)
    on_fix_value = [
        True if token == "true" else False if token == "false" else "unknown"
        for token in on_fix_token.tolist()
    ]
    paths = [str((image_dir / name).resolve()) for name in merged["filename"].astype(str)]
    missing_files = [path for path in paths if not Path(path).is_file()]
    if missing_files:
        raise FileNotFoundError(f"{len(missing_files)} original images missing, first: {missing_files[:3]}")

    basin = merged["basin"] if "basin" in merged.columns else ""
    manifest = pd.DataFrame(
        {
            "sample_id": merged["filename"].map(lambda value: Path(str(value)).stem).astype(str),
            "image_path": paths,
            "storm_id": merged["storm_id"].astype(str),
            "wind_speed_kt": merged["wind_speed_kt"],
            "wind_averaging": WIND_AVERAGING,
            "wind_averaging_s": WIND_AVERAGING_S,
            "is_synthetic": False,
            "split": merged["proposed_split"].astype(str),
            "source_split": merged["Set"].map(_source_split) if "Set" in merged.columns else "unspecified",
            "cv_group": merged["cooccurrence_group"].map(lambda value: str(int(value))),
            "on_fix": on_fix_value,
            "near_land": near_land.tolist(),
            "distance_to_land_km": distance.tolist(),
            "timestamp": merged["timestamp"].astype(str),
            "filename_time_ist": merged["filename_time_ist"].astype(str),
            "basin": basin.astype(str) if isinstance(basin, pd.Series) else basin,
            "sensor": "INSAT-3D/3DR",
            "channel": "IR",
            "pixel_units": "8bit_uncalibrated",
            "label_source": LABEL_SOURCE,
            "center_source": "dataset_precentered",
            "incyde_sequence": merged["Sequence"].astype(int),
            "object_index": merged["object_index"].astype(int),
            "qc_flag": merged["qc_flag"].astype(str),
            "storm_name": merged["name"].fillna("").astype(str) if "name" in merged.columns else "",
        }
    )
    if manifest["sample_id"].duplicated().any():
        raise IncydeDataError("duplicate sample_id after filtering")
    seconds = {wind_averaging_seconds(value) for value in manifest["wind_averaging"]}
    if seconds != {WIND_AVERAGING_S} or not (manifest["wind_averaging_s"] == WIND_AVERAGING_S).all():
        raise IncydeDataError(f"wind_averaging_s must be {WIND_AVERAGING_S} on every row")

    heights: list[int] = []
    widths: list[int] = []
    modes: list[str] = []
    if check_images:
        for image_path in manifest["image_path"]:
            height, width, mode, grey = _record_image(Path(image_path))
            if not grey:
                raise IncydeDataError(f"colour frame is not grey IR: {image_path}")
            heights.append(height)
            widths.append(width)
            modes.append(mode)
        manifest["native_height"] = heights
        manifest["native_width"] = widths
        manifest["source_mode"] = modes

    manifest = manifest.sort_values(["incyde_sequence", "timestamp", "object_index", "sample_id"]).reset_index(drop=True)
    validated = require_valid(manifest)
    for column in (
        "split",
        "source_split",
        "cv_group",
        "on_fix",
        "near_land",
        "distance_to_land_km",
        "timestamp",
        "wind_averaging_s",
        "qc_flag",
    ):
        validated[column] = manifest[column].to_numpy()
    validated["wind_averaging_s"] = WIND_AVERAGING_S

    by_split = {
        name: {
            "rows": int((validated["split"] == name).sum()),
            "storms": int(validated.loc[validated["split"] == name, "storm_id"].nunique()),
        }
        for name in ("train", "val", "test")
    }
    summary: dict[str, object] = {
        "label_scope": LABEL_SCOPE,
        "comparability": COMPARABILITY,
        "metrics_use": RAW_3MIN_METRICS,
        "on_fix_definition": ON_FIX_DEFINITION,
        "zenodo_record": ZENODO_RECORD,
        "zenodo_doi": ZENODO_DOI,
        "zip_md5": ZIP_MD5,
        "wind_averaging": WIND_AVERAGING,
        "wind_averaging_s": WIND_AVERAGING_S,
        "labels_rows": n_labels,
        "augmented_dropped": n_augmented,
        "originals_before_exclusions": int(len(originals)),
        "default_excluded_originals": n_default,
        "flagged_not_default_originals": n_flagged_kept,
        "drop_flagged": bool(drop_flagged),
        "rows_written": int(len(validated)),
        "storms_written": int(validated["storm_id"].nunique()),
        "on_fix_rows": int((on_fix_token == "true").sum()),
        "on_fix_false_rows": int((on_fix_token == "false").sum()),
        "on_fix_unknown_rows": int((on_fix_token == "unknown").sum()),
        "near_land_unknown_rows": int((near_land == "unknown").sum()),
        "distance_to_land_unknown_rows": int((distance == "unknown").sum()),
        "frame_flags_source": str(flags_file),
        "modeling_split": by_split,
        "cv_group": "cooccurrence_group",
        "split_source": str(split_file),
        "exclusions_source": str(exclusion_file),
        "pixel_units": "8bit_uncalibrated",
        "year_holdout_2019": "not_built",
    }
    if modes:
        summary["source_modes"] = sorted(set(modes))
        summary["native_height_min"] = int(min(heights))
        summary["native_height_max"] = int(max(heights))
        summary["native_width_min"] = int(min(widths))
        summary["native_width_max"] = int(max(widths))

    destination = Path(manifest_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    validated.to_csv(destination, index=False)
    destination.with_suffix(destination.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return validated, summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Unzipped Cropped_cyclone_data directory.")
    parser.add_argument("--out", type=Path, required=True, help="Manifest CSV to write. Do not commit it.")
    parser.add_argument("--storm-split", type=Path, default=None)
    parser.add_argument("--exclusions", type=Path, default=None)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument(
        "--drop-flagged",
        action="store_true",
        help="Also drop label_ne_imd_interp and outside_imd_track_span. The default run does not.",
    )
    parser.add_argument(
        "--zip",
        type=Path,
        default=None,
        help="Verify this zip's MD5 and stop when it does not match. Does not download or unzip.",
    )
    parser.add_argument("--frame-flags", type=Path, default=None)
    parser.add_argument("--skip-image-check", action="store_true")
    args = parser.parse_args(argv)
    if args.zip is not None:
        print(f"md5 {verify_zip_md5(args.zip)} matches {ZIP_MD5}")
    _frame, summary = build_manifest(
        args.root,
        args.out,
        storm_split_path=args.storm_split,
        exclusions_path=args.exclusions,
        labels_path=args.labels,
        drop_flagged=args.drop_flagged,
        check_images=not args.skip_image_check,
        frame_flags_path=args.frame_flags,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
