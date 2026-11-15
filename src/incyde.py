"""INCYDE loader for the cropped INSAT-3D North Indian Ocean set.

Images come from the Zenodo 8015544 file ``Cropped_cyclone_data.zip``
(MD5 ``ed5dac254c689e5a327dee7580324347``). This module never downloads
that zip. Point ``paths.incyde_root`` at an extracted copy.

Filename times are IST (UTC+05:30). ``Sequence`` in the labels CSV is the
storm key. The trailing ``obj`` index is not a storm id. The default split
and the exclusion list are ``incyde/audit/storm_split.csv`` and
``incyde/audit/exclusions.csv``.

``on_fix`` is not on those audit outputs, and the committed files do not
say which frames sit on an IMD best-track time, so this loader omits it.
There is also no per-frame landfall flag. ``near_land`` is unknown.
``basin`` value ``LAND`` is the IMD basin of a land depression, not a
frame landfall.

Only original frames and ``rot_180`` copies are kept. ``rot_180`` is
synthetic, stays with its source storm, and is train-only. Flips reverse
cyclone spin. ``rot_90`` and ``rot_270`` add black bars. Both are dropped.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.schema import require_valid
from src.splits import LeakageError

ZIP_MD5 = "ed5dac254c689e5a327dee7580324347"
WIND_AVERAGING = "3min"
WIND_AVERAGING_S = 180
CV_SEED = 42
YEAR_HOLDOUT_NOTE = (
    "indicative only: different training years and images from "
    "Agrawal et al. 2024, arXiv:2410.08218. Landfall frames are kept; "
    "the audit outputs have no landfall flag."
)
KEPT_AUG = frozenset({"", "rot_180"})
DROPPED_AUG = frozenset({"flip_h", "flip_v", "rot_90", "rot_270"})
_IST = timezone(timedelta(hours=5, minutes=30))
_MONTHS = {
    name: index
    for index, name in enumerate(
        ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"),
        start=1,
    )
}
_FILENAME = re.compile(
    r"^(?P<day>\d{2})(?P<mon>[A-Za-z]{3})(?P<year>\d{4})_"
    r"(?P<hour>\d{2})(?P<minute>\d{2})_(?P<obj>\d+)"
    r"(?:_(?P<aug>flip_h|flip_v|rot_90|rot_180|rot_270))?\.jpe?g$",
    re.IGNORECASE,
)
_AUG_SUFFIX = re.compile(
    r"_(?:flip_h|flip_v|rot_90|rot_180|rot_270)(?=\.jpe?g$)",
    re.IGNORECASE,
)
_STORM_ID = re.compile(r"^INCYDE-S\d+_IMD-(?P<year>\d{4})-\d+$")
_AUDIT_DIR = Path(__file__).resolve().parents[1] / "incyde" / "audit"
DEFAULT_STORM_SPLIT = _AUDIT_DIR / "storm_split.csv"
DEFAULT_EXCLUSIONS = _AUDIT_DIR / "exclusions.csv"


def imd_class_of(wind_kt: float) -> str:
    """IMD class on a 3-minute wind, in knots.

    D ≤27, DD ≤33, CS ≤47, SCS ≤63, VSCS ≤89, ESCS ≤119, SuCS ≥120.
    Below 17 kt is ``<17``. These are not the project's 17/34/48/64/91 bins.
    """
    wind = float(wind_kt)
    if wind < 17:
        return "<17"
    if wind <= 27:
        return "D"
    if wind <= 33:
        return "DD"
    if wind <= 47:
        return "CS"
    if wind <= 63:
        return "SCS"
    if wind <= 89:
        return "VSCS"
    if wind <= 119:
        return "ESCS"
    return "SuCS"


def storm_year(storm_id: object) -> int:
    """Year from ``INCYDE-Sxx_IMD-yyyy-nnn``. Filename dates are not used."""
    match = _STORM_ID.match(str(storm_id).strip())
    if match is None:
        raise ValueError(f"storm_id {storm_id!r} is not INCYDE-Sxx_IMD-yyyy-nnn")
    return int(match.group("year"))


def assert_one_storm_year(frame: pd.DataFrame) -> pd.Series:
    """Each storm id maps to one IMD year. A filename that crosses 1 January does not add a second year."""
    years = frame["storm_id"].map(storm_year)
    counts = pd.DataFrame({"storm_id": frame["storm_id"].astype(str), "year": years}).groupby("storm_id")["year"].nunique()
    if bool((counts > 1).any()):
        raise LeakageError("a storm_id maps to more than one year")
    return years


def archive_md5(path: str | Path) -> str:
    """MD5 of a local file. Does not download."""
    digest = hashlib.md5(usedforsecurity=False)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_archive_md5(path: str | Path) -> None:
    """Refuse a cropped-zip copy whose MD5 is not the Zenodo value."""
    found = archive_md5(path)
    if found != ZIP_MD5:
        raise ValueError(f"INCYDE archive md5 is {found}, expected {ZIP_MD5}")


def _optional_path(value: object, default: Path) -> Path:
    if value is None:
        return default
    text = str(value).strip()
    if text.lower() in {"", "none", "null"}:
        return default
    return Path(text)


def read_storm_split(path: str | Path | None = None) -> pd.DataFrame:
    """One row per Sequence. ``proposed_split`` is Bean's grouped split."""
    table = pd.read_csv(_optional_path(path, DEFAULT_STORM_SPLIT))
    table.columns = [str(column).strip() for column in table.columns]
    required = {"Sequence", "storm_id", "proposed_split", "cooccurrence_group"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"storm_split.csv missing {missing}")
    out = table.copy()
    out["Sequence"] = pd.to_numeric(out["Sequence"], errors="raise").astype(int)
    out["storm_id"] = out["storm_id"].astype(str).str.strip()
    out["proposed_split"] = out["proposed_split"].astype(str).str.strip()
    out["cooccurrence_group"] = pd.to_numeric(out["cooccurrence_group"], errors="raise").astype(int)
    if out["storm_id"].duplicated().any() or out["Sequence"].duplicated().any():
        raise ValueError("storm_split.csv repeats a storm")
    bad = sorted(set(out["proposed_split"]) - {"train", "val", "test"})
    if bad:
        raise ValueError(f"storm_split.csv has unknown splits {bad}")
    assert_one_storm_year(out)
    return out


def read_exclusions(path: str | Path | None = None) -> pd.DataFrame:
    table = pd.read_csv(_optional_path(path, DEFAULT_EXCLUSIONS))
    table.columns = [str(column).strip() for column in table.columns]
    required = {"filename", "qc_flag", "default_exclude"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"exclusions.csv missing {missing}")
    out = table.copy()
    out["filename"] = out["filename"].astype(str).map(lambda name: Path(name).name)
    out["qc_flag"] = out["qc_flag"].fillna("").astype(str)
    out["default_exclude"] = [_is_true(value) for value in out["default_exclude"].tolist()]
    if out["filename"].duplicated().any():
        raise ValueError("exclusions.csv repeats a filename")
    return out


def _is_true(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def parse_filename(filename: str) -> dict[str, object]:
    """Parse ``DDMONYYYY_HHMM_<obj>[_aug].jpg``. ``obj`` is not a storm id."""
    name = Path(str(filename)).name
    match = _FILENAME.match(name)
    if match is None:
        raise ValueError(
            f"INCYDE filename {name} does not match DDMONYYYY_HHMM_<obj>[_aug].jpg"
        )
    parts = match.groupdict()
    month = _MONTHS.get(parts["mon"].upper())
    if month is None:
        raise ValueError(f"INCYDE filename {name} has an unknown month")
    ist = datetime(
        int(parts["year"]),
        month,
        int(parts["day"]),
        int(parts["hour"]),
        int(parts["minute"]),
        tzinfo=_IST,
    )
    utc = ist.astimezone(timezone.utc)
    aug = "" if not parts["aug"] else parts["aug"].lower()
    return {
        "filename": name,
        "time_ist": ist.isoformat(),
        "time_utc": utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "object_index": int(parts["obj"]),
        "aug_type": aug,
        "source_frame_id": _AUG_SUFFIX.sub("", name),
    }


def _source_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Labels CSV or the audit manifest, reduced to one column set."""
    columns = {str(column).strip(): column for column in frame.columns}
    renamed = frame.rename(columns={raw: clean for clean, raw in columns.items()})
    if "Sequence" in renamed.columns and ("MSWS kts" in renamed.columns or "label_kt" in renamed.columns):
        wind_column = "MSWS kts" if "MSWS kts" in renamed.columns else "label_kt"
        filename = renamed["filename"] if "filename" in renamed.columns else renamed["file"]
        out = pd.DataFrame(
            {
                "sequence": pd.to_numeric(renamed["Sequence"], errors="raise").astype(int),
                "filename": filename.astype(str),
                "wind_kt": pd.to_numeric(renamed[wind_column], errors="coerce"),
                "source_category": renamed["Category"].astype(str) if "Category" in renamed.columns else "",
                "incyde_set": renamed["Set"].astype(str) if "Set" in renamed.columns else "",
                "given_storm_id": renamed["storm_id"].astype(str) if "storm_id" in renamed.columns else pd.NA,
            }
        )
        return out
    if "storm_id" in renamed.columns and "label_kt" in renamed.columns:
        filename = renamed["filename"] if "filename" in renamed.columns else renamed["file"]
        sequence = renamed["incyde_sequence"] if "incyde_sequence" in renamed.columns else pd.Series(pd.NA, index=renamed.index)
        category = renamed["label_class_imd"] if "label_class_imd" in renamed.columns else ""
        return pd.DataFrame(
            {
                "sequence": pd.to_numeric(sequence, errors="coerce"),
                "filename": filename.astype(str),
                "wind_kt": pd.to_numeric(renamed["label_kt"], errors="coerce"),
                "source_category": category.astype(str) if not isinstance(category, str) else category,
                "incyde_set": renamed["incyde_set"].astype(str) if "incyde_set" in renamed.columns else "",
                "given_storm_id": renamed["storm_id"].astype(str),
            }
        )
    raise ValueError(
        "INCYDE manifest needs the labels columns (Sequence, MSWS kts, filename) "
        "or the audit columns (storm_id, label_kt, file)"
    )


def _locate_labels(root: Path) -> Path:
    candidates = [
        root / "Cropped_cyclone_labels.csv",
        root / "Cropped_cyclone_data" / "Cropped_cyclone_labels.csv",
        root / "labels.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"No INCYDE labels CSV under {root}. This loader does not download imagery."
    )


def locate_image(root: Path, filename: str) -> Path:
    name = Path(filename).name
    candidates = [
        root / name,
        root / "Cropped_cyclone_images" / name,
        root / "Cropped_cyclone_data" / "Cropped_cyclone_images" / name,
        root / "Cropped_cyclone_data" / name,
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"INCYDE image {name} is missing under {root}. This loader does not download imagery."
    )


def _qc_tokens(flag: object) -> set[str]:
    text = "" if flag is None or (isinstance(flag, float) and np.isnan(flag)) else str(flag)
    return {part.strip() for part in text.split(";") if part.strip()}


def _drop_reason(aug: str, exclude: bool, qc_flag: str, drop_qc_flags: set[str]) -> str | None:
    if exclude:
        return "default_exclude"
    if aug in DROPPED_AUG:
        return f"dropped aug {aug}"
    if aug not in KEPT_AUG:
        return f"unknown aug {aug or 'blank'}"
    if drop_qc_flags and _qc_tokens(qc_flag) & drop_qc_flags:
        return "drop_qc_flags"
    return None


def read_rows(
    root: str | Path,
    *,
    manifest_csv: str | Path | None = None,
    storm_split_csv: str | Path | None = None,
    exclusions_csv: str | Path | None = None,
    drop_qc_flags: list[str] | tuple[str, ...] | None = None,
    check_images: bool = True,
) -> tuple[pd.DataFrame, int]:
    """Frame rows before the final split. ``rot_180`` is still present.

    Flips, ``rot_90``, ``rot_270``, and ``default_exclude`` rows are gone.
    The returned count is how many source rows were dropped here.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"INCYDE root {root} is missing. Set paths.incyde_root to the extracted "
            "Cropped_cyclone_data directory. This entrypoint does not download imagery."
        )
    label_path = Path(manifest_csv) if manifest_csv not in (None, "", "null") else _locate_labels(root)
    if not label_path.is_file():
        raise FileNotFoundError(f"INCYDE manifest {label_path} is missing. Nothing is downloaded.")
    storms = read_storm_split(storm_split_csv)
    exclusions = read_exclusions(exclusions_csv)
    source = _source_table(pd.read_csv(label_path, encoding="utf-8-sig"))
    parsed = [parse_filename(name) for name in source["filename"].tolist()]
    parsed_frame = pd.DataFrame(parsed).drop(columns=["filename"])
    work = pd.concat([source.reset_index(drop=True), parsed_frame], axis=1)
    work["filename"] = work["filename"].map(lambda name: Path(str(name)).name)
    by_sequence = storms.set_index("Sequence")
    by_storm = storms.drop_duplicates("storm_id").set_index("storm_id")
    storm_ids: list[str] = []
    proposed: list[str] = []
    groups: list[int] = []
    names: list[str] = []
    basins: list[str] = []
    for sequence, given in zip(work["sequence"].tolist(), work["given_storm_id"].tolist(), strict=True):
        row = None
        if pd.notna(sequence) and int(sequence) in by_sequence.index:
            row = by_sequence.loc[int(sequence)]
        elif isinstance(given, str) and given in by_storm.index:
            row = by_storm.loc[given]
        if row is None:
            raise ValueError(f"no storm_split row for sequence {sequence!r} storm {given!r}")
        if isinstance(row, pd.DataFrame):
            raise ValueError(f"storm_split matched more than one row for sequence {sequence}")
        storm_id = str(row["storm_id"])
        if isinstance(given, str) and given not in {"", "nan", "<NA>"} and given != storm_id:
            raise ValueError(f"manifest storm_id {given} does not match storm_split.csv ({storm_id})")
        storm_ids.append(storm_id)
        proposed.append(str(row["proposed_split"]))
        groups.append(int(row["cooccurrence_group"]))
        names.append("" if "name" not in row.index or pd.isna(row["name"]) else str(row["name"]))
        basins.append("" if "basin" not in row.index or pd.isna(row["basin"]) else str(row["basin"]))
    work["storm_id"] = storm_ids
    work["proposed_split"] = proposed
    work["cooccurrence_group"] = groups
    work["name"] = names
    work["basin"] = basins
    qc = exclusions.set_index("filename")
    flags = []
    exclude = []
    for filename in work["filename"].tolist():
        if filename in qc.index:
            flags.append(str(qc.at[filename, "qc_flag"]))
            exclude.append(bool(qc.at[filename, "default_exclude"]))
        else:
            flags.append("")
            exclude.append(False)
    work["qc_flag"] = flags
    drop_flags = {str(flag).strip() for flag in (drop_qc_flags or []) if str(flag).strip()}
    reasons = [
        _drop_reason(aug, dropped, flag, drop_flags)
        for aug, dropped, flag in zip(work["aug_type"], exclude, work["qc_flag"], strict=True)
    ]
    dropped_count = sum(reason is not None for reason in reasons)
    kept = work.loc[[reason is None for reason in reasons]].reset_index(drop=True)
    if kept.empty:
        raise ValueError("INCYDE filters removed every row")
    if kept["filename"].duplicated().any():
        raise ValueError("INCYDE labels repeat a filename")
    years = assert_one_storm_year(kept)
    image_paths: list[str] = []
    heights: list[int] = []
    widths: list[int] = []
    if check_images:
        from PIL import Image

        for filename in kept["filename"].tolist():
            path = locate_image(root, filename)
            image_paths.append(path.resolve().as_posix())
            with Image.open(path) as image:
                widths.append(int(image.width))
                heights.append(int(image.height))
    else:
        image_paths = kept["filename"].tolist()
        heights = [0] * len(kept)
        widths = [0] * len(kept)
    frame = pd.DataFrame(
        {
            "sample_id": [Path(name).stem for name in kept["filename"].tolist()],
            "image_path": image_paths,
            "storm_id": kept["storm_id"].astype(str),
            "wind_speed_kt": kept["wind_kt"].to_numpy(dtype=float),
            "wind_kt": kept["wind_kt"].to_numpy(dtype=float),
            "wind_averaging": WIND_AVERAGING,
            "wind_averaging_s": WIND_AVERAGING_S,
            "is_synthetic": [aug == "rot_180" for aug in kept["aug_type"].tolist()],
            "imd_class": [imd_class_of(wind) for wind in kept["wind_kt"].tolist()],
            "split": kept["proposed_split"].astype(str),
            "proposed_split": kept["proposed_split"].astype(str),
            "cooccurrence_group": kept["cooccurrence_group"].astype(int),
            "time_utc": kept["time_utc"].astype(str),
            "time_ist": kept["time_ist"].astype(str),
            "storm_year": years.to_numpy(dtype=int),
            "aug_type": kept["aug_type"].astype(str),
            "source_frame_id": kept["source_frame_id"].astype(str),
            "object_index": kept["object_index"].astype(int),
            "qc_flag": kept["qc_flag"].astype(str),
            "incyde_sequence": kept["sequence"].astype(int),
            "incyde_set": kept["incyde_set"].astype(str),
            "source_category": kept["source_category"].astype(str),
            "name": kept["name"].astype(str),
            "basin": kept["basin"].astype(str),
            "native_height": heights,
            "native_width": widths,
            "orig_size": [height if height == width else -1 for height, width in zip(heights, widths, strict=True)],
            "sensor": "INSAT-3D",
            "channel": "IR",
            "pixel_units": "jpeg_uncalibrated",
            "label_source": "incyde_imd_3min",
            "center_source": "dataset_precentered",
        }
    )
    # on_fix and landfall are omitted: the audit CSVs do not carry them.
    return require_valid(frame), int(dropped_count)


def build_cv_folds(storms: pd.DataFrame, n_folds: int, seed: int) -> pd.DataFrame:
    """Storm-grouped folds over train+val only. Test storms are absent.

    Groups are shuffled with ``seed`` and dealt round-robin, so every storm
    in one ``cooccurrence_group`` shares a fold. The training path passes
    ``incyde.cv_seed`` (42 in the YAML).
    """
    folds = int(n_folds)
    if folds < 2:
        raise ValueError("incyde.n_folds must be at least 2")
    pool = storms.loc[storms["proposed_split"].astype(str).isin(["train", "val"])].drop_duplicates("storm_id")
    if pool.empty:
        raise ValueError("storm_cv needs train and val storms")
    groups = sorted({int(group) for group in pool["cooccurrence_group"].tolist()})
    order = np.random.default_rng(int(seed)).permutation(len(groups))
    shuffled = [groups[int(index)] for index in order]
    group_fold = {group: index % folds for index, group in enumerate(shuffled)}
    out = pd.DataFrame(
        {
            "storm_id": pool["storm_id"].astype(str).tolist(),
            "fold": [int(group_fold[int(group)]) for group in pool["cooccurrence_group"].tolist()],
        }
    )
    return out.sort_values(["fold", "storm_id"]).reset_index(drop=True)


def read_cv_folds(path: str | Path) -> pd.DataFrame:
    """Use a folds file verbatim. Columns are ``storm_id`` and ``fold``."""
    table = pd.read_csv(path)
    table.columns = [str(column).strip() for column in table.columns]
    if "storm_id" not in table.columns or "fold" not in table.columns:
        raise ValueError("cv folds csv needs columns storm_id and fold")
    out = pd.DataFrame(
        {
            "storm_id": table["storm_id"].astype(str).str.strip(),
            "fold": pd.to_numeric(table["fold"], errors="raise").astype(int),
        }
    )
    if out["storm_id"].duplicated().any():
        raise ValueError("cv folds csv repeats a storm_id")
    return out.reset_index(drop=True)


def validate_cv_folds(folds: pd.DataFrame, storms: pd.DataFrame) -> None:
    """No test storm, and no cooccurrence group spread across folds."""
    test_ids = set(storms.loc[storms["proposed_split"].astype(str) == "test", "storm_id"].astype(str))
    listed = set(folds["storm_id"].astype(str))
    leaked = sorted(listed & test_ids)
    if leaked:
        raise LeakageError(f"cv folds include test storms {leaked[:8]}")
    pool = storms.loc[storms["proposed_split"].astype(str).isin(["train", "val"])].drop_duplicates("storm_id")
    pool_ids = set(pool["storm_id"].astype(str))
    missing = sorted(pool_ids - listed)
    extra = sorted(listed - pool_ids)
    if missing:
        raise ValueError(f"cv folds omit train/val storms {missing[:8]}")
    if extra:
        raise ValueError(f"cv folds name storms that are not train/val {extra[:8]}")
    merged = pool.assign(storm_id=pool["storm_id"].astype(str)).merge(folds, on="storm_id", how="left")
    spread = merged.groupby("cooccurrence_group")["fold"].nunique()
    if bool((spread > 1).any()):
        groups = [int(group) for group in spread.index[spread > 1].tolist()]
        raise LeakageError(f"cooccurrence_group spans folds {groups[:8]}")


def _folds_for(storms: pd.DataFrame, n_folds: int, seed: int, cv_folds_csv: str | Path | None) -> pd.DataFrame:
    if cv_folds_csv not in (None, "", "null"):
        folds = read_cv_folds(cv_folds_csv)
    else:
        folds = build_cv_folds(storms, n_folds, seed)
    validate_cv_folds(folds, storms)
    return folds


def _assert_storms(frame: pd.DataFrame) -> None:
    per_storm = frame.groupby(frame["storm_id"].astype(str))["split"].nunique()
    if bool((per_storm > 1).any()):
        raise LeakageError("a storm is in more than one split")
    synthetic = frame["is_synthetic"].astype(bool) & frame["split"].astype(str).ne("train")
    if bool(synthetic.any()):
        raise LeakageError("synthetic INCYDE rows are train-only")
    bad_aug = sorted(set(frame["aug_type"].astype(str)) - KEPT_AUG)
    if bad_aug:
        raise LeakageError(f"INCYDE kept dropped augmentations {bad_aug}")


def _assert_groups(frame: pd.DataFrame) -> None:
    spread = frame.groupby("cooccurrence_group")["split"].nunique()
    if bool((spread > 1).any()):
        raise LeakageError("a cooccurrence_group is in more than one split")


def assign_incyde_split(
    frame: pd.DataFrame,
    strategy: str,
    *,
    n_folds: int = 5,
    cv_val_fold: int = 0,
    cv_seed: int = CV_SEED,
    cv_folds_csv: str | Path | None = None,
) -> tuple[pd.DataFrame, int]:
    """Write ``split`` and drop synthetic rows that are not in train.

    ``storm_wise`` uses ``proposed_split`` (60/14/17 on the audit file).
    ``storm_cv`` keeps test storms out of the folds. ``year_holdout_2019``
    puts 2019 in test and 2021 in val. The year comes from ``storm_id``.
    """
    name = str(strategy).strip().lower()
    work = frame.copy()
    if "proposed_split" not in work.columns:
        raise ValueError("INCYDE rows need proposed_split from storm_split.csv")
    storms = work.groupby("storm_id", as_index=False).agg(
        proposed_split=("proposed_split", "first"),
        cooccurrence_group=("cooccurrence_group", "first"),
    )
    assert_one_storm_year(work)
    if name in {"storm_wise", "proposed", "bean"}:
        work["split"] = work["proposed_split"].astype(str)
        _assert_groups(work)
    elif name == "storm_cv":
        using_file = cv_folds_csv not in (None, "", "null")
        folds = _folds_for(storms, n_folds, cv_seed, cv_folds_csv)
        chosen = int(cv_val_fold)
        if not using_file and (chosen < 0 or chosen >= int(n_folds)):
            raise ValueError("incyde.cv_val_fold is outside 0 .. n_folds-1")
        fold_of = dict(zip(folds["storm_id"].astype(str), folds["fold"].astype(int), strict=True))
        splits = []
        for storm_id, proposed in zip(work["storm_id"].astype(str), work["proposed_split"].astype(str), strict=True):
            if proposed == "test":
                splits.append("test")
            else:
                splits.append("val" if fold_of[storm_id] == chosen else "train")
        work["split"] = splits
        work["cv_fold"] = work["storm_id"].astype(str).map(lambda storm_id: fold_of.get(storm_id, pd.NA))
        _assert_groups(work)
    elif name == "year_holdout_2019":
        years = work["storm_id"].map(storm_year)
        work["split"] = np.where(years.eq(2019), "test", np.where(years.eq(2021), "val", "train"))
    else:
        raise LeakageError(f"unknown INCYDE split strategy {strategy}")
    synthetic_eval = work["is_synthetic"].astype(bool) & work["split"].astype(str).ne("train")
    dropped = int(synthetic_eval.sum())
    work = work.loc[~synthetic_eval].reset_index(drop=True)
    _assert_storms(work)
    for split_name in ("train", "val", "test"):
        if split_name not in set(work["split"].astype(str)):
            raise LeakageError(f"INCYDE split {name} produced no {split_name} rows")
    return work, dropped


def load_frame(config: dict) -> tuple[pd.DataFrame, int]:
    """Rows for ``python -m src.train --dataset incyde``. Nothing is downloaded."""
    paths = config["paths"]
    block = config["incyde"]
    archive = block["archive"]
    if archive not in (None, "", "null"):
        assert_archive_md5(archive)
    frame, dropped = read_rows(
        paths["incyde_root"],
        manifest_csv=block["manifest_csv"],
        storm_split_csv=block["storm_split_csv"],
        exclusions_csv=block["exclusions_csv"],
        drop_qc_flags=list(block["drop_qc_flags"] or []),
        check_images=True,
    )
    assigned, synthetic_dropped = assign_incyde_split(
        frame,
        str(config["split"]["strategy"]),
        n_folds=int(block["n_folds"]),
        cv_val_fold=int(block["cv_val_fold"]),
        cv_seed=int(block["cv_seed"]),
        cv_folds_csv=block["cv_folds_csv"],
    )
    return assigned, int(dropped) + int(synthetic_dropped)


def main(argv: list[str] | None = None) -> Path:
    """Write a manifest from a local extracted tree. Does not download."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest-csv", type=Path, default=None)
    parser.add_argument("--storm-split", type=Path, default=None)
    parser.add_argument("--exclusions", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strategy", default="storm_wise")
    parser.add_argument("--archive", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.archive is not None:
        assert_archive_md5(args.archive)
    frame, dropped = read_rows(
        args.root,
        manifest_csv=args.manifest_csv,
        storm_split_csv=args.storm_split,
        exclusions_csv=args.exclusions,
    )
    assigned, synthetic_dropped = assign_incyde_split(frame, args.strategy)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    assigned.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(assigned)} rows, dropped {dropped + synthetic_dropped}). Not a forecast.")
    return args.out


if __name__ == "__main__":
    main()
