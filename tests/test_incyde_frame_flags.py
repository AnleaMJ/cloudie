"""Checks on the committed INCYDE per-frame audit flags.

These tests read only files under incyde/audit/. They do not download
data, parse the IMD workbook, or import the builder's audit-only
dependencies. Augmented copies are not in frame_flags.csv; they inherit
flags from the source filename.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

AUDIT_DIR = Path(__file__).resolve().parents[1] / "incyde" / "audit"
FRAME_FLAGS = AUDIT_DIR / "frame_flags.csv"
EXCLUSIONS = AUDIT_DIR / "exclusions.csv"

COLUMNS = ["filename", "on_fix", "lat", "lon", "dist_to_coast_km", "near_land"]
N_ROWS = 21313
ON_FIX_TRUE = 1854
ON_FIX_FALSE = 19459
NEAR_LAND_TRUE = 5547
NEAR_LAND_FALSE = 15492
NEAR_LAND_UNKNOWN = 274

# Hand-checked against IMD Best_Tracks_Data_1982-2026.xlsx
# (imdtrack 0.3.0 time_resolved) and the IST-minus-5h30 filename clock.
# PHAILIN = IMD 2013-005, KYAAR = 2019-007, AMPHAN = 2020-001.
HAND_CHECKED_ON_FIX = {
    "08OCT2013_0830_1.jpg": "true",   # PHAILIN, 2013-10-08 03:00 UTC fix
    "08OCT2013_1130_1.jpg": "true",   # PHAILIN, 2013-10-08 06:00 UTC fix
    "08OCT2013_1200_1.jpg": "false",  # PHAILIN, 2013-10-08 06:30 UTC, no fix
    "08OCT2013_2330_1.jpg": "true",   # PHAILIN, 2013-10-08 18:00 UTC fix
    "24OCT2019_0830_1.jpg": "true",   # KYAAR, 2019-10-24 03:00 UTC fix
    "24OCT2019_0845_1.jpg": "false",  # KYAAR, 2019-10-24 03:15 UTC, no fix
    "16MAY2020_0530_1.jpg": "true",   # AMPHAN, 2020-05-16 00:00 UTC fix
    "16MAY2020_0600_1.jpg": "false",  # AMPHAN, 2020-05-16 00:30 UTC, no fix
    "20MAY2020_1730_1.jpg": "true",   # AMPHAN, 2020-05-20 12:00 UTC fix
    "13OCT2013_0530_1.jpg": "true",   # PHAILIN, 2013-10-13 00:00 UTC fix
    "12OCT2013_2330_1.jpg": "true",   # PHAILIN, 2013-10-12 18:00 UTC fix
}

AUG_SUFFIX = re.compile(r"_(flip_h|flip_v|rot_90|rot_180|rot_270)\.jpg$")


def _load_flags() -> pd.DataFrame:
    return pd.read_csv(
        FRAME_FLAGS,
        dtype={c: "string" for c in COLUMNS},
        keep_default_na=False,
        na_values=[],
    )


def _source_filename(name: str) -> str:
    return AUG_SUFFIX.sub(".jpg", name)


def test_frame_flags_row_count_and_columns() -> None:
    flags = _load_flags()
    assert list(flags.columns) == COLUMNS
    assert len(flags) == N_ROWS
    assert flags["filename"].is_unique
    assert flags["filename"].str.len().gt(0).all()
    assert not flags["filename"].map(lambda s: AUG_SUFFIX.search(s) is not None).any()
    assert set(flags["on_fix"].unique()) <= {"true", "false"}
    assert set(flags["near_land"].unique()) <= {"true", "false", "unknown"}
    assert (flags["on_fix"] == "true").sum() == ON_FIX_TRUE
    assert (flags["on_fix"] == "false").sum() == ON_FIX_FALSE
    assert (flags["near_land"] == "true").sum() == NEAR_LAND_TRUE
    assert (flags["near_land"] == "false").sum() == NEAR_LAND_FALSE
    assert (flags["near_land"] == "unknown").sum() == NEAR_LAND_UNKNOWN


def test_frame_flags_uses_exclusions_filename_keys() -> None:
    flags = _load_flags()
    exclusions = pd.read_csv(EXCLUSIONS, dtype={"filename": "string"})
    sources = exclusions["filename"].map(_source_filename)
    missing = set(sources) - set(flags["filename"])
    assert not missing, f"exclusion source filenames missing from frame_flags.csv: {sorted(missing)[:10]}"


def test_on_fix_matches_hand_checked_imd_fixes() -> None:
    flags = _load_flags().set_index("filename")
    missing = [name for name in HAND_CHECKED_ON_FIX if name not in flags.index]
    assert not missing, f"hand-checked filenames missing: {missing}"
    got = flags.loc[list(HAND_CHECKED_ON_FIX), "on_fix"].tolist()
    assert got == list(HAND_CHECKED_ON_FIX.values())


def test_unknown_position_rows_leave_coordinates_blank() -> None:
    flags = _load_flags()
    unknown = flags["near_land"] == "unknown"
    known = ~unknown
    assert (flags.loc[unknown, ["lat", "lon", "dist_to_coast_km"]] == "").all().all()
    assert (flags.loc[known, ["lat", "lon", "dist_to_coast_km"]] != "").all().all()
    dist = flags.loc[known, "dist_to_coast_km"].astype(float)
    assert ((flags.loc[known, "near_land"] == "true") == (dist <= 20.0)).all()
    # Inland AMPHAN (21.9N 88.4E) and post-landfall PHAILIN (20.5N 84.5E).
    inland = flags.set_index("filename").loc[
        ["20MAY2020_1730_1.jpg", "13OCT2013_0530_1.jpg"]
    ]
    assert (inland["near_land"] == "true").all()
    assert (inland["dist_to_coast_km"].astype(float) == 0.0).all()
