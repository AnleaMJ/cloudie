"""Checks on the committed TCIR IBTrACS distance-to-land flags.

These tests read only files under tcir/audit/. They do not download
IBTrACS or TCIR imagery, and they do not import the builder.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

AUDIT_DIR = Path(__file__).resolve().parents[1] / "tcir" / "audit"
FLAGS = AUDIT_DIR / "distance_to_land.csv"

COLUMNS = ["sample_id", "distance_to_land_km", "near_land"]
N_ROWS = 69867
NEAR_LAND_TRUE = 5027
NEAR_LAND_FALSE = 64768
NEAR_LAND_UNKNOWN = 72

# Hand-checked against the official NCEI IBTrACS v04r01 since1980 CSV.
# 200301L → USA_ATCF_ID AL012003 (Ana). 201011E → EP112010 (Hermine).
HAND_CHECKED = {
    "tcir_200301L_2003041800": ("1116", "false"),  # AL012003 2003-04-18 00:00 UTC, DIST2LAND 1116
    "tcir_200303L_2003062806": ("0", "true"),      # AL032003 2003-06-28 06:00 UTC, DIST2LAND 0
    "tcir_201011E_2010090406": ("11", "true"),     # EP112010 2010-09-04 06:00 UTC, DIST2LAND 11
    "tcir_201011E_2010090409": ("0", "true"),      # EP112010 2010-09-04 09:00 UTC, DIST2LAND 0
    "tcir_201011E_2010090418": ("", "unknown"),    # EP112010 track ends at 15:00 UTC
    "tcir_201622E_2016111718": ("", "unknown"),    # EP222016 is not in IBTrACS
}


def _load() -> pd.DataFrame:
    return pd.read_csv(
        FLAGS,
        dtype={c: "string" for c in COLUMNS},
        keep_default_na=False,
        na_values=[],
    )


def test_distance_to_land_row_count_and_columns() -> None:
    flags = _load()
    assert list(flags.columns) == COLUMNS
    assert len(flags) == N_ROWS
    assert flags["sample_id"].is_unique
    assert flags["sample_id"].str.startswith("tcir_").all()
    assert set(flags["near_land"].unique()) <= {"true", "false", "unknown"}
    assert (flags["near_land"] == "true").sum() == NEAR_LAND_TRUE
    assert (flags["near_land"] == "false").sum() == NEAR_LAND_FALSE
    assert (flags["near_land"] == "unknown").sum() == NEAR_LAND_UNKNOWN


def test_unknown_rows_leave_distance_blank() -> None:
    flags = _load()
    unknown = flags["near_land"] == "unknown"
    known = ~unknown
    assert (flags.loc[unknown, "distance_to_land_km"] == "").all()
    assert (flags.loc[known, "distance_to_land_km"] != "").all()
    dist = flags.loc[known, "distance_to_land_km"].astype(float)
    assert ((flags.loc[known, "near_land"] == "true") == (dist <= 20.0)).all()


def test_hand_checked_ibtracs_fixes() -> None:
    flags = _load().set_index("sample_id")
    missing = [name for name in HAND_CHECKED if name not in flags.index]
    assert not missing, f"hand-checked sample_id missing: {missing}"
    for sample_id, (distance, near_land) in HAND_CHECKED.items():
        row = flags.loc[sample_id]
        assert row["distance_to_land_km"] == distance
        assert row["near_land"] == near_land
