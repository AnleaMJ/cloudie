"""Join IBTrACS v04 DIST2LAND onto the TCIR manifest.

CPU only. Not part of CI. Does not download IBTrACS or the TCIR images.
Paths are configurable; nothing is hardcoded to a machine-specific location.

Each output row is keyed by ``sample_id``, the same key ``src.tcir`` writes
on the manifest. ``distance_to_land_km`` is IBTrACS ``DIST2LAND`` (km to
nearest land; 0 means landfall within 6 hours). ``near_land`` is true when
that distance is <= 20 km, false when it is greater than 20 km, and
unknown (with a blank distance) when there is no IBTrACS row for the
storm's USA_ATCF_ID at that exact time.

The TCIR storm id ``YYYYNNX`` is mapped to the IBTrACS ``USA_ATCF_ID``
``PPNNYYYY`` with basin letters L=AL, E=EP, W=WP, C=CP, I=IO, S=SH.
The join is exact on that id and ISO_TIME (0-second tolerance). No
interpolation and no position fallback.

Audit-only dependencies: pandas. The official NOAA NCEI IBTrACS v04r01
CSV is not committed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

AUDIT_DIR = Path(__file__).resolve().parent
OPEN_WATER_KM = 20.0
TCIR_LETTER_TO_ATCF = {
    "L": "AL",
    "E": "EP",
    "W": "WP",
    "C": "CP",
    "I": "IO",
    "S": "SH",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--manifest", type=Path, required=True, help="TCIR manifest.csv from src.tcir.")
    p.add_argument(
        "--ibtracs",
        type=Path,
        required=True,
        help="Official NCEI ibtracs.since1980.list.v04r01.csv (or the ALL list).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=AUDIT_DIR / "distance_to_land.csv",
        help="Output CSV (default: tcir/audit/distance_to_land.csv).",
    )
    p.add_argument(
        "--open-water-km",
        type=float,
        default=OPEN_WATER_KM,
        help=f"near_land is true at or below this distance (default: {OPEN_WATER_KM:g}).",
    )
    return p.parse_args()


def tcir_storm_id_to_usa_atcf_id(storm_id: str) -> str:
    if not isinstance(storm_id, str) or len(storm_id) != 7 or storm_id[-1] not in TCIR_LETTER_TO_ATCF:
        raise ValueError(f"unrecognised TCIR storm_id: {storm_id!r}")
    return f"{TCIR_LETTER_TO_ATCF[storm_id[-1]]}{storm_id[4:6]}{storm_id[:4]}"


def load_ibtracs_fixes(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        skiprows=[1],
        usecols=["USA_ATCF_ID", "ISO_TIME", "DIST2LAND"],
        low_memory=False,
    )
    frame["usa_atcf_id"] = frame["USA_ATCF_ID"].astype(str).str.strip().str.upper()
    frame = frame[frame["usa_atcf_id"].str.len() >= 8]
    frame["t"] = pd.to_datetime(frame["ISO_TIME"]).dt.tz_localize(None)
    frame["DIST2LAND"] = pd.to_numeric(frame["DIST2LAND"], errors="coerce")
    return frame.drop_duplicates(["usa_atcf_id", "t"], keep="first")[["usa_atcf_id", "t", "DIST2LAND"]]


def build(args: argparse.Namespace) -> pd.DataFrame:
    manifest = pd.read_csv(args.manifest, keep_default_na=False, na_values=[""], low_memory=False)
    required = {"sample_id", "storm_id", "timestamp"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")
    if manifest["sample_id"].duplicated().any():
        raise ValueError("manifest sample_id values are not unique")

    rows = manifest[["sample_id", "storm_id", "timestamp"]].copy()
    rows["usa_atcf_id"] = rows["storm_id"].map(tcir_storm_id_to_usa_atcf_id)
    rows["t"] = pd.to_datetime(rows["timestamp"], utc=True).dt.tz_localize(None)

    joined = rows.merge(load_ibtracs_fixes(args.ibtracs), on=["usa_atcf_id", "t"], how="left")
    if len(joined) != len(rows):
        raise RuntimeError("IBTrACS join created duplicate rows")

    known = joined["DIST2LAND"].notna()
    near = pd.Series("unknown", index=joined.index, dtype="object")
    near.loc[known & (joined["DIST2LAND"] <= args.open_water_km)] = "true"
    near.loc[known & (joined["DIST2LAND"] > args.open_water_km)] = "false"

    def _fmt(value: object) -> str:
        if pd.isna(value):
            return ""
        number = float(value)
        return str(int(number)) if number.is_integer() else f"{number:g}"

    out = pd.DataFrame(
        {
            "sample_id": joined["sample_id"],
            "distance_to_land_km": joined["DIST2LAND"].map(_fmt),
            "near_land": near,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    counts = out["near_land"].value_counts()
    print(f"wrote {len(out)} rows to {args.output}")
    print(
        "near_land true="
        f"{int(counts.get('true', 0))} false={int(counts.get('false', 0))} "
        f"unknown={int(counts.get('unknown', 0))}"
    )
    return out


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
