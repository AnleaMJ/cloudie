"""Build per-frame on_fix and near-land columns for INCYDE.

CPU only. Not part of CI. Does not download imagery. Paths are
configurable; nothing is hardcoded to a machine-specific location.

By default this writes one row per original-frame filename (the same
basename key as exclusions.csv). Augmented copies inherit the source
frame's flags. Pass --all-frames to write every manifest filename. Columns:

- on_fix: true iff the frame's UTC time equals an IMD best-track
  time_resolved value for that storm. Filename clocks are IST; UTC is
  already stored on the manifest (IST minus 5 h 30 min). The tolerance
  is 0 seconds: both clocks are minute-aligned, and widening the window
  to 5 minutes does not change any flag.
- lat, lon: storm-centre position at the frame time, taken from the
  coincident fix or linearly interpolated in time between the two
  neighbouring usable fixes. Left empty when a position cannot be
  derived without guessing.
- distance_to_land_km: geodesic-approximated distance to Natural Earth
  10m land (0 when the centre is on or over land). Empty when lat/lon
  are unknown.
- near_land: true if distance_to_land_km <= 20 or the centre is over land,
  false if the centre is more than 20 km offshore, unknown if there is
  no position.

Audit-only dependencies (not required for CI): pandas, numpy, imdtrack,
shapely, pyproj, pyshp.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AUDIT_DIR = Path(__file__).resolve().parent

NEAR_LAND_KM = 20.0
ON_FIX_TOLERANCE_S = 0
# Generous box around the North Indian Ocean. Points outside it still
# get a distance, computed against the unclipped land set.
NIO_BBOX = (20.0, -15.0, 130.0, 45.0)  # lon_min, lat_min, lon_max, lat_max
AEQD_CRS = "+proj=aeqd +lat_0=15 +lon_0=80 +datum=WGS84 +units=m +no_defs"
NE_LAND_NAME = "Natural Earth 10m land"
NE_LAND_VERSION = "5.1.1"
NE_LAND_URL = "https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip"
NE_LAND_LICENSE = "public domain"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Full INCYDE manifest.csv (not committed; ~103k rows).",
    )
    p.add_argument(
        "--imd-best-track",
        type=Path,
        required=True,
        help="IMD Best_Tracks_Data_1982-2026.xlsx, parsed with imdtrack.",
    )
    p.add_argument(
        "--land-shapefile",
        type=Path,
        required=True,
        help="Path to ne_10m_land.shp (Natural Earth 10m land).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=AUDIT_DIR / "frame_flags.csv.gz",
        help="Output CSV (default: incyde/audit/frame_flags.csv.gz).",
    )
    p.add_argument(
        "--near-land-km",
        type=float,
        default=NEAR_LAND_KM,
        help=f"near_land threshold in km (default: {NEAR_LAND_KM:g}).",
    )
    p.add_argument(
        "--originals-only",
        action="store_true",
        default=True,
        help="Write one row per original frame (default). Augmented copies "
        "inherit the source frame's flags by stripping the aug suffix.",
    )
    p.add_argument(
        "--all-frames",
        action="store_true",
        help="Write a row for every manifest filename, including augmented copies.",
    )
    return p.parse_args()


def _usable_fix(lat: float, lon: float, pos_suspect: bool) -> bool:
    return (
        not pos_suspect
        and lat is not None
        and lon is not None
        and math.isfinite(float(lat))
        and math.isfinite(float(lon))
    )


def locate_frames(
    times: np.ndarray,
    fix_times: np.ndarray,
    fix_lats: np.ndarray,
    fix_lons: np.ndarray,
    fix_usable: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (on_fix, lat, lon) for query times of one storm.

    times and fix_times are int64 nanoseconds (UTC, tz-naive).
    on_fix is exact equality with a fix time (0-second tolerance).
    lat/lon are NaN when they cannot be taken from a usable fix or from
    linear interpolation between two neighbouring usable fixes.
    """
    n = len(times)
    on_fix = np.zeros(n, dtype=bool)
    lat = np.full(n, np.nan, dtype=np.float64)
    lon = np.full(n, np.nan, dtype=np.float64)
    if len(fix_times) == 0:
        return on_fix, lat, lon

    # Exact coincidence (0-second tolerance). np.searchsorted + equality
    # is enough because both clocks are minute-aligned.
    idx = np.searchsorted(fix_times, times)
    in_range = (idx < len(fix_times)) & (idx >= 0)
    exact = np.zeros(n, dtype=bool)
    exact[in_range] = fix_times[np.clip(idx[in_range], 0, len(fix_times) - 1)] == times[in_range]
    # searchsorted can land past a match when times equal fix_times[i]
    # if idx points at that i; the check above covers it. Also catch the
    # case where searchsorted returns i+1 because of identical values:
    # for exact matches, searchsorted(side="left") returns the first
    # equal element, so equality at idx is the right test.
    on_fix[exact] = True
    usable_exact = exact & (idx < len(fix_times))
    usable_exact[usable_exact] &= fix_usable[idx[usable_exact]]
    lat[usable_exact] = fix_lats[idx[usable_exact]]
    lon[usable_exact] = fix_lons[idx[usable_exact]]

    # Interpolate only between neighbouring usable fixes, and never
    # across an unusable one.
    remaining = ~exact
    if not remaining.any() or len(fix_times) < 2:
        return on_fix, lat, lon

    # For a time strictly between fix[i] and fix[i+1], searchsorted
    # returns i+1.
    right = idx
    left = right - 1
    between = remaining & (left >= 0) & (right < len(fix_times))
    if not between.any():
        return on_fix, lat, lon
    left_i = left[between]
    right_i = right[between]
    ok = fix_usable[left_i] & fix_usable[right_i]
    # Refuse to interpolate if any unusable fix sits in (left, right),
    # which for neighbouring indices is just those two bounds.
    # left and right are already neighbours in the fix array.
    t = times[between]
    t0 = fix_times[left_i]
    t1 = fix_times[right_i]
    # Guard against duplicate fix times.
    span = t1 != t0
    ok &= span
    if not ok.any():
        return on_fix, lat, lon
    w = (t[ok].astype(np.float64) - t0[ok].astype(np.float64)) / (
        t1[ok].astype(np.float64) - t0[ok].astype(np.float64)
    )
    dest = np.flatnonzero(between)[ok]
    lat[dest] = fix_lats[left_i[ok]] + w * (fix_lats[right_i[ok]] - fix_lats[left_i[ok]])
    lon[dest] = fix_lons[left_i[ok]] + w * (fix_lons[right_i[ok]] - fix_lons[left_i[ok]])
    return on_fix, lat, lon


def _load_land(shapefile_path: Path):
    import shapefile
    from shapely.geometry import box, shape
    from shapely.ops import unary_union
    from pyproj import Transformer

    reader = shapefile.Reader(str(shapefile_path))
    geoms = []
    lon_min, lat_min, lon_max, lat_max = NIO_BBOX
    clip = box(lon_min, lat_min, lon_max, lat_max)
    for shp in reader.shapes():
        geom = shape(shp.__geo_interface__)
        if geom.is_empty:
            continue
        clipped = geom.intersection(clip)
        if not clipped.is_empty:
            geoms.append(clipped)
    if not geoms:
        raise RuntimeError(f"No land polygons intersect the NIO bbox in {shapefile_path}")
    land_wgs = unary_union(geoms)
    transformer = Transformer.from_crs("EPSG:4326", AEQD_CRS, always_xy=True)

    def _project(coords):
        x, y = transformer.transform(coords[:, 0], coords[:, 1])
        return np.column_stack([x, y])

    from shapely import prepare, transform as shapely_transform

    land_m = shapely_transform(land_wgs, _project)

    prepare(land_wgs)
    prepare(land_m)
    return land_wgs, land_m, transformer


def distances_to_land(
    lats: np.ndarray,
    lons: np.ndarray,
    land_wgs,
    land_m,
    transformer,
) -> np.ndarray:
    """Distance in km to Natural Earth land. 0 if on or over land. NaN if no position."""
    from shapely.geometry import Point

    out = np.full(len(lats), np.nan, dtype=np.float64)
    known = np.isfinite(lats) & np.isfinite(lons)
    if not known.any():
        return out
    xs, ys = transformer.transform(lons[known], lats[known])
    dist = np.empty(known.sum(), dtype=np.float64)
    pts_m = [Point(x, y) for x, y in zip(xs, ys)]
    pts_wgs = [Point(lon, lat) for lon, lat in zip(lons[known], lats[known])]
    for i, (pt_wgs, pt_m) in enumerate(zip(pts_wgs, pts_m)):
        if land_wgs.covers(pt_wgs):
            dist[i] = 0.0
        else:
            dist[i] = land_m.distance(pt_m) / 1000.0
    out[known] = dist
    return out


def _csv_bool(value: bool) -> str:
    return "true" if value else "false"


def build(args: argparse.Namespace) -> pd.DataFrame:
    import imdtrack

    manifest = pd.read_csv(args.manifest, keep_default_na=False, na_values=[""], low_memory=False)
    required = {"file", "timestamp", "imd_storm_id"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")

    manifest = manifest.copy()
    manifest["filename"] = manifest["file"].str.rsplit("/", n=1).str[-1]
    if manifest["filename"].duplicated().any():
        raise ValueError("manifest filenames are not unique")
    manifest["ts"] = pd.to_datetime(manifest["timestamp"], utc=True).dt.tz_localize(None)

    bt = imdtrack.load(path=str(args.imd_best_track)).observations
    bt = bt.copy()
    bt["t"] = pd.to_datetime(bt["time_resolved"])

    unique = (
        manifest[["imd_storm_id", "ts"]]
        .drop_duplicates()
        .sort_values(["imd_storm_id", "ts"])
        .reset_index(drop=True)
    )

    on_fix_u = np.zeros(len(unique), dtype=bool)
    lat_u = np.full(len(unique), np.nan, dtype=np.float64)
    lon_u = np.full(len(unique), np.nan, dtype=np.float64)

    for sid, grp in unique.groupby("imd_storm_id", sort=False):
        fixes = bt[bt["storm_id"] == sid].sort_values("t")
        # If two observations share a time, keep the first usable one.
        if fixes["t"].duplicated().any():
            usable_rank = fixes.apply(
                lambda r: _usable_fix(r["lat"], r["lon"], bool(r["pos_suspect"])),
                axis=1,
            )
            fixes = fixes.assign(_usable=usable_rank).sort_values(
                ["t", "_usable"], ascending=[True, False]
            )
            fixes = fixes.drop_duplicates("t", keep="first")
        fix_times = fixes["t"].to_numpy(dtype="datetime64[ns]").astype("int64")
        fix_lats = fixes["lat"].to_numpy(dtype=np.float64) if len(fixes) else np.array([], dtype=np.float64)
        fix_lons = fixes["lon"].to_numpy(dtype=np.float64) if len(fixes) else np.array([], dtype=np.float64)
        if len(fixes):
            fix_usable = np.array(
                [
                    _usable_fix(lat, lon, bool(ps))
                    for lat, lon, ps in zip(fixes["lat"], fixes["lon"], fixes["pos_suspect"])
                ],
                dtype=bool,
            )
        else:
            fix_usable = np.array([], dtype=bool)
        times = grp["ts"].to_numpy(dtype="datetime64[ns]").astype("int64")
        of, la, lo = locate_frames(times, fix_times, fix_lats, fix_lons, fix_usable)
        on_fix_u[grp.index] = of
        lat_u[grp.index] = la
        lon_u[grp.index] = lo

    unique["on_fix"] = on_fix_u
    unique["lat"] = lat_u
    unique["lon"] = lon_u

    land_wgs, land_m, transformer = _load_land(args.land_shapefile)
    # Compute distance on unique finite positions, then map back.
    pos_keys = unique[["lat", "lon"]].drop_duplicates()
    dist_unique = distances_to_land(
        pos_keys["lat"].to_numpy(),
        pos_keys["lon"].to_numpy(),
        land_wgs,
        land_m,
        transformer,
    )
    pos_keys = pos_keys.assign(distance_to_land_km=dist_unique)
    unique = unique.merge(pos_keys, on=["lat", "lon"], how="left")

    out = manifest.merge(unique, on=["imd_storm_id", "ts"], how="left")
    if out["on_fix"].isna().any():
        raise RuntimeError("internal join dropped frames")

    threshold = args.near_land_km
    known = out["distance_to_land_km"].notna()
    near = pd.Series("unknown", index=out.index, dtype="object")
    near.loc[known & (out["distance_to_land_km"] <= threshold)] = "true"
    near.loc[known & (out["distance_to_land_km"] > threshold)] = "false"
    out["near_land"] = near
    out["on_fix"] = out["on_fix"].map(_csv_bool)
    out["lat"] = out["lat"].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
    out["lon"] = out["lon"].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
    out["distance_to_land_km"] = out["distance_to_land_km"].map(
        lambda v: "" if pd.isna(v) else f"{v:.2f}"
    )

    cols = ["filename", "on_fix", "lat", "lon", "distance_to_land_km", "near_land"]
    if not args.all_frames:
        if "is_augmented" in manifest.columns:
            keep = manifest["is_augmented"].astype(str).str.lower().isin(["false", "0", ""])
            out = out.loc[keep.values]
        else:
            aug = out["filename"].str.contains(
                r"_(?:flip_h|flip_v|rot_90|rot_180|rot_270)\.jpg$", regex=True
            )
            out = out.loc[~aug]
    result = out[cols].copy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    n = len(result)
    n_on = int((result["on_fix"] == "true").sum())
    n_off = int((result["on_fix"] == "false").sum())
    vc = result["near_land"].value_counts()
    print(f"wrote {n} rows to {args.output}")
    print(f"on_fix true={n_on} false={n_off}")
    print(
        "near_land true="
        f"{int(vc.get('true', 0))} false={int(vc.get('false', 0))} "
        f"unknown={int(vc.get('unknown', 0))}"
    )
    print(
        "sources: "
        f"{NE_LAND_NAME} v{NE_LAND_VERSION} ({NE_LAND_LICENSE}, {NE_LAND_URL}); "
        f"on_fix tolerance {ON_FIX_TOLERANCE_S}s; near_land <= {threshold:g} km"
    )
    return result


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
