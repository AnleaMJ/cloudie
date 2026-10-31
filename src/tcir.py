"""Build a Cloudie manifest from TCIR HDF5 files, IR1 channel only.

TCIR (Chen, Chen and Lin, KDD 2018, https://doi.org/10.1145/3219819.3219926;
data https://github.com/BoyoChen/TCIR) ships two HDF5 files:
``TCIR-ATLN_EPAC_WPAC.h5`` and ``TCIR-CPAC_IO_SH.h5``. Each has

- ``matrix``: N x 201 x 201 x 4 float32. The README lists the channels in
  the order IR1, WV, VIS, PMW, so IR1 is index 0 (``IR1_INDEX``). The file
  has no channel attributes. On 512 random real frames per file, channel 0
  has a median near 275 and is warmer than channel 1 (WV, median near 238)
  in over 98% of frames, channel 2 is a 0 to 2.2 reflectance with many NaNs
  (VIS at night) and channel 3 is a rain rate. That fits the README order.
  IR1 comes from GridSat-B1, which stores brightness temperature in kelvin.
  Reports still call these "pixel stats".
- ``info``: a pandas 0.15 "fixed" frame (data_set, ID, time, lon, lat, Vmax,
  R35_4qAVG, MSLP). Current pandas cannot read it, so ``read_info`` decodes
  it with h5py and a restricted unpickler that only rebuilds numpy arrays.

Only channel 0 is read (an h5py hyperslab, ``matrix[a:b, :, :, 0]``). Frames
keep their native 201 x 201 grid, centred on the best-track position, about
7/100 degree per pixel.

Labels. TCIR documents JTWC best tracks for WPAC and HURDAT2 for ATLN and
EPAC. Both are 1-minute sustained winds, so ``wind_averaging`` is ``1min``.
TCIR does not document the label source for CPAC, IO or SH. Those rows carry
``label_source_documented = False`` and an ``_inferred`` label source. A
cross-check against IBTrACS v04r01 (matched on time and a 0.6 degree
position window) found TCIR ``Vmax`` equal to ``USA_WIND`` on 99.8% of IO
and SH rows (JTWC, 1-minute) and on 92% of CPAC rows (99.5% within 5 kt;
HURDAT2/CPHC, 1-minute), while the 10-minute (BoM, RSMC La Reunion, Nadi)
and 3-minute (New Delhi) agency winds match 14-21% of the time. So
``wind_averaging`` is ``1min`` on every row.

Southern Hemisphere storms rotate clockwise. Nothing here flips an image by
default. ``hemisphere`` (N or S, from the row's latitude) is recorded per
row. ``flip_southern=True`` is an explicit opt-in that mirrors S frames
north-south when exporting and sets ``flipped = True`` on those rows.

Frames whose IR1 is all NaN, has a NaN fraction above
``max_missing_fraction`` (0.30, the preprocessing gate), or has any finite
value outside ``IR1_VALID_RANGE`` (150 to 340 K; the real files hold a few
frames with pixels down to 76 K or up to 348 K) are dropped and listed with a reason in an
exclusions CSV. Kept frames may still hold a few NaNs; ``ir1_nan_fraction``
records how many, and ``src.preprocessing`` fills them later.

TCIR has no official split inside these files, so the manifest has no
``source_split`` and never has ``split``. ``src.splits.assign_storms`` writes
``split`` by storm. MSLP and R35 are label-derived, so they stay out of the
manifest.
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.schema import require_valid

CHANNELS = ("IR1", "WV", "VIS", "PMW")  # order in the TCIR README
IR1_INDEX = 0
IMAGE_SIZE = 201
WIND_AVERAGING = "1min"
IR1_VALID_RANGE = (150.0, 340.0)  # kelvin; colder or warmer than any plausible IR window value
MAX_MISSING_FRACTION = 0.30  # same gate as src.preprocessing
BLOCK = 256

# data_set -> (label source, documented by TCIR)
LABEL_SOURCES = {
    "WPAC": ("jtwc_best_track", True),
    "ATLN": ("hurdat2", True),
    "EPAC": ("hurdat2", True),
    # Not documented by TCIR; inferred from an IBTrACS v04r01 match (PR notes).
    "CPAC": ("hurdat2_inferred", False),
    "IO": ("jtwc_best_track_inferred", False),
    "SH": ("jtwc_best_track_inferred", False),
}
EXCLUSION_COLUMNS = [
    "sample_id",
    "storm_id",
    "basin",
    "source_file",
    "h5_index",
    "ir1_nan_fraction",
    "reason",
]

_ALLOWED_GLOBALS = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("_codecs", "encode"),  # Python 3 protocol-2 pickles of byte strings
}


class _NumpyOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):  # noqa: ANN201
        if (module, name) not in _ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(f"refusing to load {module}.{name} from TCIR info")
        target = "numpy._core.multiarray" if module.endswith(".multiarray") else module
        import importlib

        return getattr(importlib.import_module(target), name)


def _text(value: object) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _block_values(dataset) -> np.ndarray:  # noqa: ANN001
    if dataset.dtype.kind in "fiub":
        return dataset[()]
    raw = dataset[0]
    data = bytes(np.asarray(raw, dtype=np.uint8)) if not isinstance(raw, bytes) else raw
    return _NumpyOnlyUnpickler(io.BytesIO(data), encoding="latin1").load()


def read_info(path: str | Path) -> pd.DataFrame:
    """Decode the ``info`` frame with h5py. Adds ``h5_index`` (row in ``matrix``)."""
    import h5py

    with h5py.File(path, "r") as handle:
        group = handle["info"]
        order = [_text(name) for name in group["axis0"][()]]
        n_rows = int(group["axis1"].shape[0])
        columns: dict[str, np.ndarray] = {}
        for block in range(int(group.attrs["nblocks"])):
            items = [_text(name) for name in group[f"block{block}_items"][()]]
            values = np.asarray(_block_values(group[f"block{block}_values"]))
            if values.shape != (n_rows, len(items)):
                raise ValueError(f"{path}: info block{block} has shape {values.shape}")
            for position, item in enumerate(items):
                columns[item] = values[:, position]
        n_matrix = int(handle["matrix"].shape[0])
    if n_matrix != n_rows:
        raise ValueError(f"{path}: info has {n_rows} rows but matrix has {n_matrix}")
    frame = pd.DataFrame({name: columns[name] for name in order})
    for name in frame.columns:
        if frame[name].dtype == object:
            frame[name] = frame[name].map(_text)
    frame["h5_index"] = np.arange(n_rows)
    return frame


def iter_ir1(path: str | Path, block: int = BLOCK) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(start, frames)`` with frames shaped (k, 201, 201), IR1 only."""
    import h5py

    with h5py.File(path, "r") as handle:
        matrix = handle["matrix"]
        if matrix.ndim != 4 or matrix.shape[1:] != (IMAGE_SIZE, IMAGE_SIZE, len(CHANNELS)):
            raise ValueError(f"{path}: matrix shape {matrix.shape}, expected N x 201 x 201 x 4")
        for start in range(0, matrix.shape[0], block):
            stop = min(start + block, matrix.shape[0])
            yield start, np.asarray(matrix[start:stop, :, :, IR1_INDEX], dtype=np.float32)


def exclusion_reason(
    frame: np.ndarray,
    max_missing_fraction: float = MAX_MISSING_FRACTION,
    valid_range: tuple[float, float] = IR1_VALID_RANGE,
) -> tuple[str | None, float]:
    """Return (reason or None, NaN fraction) for one IR1 frame."""
    finite = np.isfinite(frame)
    nan_fraction = float(1.0 - finite.mean())
    if not finite.any():
        return "ir1 all NaN", nan_fraction
    if nan_fraction > max_missing_fraction:
        return f"ir1 NaN fraction {nan_fraction:.3f} > {max_missing_fraction:.2f}", nan_fraction
    values = frame[finite]
    low, high = valid_range
    if float(values.min()) < low or float(values.max()) > high:
        return (
            f"ir1 value outside [{low:g}, {high:g}] K "
            f"(min {float(values.min()):.2f}, max {float(values.max()):.2f})",
            nan_fraction,
        )
    return None, nan_fraction


def hemisphere_of(lat: pd.Series) -> pd.Series:
    return pd.Series(np.where(pd.to_numeric(lat) < 0, "S", "N"), index=lat.index)


def orient(
    frame: np.ndarray, hemisphere: str, flip_southern: bool = False
) -> tuple[np.ndarray, bool]:
    """Return (frame, flipped). Only flips when ``flip_southern`` is set and the row is S."""
    if flip_southern and hemisphere == "S":
        return np.ascontiguousarray(frame[::-1, :]), True
    return frame, False


def _rows(info: pd.DataFrame, source_file: str) -> pd.DataFrame:
    unknown = sorted(set(info["data_set"]) - set(LABEL_SOURCES))
    if unknown:
        raise ValueError(f"unknown TCIR data_set values {unknown}")
    stamp = pd.to_datetime(info["time"].astype(str), format="%Y%m%d%H", utc=True)
    lon = pd.to_numeric(info["lon"]).astype(float)
    rows = pd.DataFrame(
        {
            "sample_id": "tcir_" + info["ID"].astype(str) + "_" + info["time"].astype(str),
            "storm_id": info["ID"].astype(str),
            "timestamp": stamp.dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "year": stamp.dt.year.astype(int),
            "wind_speed_kt": pd.to_numeric(info["Vmax"]).astype(float),
            "basin": info["data_set"].astype(str),
            "lat": pd.to_numeric(info["lat"]).astype(float),
            "lon": ((lon + 180.0) % 360.0) - 180.0,
            "hemisphere": hemisphere_of(info["lat"]),
            "source": "tcir",
            "source_file": source_file,
            "h5_index": info["h5_index"].astype(int),
            "sensor": "GridSat-B1",
            "channel": "IR1",
            "channel_index": IR1_INDEX,
            "native_height": IMAGE_SIZE,
            "native_width": IMAGE_SIZE,
            "orig_size": IMAGE_SIZE,
            "pixel_units": "kelvin_brightness_temperature",
            "wind_averaging": WIND_AVERAGING,
            "label_source": info["data_set"].map(lambda b: LABEL_SOURCES[b][0]),
            "label_source_documented": info["data_set"].map(lambda b: LABEL_SOURCES[b][1]),
            "center_source": "dataset_precentered",
            "is_synthetic": False,
        }
    )
    return rows


class _PixelStats:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.total_sq = 0.0
        self.low = np.inf
        self.high = -np.inf

    def add(self, frame: np.ndarray) -> None:
        values = frame[np.isfinite(frame)].astype(np.float64)
        if values.size:
            self.count += int(values.size)
            self.total += float(values.sum())
            self.total_sq += float(np.square(values).sum())
            self.low = min(self.low, float(values.min()))
            self.high = max(self.high, float(values.max()))

    def as_dict(self) -> dict[str, float | int]:
        if not self.count:
            return {"finite_pixels": 0}
        mean = self.total / self.count
        return {
            "finite_pixels": self.count,
            "min": round(self.low, 3),
            "max": round(self.high, 3),
            "mean": round(mean, 3),
            "std": round(float(np.sqrt(max(self.total_sq / self.count - mean**2, 0.0))), 3),
        }


def build_manifest(
    h5_paths: Sequence[str | Path],
    manifest_dir: str | Path,
    export: bool = True,
    flip_southern: bool = False,
    max_missing_fraction: float = MAX_MISSING_FRACTION,
    block: int = BLOCK,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Return (manifest, exclusions, pixel stats).

    With ``export`` each kept IR1 frame is saved as ``ir1/<sample_id>.npy``
    under ``manifest_dir`` and ``image_path`` points there (relative), so
    ``src.data_loader`` can read it. Without ``export`` ``image_path`` is
    ``<file>#<h5_index>`` for reference only.
    """
    manifest_dir = Path(manifest_dir).resolve()
    image_dir = manifest_dir / "ir1"
    if export:
        image_dir.mkdir(parents=True, exist_ok=True)
    kept_parts, excluded_parts = [], []
    stats = _PixelStats()
    frames_with_nan = 0
    for h5_path in h5_paths:
        h5_path = Path(h5_path)
        rows = _rows(read_info(h5_path), h5_path.name)
        reasons: list[str | None] = [None] * len(rows)
        nan_fractions = np.zeros(len(rows))
        flipped = np.zeros(len(rows), dtype=bool)
        winds = rows["wind_speed_kt"].to_numpy()
        bad_wind = ~np.isfinite(winds) | (winds < 0)
        for start, frames in iter_ir1(h5_path, block):
            for offset, frame in enumerate(frames):
                index = start + offset
                reason, nan_fractions[index] = exclusion_reason(frame, max_missing_fraction)
                if reason is None and bad_wind[index]:
                    reason = "wind missing or negative"
                reasons[index] = reason
                if reason is not None:
                    continue
                stats.add(frame)
                frames_with_nan += int(nan_fractions[index] > 0)
                if export:
                    out, flipped[index] = orient(frame, rows.at[index, "hemisphere"], flip_southern)
                    np.save(image_dir / f"{rows.at[index, 'sample_id']}.npy", out.astype(np.float32))
        rows["ir1_nan_fraction"] = np.round(nan_fractions, 5)
        rows["flipped"] = flipped
        rows["image_path"] = (
            "ir1/" + rows["sample_id"] + ".npy"
            if export
            else rows["source_file"] + "#" + rows["h5_index"].astype(str)
        )
        drop = pd.Series([reason is not None for reason in reasons], index=rows.index)
        excluded = rows.loc[drop].assign(reason=[r for r in reasons if r is not None])
        kept_parts.append(rows.loc[~drop])
        excluded_parts.append(excluded[EXCLUSION_COLUMNS])
    manifest = pd.concat(kept_parts, ignore_index=True)
    excluded = pd.concat(excluded_parts, ignore_index=True)
    if "split" in manifest.columns or "source_split" in manifest.columns:
        raise AssertionError("TCIR manifest must not carry split or source_split")
    manifest = require_valid(manifest)
    pixel = stats.as_dict()
    pixel["kept_frames_with_some_nan"] = frames_with_nan
    return manifest, excluded, pixel


def storm_split_check(
    frame: pd.DataFrame, strategy: str = "storm_wise", seed: int = 42
) -> dict[str, object]:
    """Run the project splitter on the manifest and assert no storm leaks."""
    from src.splits import assert_no_group_leakage, assign_storms

    if "split" in frame.columns:
        raise ValueError("manifest must not carry split")
    assigned = assign_storms(frame, strategy, seed=seed)
    names = ("train", "val", "test")
    parts = [assigned.loc[assigned["split"] == name] for name in names]
    assert_no_group_leakage(*parts)
    return {
        "strategy": strategy,
        "rows": {name: int(len(part)) for name, part in zip(names, parts)},
        "storms": {name: int(part["storm_id"].nunique()) for name, part in zip(names, parts)},
        "leakage": "none",
    }


def _reason_kind(reason: str) -> str:
    """Strip per-frame numbers so reasons can be counted."""
    if reason.startswith("ir1 NaN fraction"):
        return f"ir1 NaN fraction > {MAX_MISSING_FRACTION:.2f}"
    return reason.split(" (")[0]


def summarize(
    frame: pd.DataFrame, excluded: pd.DataFrame, pixel: dict[str, object]
) -> dict[str, object]:
    def counts(column: str) -> dict[str, dict[str, int]]:
        return {
            str(key): {"rows": int(len(group)), "storms": int(group["storm_id"].nunique())}
            for key, group in frame.groupby(column)
        }

    storm_hemispheres = frame.groupby("storm_id")["hemisphere"].nunique()
    return {
        "rows": int(len(frame)),
        "unique_storms": int(frame["storm_id"].nunique()),
        "by_basin": counts("basin"),
        "by_hemisphere": counts("hemisphere"),
        "by_source_file": counts("source_file"),
        "storms_crossing_equator": sorted(storm_hemispheres[storm_hemispheres > 1].index.tolist()),
        "basin_vs_hemisphere": {
            f"{basin}/{hemi}": int(n)
            for (basin, hemi), n in frame.groupby(["basin", "hemisphere"]).size().items()
        },
        "years": [int(frame["year"].min()), int(frame["year"].max())],
        "wind_kt_min": float(frame["wind_speed_kt"].min()),
        "wind_kt_max": float(frame["wind_speed_kt"].max()),
        "wind_kt_mean": round(float(frame["wind_speed_kt"].mean()), 2),
        "pixel_stats_ir1": pixel,
        "flipped_rows": int(frame["flipped"].sum()),
        "label_source_counts": frame["label_source"].value_counts().to_dict(),
        "label_source_undocumented_rows": int((~frame["label_source_documented"].astype(bool)).sum()),
        "wind_averaging": sorted(frame["wind_averaging"].unique()),
        "native_sizes": sorted({f"{h}x{w}" for h, w in zip(frame["native_height"], frame["native_width"])}),
        "has_split_column": "split" in frame.columns,
        "has_source_split_column": "source_split" in frame.columns,
        "is_synthetic": bool(frame["is_synthetic"].any()),
        "excluded": {
            "count": int(len(excluded)),
            "by_reason": excluded["reason"].map(_reason_kind).value_counts().to_dict(),
            "by_basin": excluded["basin"].value_counts().to_dict() if len(excluded) else {},
        },
    }


def save_sample_png(
    frame: pd.DataFrame, manifest_dir: str | Path, out: str | Path
) -> list[dict[str, object]]:
    """One N and one S frame side by side, as stored (no flip unless opted in)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    picks = []
    for hemi in ("N", "S"):
        subset = frame[frame["hemisphere"] == hemi]
        if len(subset):
            picks.append(subset.loc[subset["wind_speed_kt"].idxmax()])
    fig, axes = plt.subplots(1, len(picks), figsize=(4.2 * len(picks), 4.4), squeeze=False)
    info = []
    for ax, row in zip(axes[0], picks):
        image = np.load(Path(manifest_dir) / row["image_path"])
        shown = ax.imshow(image, cmap="gray_r")
        fig.colorbar(shown, ax=ax, fraction=0.046, label="IR1 pixel value")
        ax.set_title(
            f"{row['storm_id']} {row['timestamp'][:13]}Z {row['basin']}/{row['hemisphere']}\n"
            f"{row['wind_speed_kt']:.0f} kt (1min) | flipped={bool(row['flipped'])} | TCIR real",
            fontsize=8,
        )
        ax.axis("off")
        info.append(
            {
                "sample_id": row["sample_id"],
                "hemisphere": row["hemisphere"],
                "wind_speed_kt": float(row["wind_speed_kt"]),
                "shape": list(image.shape),
                "flipped": bool(row["flipped"]),
            }
        )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--h5", nargs="+", required=True, help="TCIR .h5 files")
    parser.add_argument("--out", default="data/tcir/manifest.csv")
    parser.add_argument("--summary", default="reports/tcir_summary.json")
    parser.add_argument("--sample-png", default="reports/tcir_sample.png")
    parser.add_argument("--exclusions", default=None, help="default: <manifest dir>/tcir_exclusions.csv")
    parser.add_argument("--no-export", action="store_true", help="do not write ir1/*.npy")
    parser.add_argument(
        "--flip-southern",
        action="store_true",
        help="opt in: mirror Southern Hemisphere frames north-south on export (default: no flip)",
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame, excluded, pixel = build_manifest(
        args.h5, out.parent, export=not args.no_export, flip_southern=args.flip_southern
    )
    exclusions_path = Path(args.exclusions) if args.exclusions else out.parent / "tcir_exclusions.csv"
    excluded.to_csv(exclusions_path, index=False)
    frame.drop(columns=["imd5_class"], errors="ignore").to_csv(out, index=False)
    summary = summarize(frame, excluded, pixel)
    summary["exclusions_file"] = str(exclusions_path)
    summary["storm_split_check"] = storm_split_check(frame)
    summary["flip_southern"] = bool(args.flip_southern)
    summary["manifest"] = str(out)
    if not args.no_export:
        summary["sample"] = save_sample_png(frame, out.parent, args.sample_png)
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
