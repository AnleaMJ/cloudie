"""Build a Cloudie manifest from the NASA Tropical Storm Wind Speed dataset.

The main path is TorchGeo ``TropicalCyclone`` (Maskey et al. 2020,
https://doi.org/10.1109/JSTARS.2020.3011907). TorchGeo checks the files and
reads ``{training,test}_set_{features,labels}.csv``; images sit at
``<root>/<split>/<Image ID>.jpg``. Download with
``TropicalCyclone(root, split, download=True)``, which needs ``azcopy`` and
pulls from the public Source Cooperative blob. The Hugging Face repo
``torchgeo/tropical_cyclone`` is only a mirror. CI never downloads either.

Images keep their native size. Nothing in this module resizes. Most frames
are 366 x 366, but the real train side also has 1093 x 1093 frames (stored
as grey in an RGB JPEG), which TorchGeo's own ``__getitem__`` bilinearly
resizes to 366. ``native_height`` and ``native_width`` are recorded per row.

Pixel values are an 8-bit rendering of IR brightness, not calibrated Kelvin.
Call them pixel stats until a calibration table exists.

Labels are HURDAT2 best-track maximum sustained winds in knots. HURDAT2 uses
1-minute sustained winds, so ``wind_averaging`` is ``1min``.

The competition's own train/test sides split frames, not storms: many test
storms also appear in train. ``source_split`` keeps that side for reference
only. Do not train on one side and score on the other. Use the storm-wise
splitter in ``src.splits`` on the union instead.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from src.schema import require_valid

WIND_AVERAGING = "1min"
LABEL_SOURCE = "nasa_tropical_storm_competition_hurdat2"
SPLITS = ("train", "test")
_FILE_STEM = {"train": "training_set", "test": "test_set"}


def torchgeo_tables(root: str | Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return TorchGeo's features and labels frames for one competition side.

    Imports TorchGeo lazily so tests and the web app do not need it.
    """
    from torchgeo.datasets import TropicalCyclone

    dataset = TropicalCyclone(root=str(root), split=split, download=False)
    return dataset.features, dataset.labels


def csv_tables(root: str | Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the same CSVs TorchGeo reads, without importing TorchGeo."""
    stem = _FILE_STEM[split]
    root = Path(root)
    return (
        pd.read_csv(root / f"{stem}_features.csv"),
        pd.read_csv(root / f"{stem}_labels.csv"),
    )


def build_manifest(
    root: str | Path,
    manifest_dir: str | Path,
    splits: tuple[str, ...] = SPLITS,
    tables: Callable[[str | Path, str], tuple[pd.DataFrame, pd.DataFrame]] = torchgeo_tables,
    check_images: bool = True,
) -> pd.DataFrame:
    """Return a validated manifest. ``image_path`` is relative to ``manifest_dir``."""
    root = Path(root).resolve()
    manifest_dir = Path(manifest_dir).resolve()
    frames = []
    for split in splits:
        if split not in _FILE_STEM:
            raise ValueError(f"unknown split {split}")
        features, labels = tables(root, split)
        merged = features.merge(labels, on="Image ID", how="inner", validate="1:1")
        if len(merged) != len(labels) or len(merged) != len(features):
            raise ValueError(f"{split}: features and labels do not line up")
        image_dir = root / split
        paths = [image_dir / f"{image_id}.jpg" for image_id in merged["Image ID"]]
        frames.append(
            pd.DataFrame(
                {
                    "sample_id": merged["Image ID"].astype(str),
                    "image_path": [_relative(path, manifest_dir) for path in paths],
                    "storm_id": merged["Storm ID"].astype(str),
                    "wind_speed_kt": pd.to_numeric(merged["Wind Speed"]),
                    "relative_time_s": pd.to_numeric(merged["Relative Time"]),
                    "ocean": merged["Ocean"].astype(int),
                    "basin": merged["Ocean"].map({1: "ATL", 2: "EPAC"}).fillna("unknown"),
                    "sensor": "GOES",
                    "channel": "IR",
                    "pixel_units": "8bit_uncalibrated",
                    "wind_averaging": WIND_AVERAGING,
                    "label_source": LABEL_SOURCE,
                    "center_source": "dataset_precentered",
                    "source_split": split,
                    "is_synthetic": False,
                }
            )
        )
    frame = pd.concat(frames, ignore_index=True)
    if check_images:
        frame = add_native_size(frame, manifest_dir)
    return require_valid(frame)


def _relative(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def add_native_size(frame: pd.DataFrame, manifest_dir: str | Path) -> pd.DataFrame:
    """Record native size and JPEG mode. Fails if any image is missing."""
    from PIL import Image

    manifest_dir = Path(manifest_dir)
    heights, widths, modes, grey, missing = [], [], [], [], []
    for image_path in frame["image_path"]:
        path = Path(image_path)
        path = path if path.is_absolute() else manifest_dir / path
        if not path.is_file():
            missing.append(str(path))
            heights.append(-1)
            widths.append(-1)
            modes.append("")
            grey.append(False)
            continue
        with Image.open(path) as image:
            widths.append(image.width)
            heights.append(image.height)
            modes.append(image.mode)
            grey.append(image.mode == "L" or _channels_identical(np.asarray(image)))
    if missing:
        raise FileNotFoundError(f"{len(missing)} images missing, first: {missing[:3]}")
    out = frame.copy()
    out["native_height"] = heights
    out["native_width"] = widths
    out["source_mode"] = modes
    # False for colour-enhanced frames (npf_000/001 in this release). load_ir
    # refuses them, so filter on this column before training.
    out["grey_ir"] = grey
    return out


def _channels_identical(array: np.ndarray) -> bool:
    if array.ndim != 3:
        return True
    return bool(np.array_equal(array[..., 0], array[..., 1]) and np.array_equal(array[..., 0], array[..., 2]))


def load_ir(path: str | Path) -> np.ndarray:
    """Read one frame as float32 at native size, single channel.

    Three-channel JPEGs are accepted only when all channels match, so a
    colour enhancement is never mistaken for IR.
    """
    from PIL import Image

    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim == 3:
        if not _channels_identical(array):
            raise ValueError(f"{path}: channels differ, not a grey IR render")
        array = array[..., 0]
    return array.astype(np.float32)


def summarize(frame: pd.DataFrame) -> dict[str, object]:
    by_side = {
        side: {
            "rows": int((frame["source_split"] == side).sum()),
            "storms": int(frame.loc[frame["source_split"] == side, "storm_id"].nunique()),
        }
        for side in sorted(frame["source_split"].unique())
    }
    sides = [set(frame.loc[frame["source_split"] == side, "storm_id"]) for side in by_side]
    shared = len(set.intersection(*sides)) if len(sides) > 1 else 0
    return {
        "rows": int(len(frame)),
        "unique_storms": int(frame["storm_id"].nunique()),
        "by_source_split": by_side,
        "storms_on_both_source_sides": shared,
        "wind_kt_min": float(frame["wind_speed_kt"].min()),
        "wind_kt_max": float(frame["wind_speed_kt"].max()),
        "wind_kt_mean": round(float(frame["wind_speed_kt"].mean()), 2),
        "native_sizes": sorted(
            {f"{h}x{w}" for h, w in zip(frame["native_height"], frame["native_width"])}
        )
        if "native_height" in frame
        else [],
        "source_modes": frame["source_mode"].value_counts().to_dict()
        if "source_mode" in frame
        else {},
        "frames_not_grey_ir": sorted(frame.loc[~frame["grey_ir"], "sample_id"]) if "grey_ir" in frame else [],
        "native_size_counts": {
            f"{h}x{w}": int(n)
            for (h, w), n in frame.groupby(["native_height", "native_width"]).size().items()
        }
        if "native_height" in frame
        else {},
        "wind_averaging": sorted(frame["wind_averaging"].unique()),
        "is_synthetic": bool(frame["is_synthetic"].any()),
    }


def save_sample_png(frame: pd.DataFrame, manifest_dir: str | Path, out: str | Path, index: int = 0) -> dict[str, object]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row = frame.iloc[index]
    path = Path(row["image_path"])
    path = path if path.is_absolute() else Path(manifest_dir) / path
    image = load_ir(path)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image, cmap="gray")
    ax.set_title(f"{row['sample_id']} | {row['wind_speed_kt']:.0f} kt (1min) | NASA TC, real")
    ax.axis("off")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return {"sample_id": row["sample_id"], "wind_speed_kt": float(row["wind_speed_kt"]), "shape": list(image.shape), "png": str(out)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, help="TorchGeo TropicalCyclone root")
    parser.add_argument("--out", default="data/nasa_tc/manifest.csv")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS))
    parser.add_argument("--reader", choices=["torchgeo", "csv"], default="torchgeo")
    parser.add_argument("--summary", default="reports/nasa_tc_summary.json")
    parser.add_argument("--sample-png", default="reports/nasa_tc_sample.png")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tables = torchgeo_tables if args.reader == "torchgeo" else csv_tables
    frame = build_manifest(args.root, out.parent, tuple(args.splits), tables)
    frame.drop(columns=["imd5_class"]).to_csv(out, index=False)
    summary = summarize(frame)
    summary["reader"] = args.reader
    summary["manifest"] = str(out)
    summary["sample"] = save_sample_png(frame, out.parent, args.sample_png)
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
