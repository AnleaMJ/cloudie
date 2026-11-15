"""Cache INCYDE frames at a fixed working size for CPU training.

Kept separate from the INCYDE loader in ``src`` (Clara's PR) so the two do
not collide. Inputs:

- the audit manifest (``manifest.csv`` built by ``incyde/audit/build_manifest.py``),
- ``incyde/audit/storm_split.csv`` (storm-ID split and ``cooccurrence_group``),
- ``incyde/audit/exclusions.csv`` (only ``default_exclude=true`` rows are dropped).

Only original frames and INCYDE's own ``rot_180`` copies are cached.
``flip_h``, ``flip_v``, ``rot_90`` and ``rot_270`` are not used (no flips;
the 90/270 copies keep the original canvas and lose corners). Each image goes
through ``src.preprocessing.prepare_image`` (median fill, bilinear resize),
the same call used at inference. Labels are IMD 3-minute winds, unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import load_image
from src.preprocessing import prepare_image

REPO = Path(__file__).resolve().parents[2]


def _load(args: tuple[str, int]) -> np.ndarray:
    path, size = args
    return prepare_image(load_image(path), size, None, 1.0)[0].astype(np.float16)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_index(manifest: Path) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(manifest, low_memory=False)
    split = pd.read_csv(REPO / "incyde/audit/storm_split.csv")
    excl = pd.read_csv(REPO / "incyde/audit/exclusions.csv")
    excl["default_exclude"] = excl["default_exclude"].astype(str).str.lower() == "true"
    drop_names = set(excl.loc[excl["default_exclude"], "filename"])
    frame["filename"] = frame["file"].str.split("/").str[-1]
    n_all = len(frame)
    dropped = frame["filename"].isin(drop_names)
    frame = frame.loc[~dropped].copy()
    keep = frame["aug_type"].isna() | (frame["aug_type"] == "rot_180")
    frame = frame.loc[keep].copy()
    frame = frame.drop(columns=["proposed_split"]).merge(
        split[["Sequence", "proposed_split", "cooccurrence_group", "peak_imd_class"]],
        left_on="incyde_sequence",
        right_on="Sequence",
        how="left",
        validate="many_to_one",
    )
    if frame["proposed_split"].isna().any():
        raise ValueError("a frame has no storm_split.csv row")
    flags = excl.loc[~excl["default_exclude"]].set_index("filename")["qc_flag"]
    frame["qc_flag_kept"] = frame["filename"].map(flags)
    frame["is_original"] = frame["aug_type"].isna()
    frame["source_filename"] = frame["source_frame"].str.split("/").str[-1]
    frame["wind_kt_3min"] = frame["label_kt"].astype(float)
    frame["wind_kt_raw"] = frame["label_kt"].astype(float)
    frame["wind_averaging_s"] = 180
    frame["wind_converted"] = False
    stats = {
        "manifest_rows": int(n_all),
        "dropped_default_exclude": int(dropped.sum()),
        "kept_originals": int(frame["is_original"].sum()),
        "kept_rot_180": int((~frame["is_original"]).sum()),
        "flag_only_rows_kept": int(frame["qc_flag_kept"].notna().sum()),
    }
    return frame.reset_index(drop=True), stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    frame, stats = build_index(args.manifest)
    paths = [str(args.image_root / rel) for rel in frame["file"]]
    with ProcessPoolExecutor(args.workers) as pool:
        images = list(pool.map(_load, [(p, args.size) for p in paths], chunksize=256))
    array = np.stack(images)
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / f"incyde_{args.size}.npy", array)
    frame.to_csv(args.out / "incyde_index.csv", index=False)
    stats.update(
        {
            "size": args.size,
            "manifest_sha256": sha256(args.manifest),
            "storm_split_sha256": sha256(REPO / "incyde/audit/storm_split.csv"),
            "exclusions_sha256": sha256(REPO / "incyde/audit/exclusions.csv"),
            "zenodo_zip_md5": "ed5dac254c689e5a327dee7580324347",
            "cache_sha256": sha256(args.out / f"incyde_{args.size}.npy"),
        }
    )
    (args.out / "incyde_prep.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
