"""Cache TCIR IR1 frames at the INCYDE working size for CPU pretraining.

Reuses ``src.tcir`` (``read_info``, ``iter_ir1``, ``exclusion_reason``) and
``src.preprocessing.prepare_image``. Nothing is downloaded.

Labels: TCIR ``Vmax`` is a 1-minute sustained wind (see ``src/tcir.py``).
Every row keeps ``wind_kt_raw`` and ``wind_averaging_s = 60``. The 3-minute
target is ``wind_kt_3min = factor * wind_kt_raw`` with the single factor
from ``config/incyde_cpu.yaml`` (``labels.one_to_three_min_factor``), which
is a derived at-sea value, not a WMO constant. Rows whose centre is within
``labels.near_land_km`` of land (approximate, ``global-land-mask`` 1 km grid
sampled on rings) get ``near_land = True`` and are flagged, not silently
converted: the pretraining config decides whether to keep them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.preprocessing import prepare_image
from src.tcir import _rows, exclusion_reason, iter_ir1, read_info

from scripts.incyde_cpu.prep_incyde import sha256

REPO = Path(__file__).resolve().parents[2]


def near_land(lat: np.ndarray, lon: np.ndarray, radius_km: float) -> np.ndarray:
    from global_land_mask import globe

    out = np.asarray(globe.is_land(np.clip(lat, -89.9, 89.9), lon), dtype=bool)
    for ring in (radius_km / 2.0, radius_km):
        for bearing in np.deg2rad(np.arange(0, 360, 30)):
            dlat = ring / 111.0 * np.cos(bearing)
            dlon = ring / (111.0 * np.maximum(np.cos(np.deg2rad(lat)), 0.05)) * np.sin(bearing)
            plat = np.clip(lat + dlat, -89.9, 89.9)
            plon = ((lon + dlon + 180.0) % 360.0) - 180.0
            out |= np.asarray(globe.is_land(plat, plon), dtype=bool)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=REPO / "config/incyde_cpu.yaml")
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    factor = float(cfg["labels"]["one_to_three_min_factor"])
    radius = float(cfg["labels"]["near_land_km"])
    frames, images, excluded = [], [], 0
    for path in args.files:
        info = read_info(path)
        rows = _rows(info, path.name)
        keep = np.zeros(len(rows), dtype=bool)
        for start, block in iter_ir1(path):
            for offset, frame in enumerate(block):
                reason, _nan = exclusion_reason(frame)
                if reason is None:
                    keep[start + offset] = True
                    images.append(prepare_image(frame, args.size, None, 1.0)[0].astype(np.float16))
                else:
                    excluded += 1
            print(path.name, start, flush=True)
        frames.append(rows.loc[keep])
    frame = pd.concat(frames, ignore_index=True)
    frame["wind_kt_raw"] = frame["wind_speed_kt"].astype(float)
    frame["wind_averaging_s"] = 60
    frame["wind_kt_3min"] = frame["wind_kt_raw"] * factor
    frame["wind_kt_3min_derived"] = True
    frame["wind_conversion_factor"] = factor
    frame["near_land"] = near_land(frame["lat"].to_numpy(), frame["lon"].to_numpy(), radius)
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / f"tcir_{args.size}.npy", np.stack(images))
    frame.to_csv(args.out / "tcir_index.csv", index=False)
    stats = {
        "kept": int(len(frame)),
        "excluded_ir1_gate": int(excluded),
        "storms": int(frame["storm_id"].nunique()),
        "near_land_rows": int(frame["near_land"].sum()),
        "factor": factor,
        "size": args.size,
        "cache_sha256": sha256(args.out / f"tcir_{args.size}.npy"),
    }
    (args.out / "tcir_prep.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
