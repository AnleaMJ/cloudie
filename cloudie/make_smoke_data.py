"""Synthetic images for software tests. These are not cyclones."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _synthetic_ir(size: int, wind_kt: float, rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    radius = np.hypot(yy - size / 2, xx - size / 2)
    cold_core = 220.0 - (wind_kt - 20.0) * 0.7 * np.exp(-(radius**2) / (2 * 8.0**2))
    noise = rng.normal(0.0, 1.5, size=(size, size))
    return (cold_core + noise).astype(np.float32)


def write_smoke(
    root: Path,
    n_storms: int = 8,
    frames_per_storm: int = 4,
    size: int = 64,
    seed: int = 7,
) -> Path:
    """Write a manifest and ``.npy`` images.

    Storm ``S06`` has frames in 2020 and 2021 so a year holdout cannot cut it.
    """
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    image_dir = root / "images"
    image_dir.mkdir(exist_ok=True)
    rows: list[dict[str, object]] = []
    for index in range(n_storms):
        storm_id = f"S{index:02d}"
        basin = "NIO" if index % 2 else "NA"
        year = 2018 + (index % 4)
        for time_index in range(frames_per_storm):
            year_t = 2021 if index == 6 and time_index >= 2 else year
            wind = float(np.clip(25 + index * 12 + time_index * 2, 20, 140))
            relative = Path("images") / f"{storm_id}_{time_index:02d}.npy"
            np.save(root / relative, _synthetic_ir(size, wind, rng))
            rows.append(
                {
                    "sample_id": f"{storm_id}_{time_index:02d}",
                    "image_path": relative.as_posix(),
                    "storm_id": storm_id,
                    "timestamp": f"{year_t}-06-01T{time_index:02d}:00:00Z",
                    "year": year_t,
                    "basin": basin,
                    "sensor": "synthetic",
                    "channel": "IR",
                    "wind_speed_kt": wind,
                    "wind_averaging": "synthetic",
                    "label_source": "smoke",
                    "is_synthetic": True,
                }
            )
    manifest = root / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/smoke"))
    args = parser.parse_args()
    path = write_smoke(args.out)
    print(f"Wrote synthetic manifest {path}. Not real cyclones. Do not report scores from it.")


if __name__ == "__main__":
    main()
