"""Synthetic images for software tests. These are not cyclones."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Project-class winds that stay inside the bin when a few knots are added.
CLASS_WINDS = (
    ("D_DD", 26.0),
    ("CS", 40.0),
    ("SCS", 55.0),
    ("VSCS", 75.0),
    ("ESCS_SUCS", 105.0),
)


def _synthetic_ir(size: int, wind_kt: float, rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    radius = np.hypot(yy - size / 2, xx - size / 2)
    cold_core = 220.0 - (wind_kt - 20.0) * 0.7 * np.exp(-(radius**2) / (2 * 8.0**2))
    noise = rng.normal(0.0, 1.5, size=(size, size))
    return (cold_core + noise).astype(np.float32)


def _write_rows(root: Path, rows: list[dict[str, object]]) -> Path:
    manifest = root / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


def write_smoke(
    root: Path,
    n_storms: int = 8,
    frames_per_storm: int = 4,
    size: int = 64,
    seed: int = 7,
) -> Path:
    """Write a manifest and ``.npy`` images.

    Storm ``S06`` has frames in 2020 and 2021 so the year-holdout test can
    prove that a boundary storm is not cut.
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
    return _write_rows(root, rows)


def write_balanced_smoke(
    root: Path,
    storms_per_class: int = 5,
    frames_per_storm: int = 4,
    size: int = 48,
    seed: int = 7,
) -> Path:
    """One wind bin per storm, several storms per project class.

    Used to exercise grouped classification splits. Still synthetic.
    """
    if storms_per_class < 3:
        raise ValueError("storms_per_class must be at least 3 for grouped folds")
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    image_dir = root / "images"
    image_dir.mkdir(exist_ok=True)
    rows: list[dict[str, object]] = []
    for class_index, (_name, base_wind) in enumerate(CLASS_WINDS):
        for storm_index in range(storms_per_class):
            storm_id = f"C{class_index}{storm_index}"
            basin = "NIO" if (class_index + storm_index) % 2 else "NA"
            year = 2016 + ((class_index + storm_index) % 6)
            for time_index in range(frames_per_storm):
                wind = float(base_wind + time_index)
                relative = Path("images") / f"{storm_id}_{time_index:02d}.npy"
                np.save(root / relative, _synthetic_ir(size, wind, rng))
                rows.append(
                    {
                        "sample_id": f"{storm_id}_{time_index:02d}",
                        "image_path": relative.as_posix(),
                        "storm_id": storm_id,
                        "timestamp": f"{year}-08-01T{time_index:02d}:00:00Z",
                        "year": year,
                        "basin": basin,
                        "sensor": "synthetic",
                        "channel": "IR",
                        "wind_speed_kt": wind,
                        "wind_averaging": "synthetic",
                        "label_source": "smoke",
                        "is_synthetic": True,
                    }
                )
    return _write_rows(root, rows)


def _default_out() -> Path:
    """Directory that holds ``paths.manifest`` in the smoke config."""
    from src.config import load_config

    root = Path(__file__).resolve().parents[1]
    manifest = Path(load_config(root / "config" / "config_smoke.yaml")["paths"]["manifest"])
    return manifest.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=_default_out())
    parser.add_argument(
        "--profile",
        choices=("span", "balanced"),
        default="span",
        help="span: year-boundary storm. balanced: several storms per class.",
    )
    args = parser.parse_args()
    if args.profile == "balanced":
        path = write_balanced_smoke(args.out)
    else:
        path = write_smoke(args.out)
    print(f"Wrote synthetic manifest {path}. Not real cyclones. Do not report scores from it.")


if __name__ == "__main__":
    main()
