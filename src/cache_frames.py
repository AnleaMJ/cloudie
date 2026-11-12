"""Write fp16 ``.npy`` frames and an index CSV for one dataset.

Training reads this cache when ``paths.frame_cache/<dataset>/index.csv``
exists and otherwise reads the source files. Nothing is downloaded.
Workers then load a small array instead of decoding the source.

``python -m src.cache_frames --dataset nasa``
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

_SAMPLE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_name(sample_id: str) -> str:
    cleaned = _SAMPLE_CHARS.sub("_", sample_id.strip())
    return cleaned or "frame"


def cache_dataset(dataset: str, config: dict) -> Path:
    """Read one local dataset and write fp16 frames plus ``index.csv``."""
    from src.data_loader import load_image
    from src.train import load_training_frame

    name = str(dataset).strip().lower()
    root = Path(config["paths"]["frame_cache"])
    frame, _excluded = load_training_frame(name, config)
    dest_dir = root / name / "frames"
    dest_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for row in frame.itertuples(index=False):
        array = np.asarray(load_image(row.image_path), dtype=np.float16)
        path = dest_dir / f"{_safe_name(str(row.sample_id))}.npy"
        np.save(path, array)
        rows.append(
            {
                "sample_id": str(row.sample_id),
                "cache_path": path.resolve().as_posix(),
                "source_path": str(row.image_path),
                "dataset": name,
                "dtype": "float16",
            }
        )
    index = root / name / "index.csv"
    pd.DataFrame(rows).to_csv(index, index=False)
    print(f"Wrote {index} ({len(rows)} fp16 frames). Not a model result.")
    return index


def main(argv: list[str] | None = None) -> Path:
    """``python -m src.cache_frames --config config/config.yaml --dataset nasa``."""
    from src.config import load_config
    from src.train import _DATASETS

    parser = argparse.ArgumentParser(
        description="Cache one dataset as fp16 npy frames. Does not download imagery."
    )
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--dataset", required=True, choices=_DATASETS)
    args = parser.parse_args(argv)
    return cache_dataset(args.dataset, load_config(args.config))


if __name__ == "__main__":
    main()
