"""NASA TC manifest builder on tiny fake tables. Never downloads data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.nasa_tc import build_manifest, csv_tables, load_ir, summarize


def _fake_root(root: Path, size: int = 366) -> Path:
    rows = {"train": [("aaa_000", "aaa", 0, 1, 30), ("aaa_001", "aaa", 1800, 1, 35), ("bbb_000", "bbb", 0, 2, 90)],
            "test": [("aaa_002", "aaa", 3600, 1, 40), ("ccc_000", "ccc", 0, 2, 120)]}
    for split, items in rows.items():
        stem = "training_set" if split == "train" else "test_set"
        (root / split).mkdir(parents=True)
        pd.DataFrame([i[:4] for i in items], columns=["Image ID", "Storm ID", "Relative Time", "Ocean"]).to_csv(root / f"{stem}_features.csv", index=False)
        pd.DataFrame([(i[0], i[4]) for i in items], columns=["Image ID", "Wind Speed"]).to_csv(root / f"{stem}_labels.csv", index=False)
        for item in items:
            Image.fromarray(np.full((size, size), 128, dtype=np.uint8)).save(root / split / f"{item[0]}.jpg")
    return root


def test_manifest_schema_and_native_size(tmp_path: Path) -> None:
    root = _fake_root(tmp_path / "tc")
    frame = build_manifest(root, tmp_path / "manifest", tables=csv_tables)
    for column in ("sample_id", "image_path", "storm_id", "wind_speed_kt", "is_synthetic", "wind_averaging", "native_height", "native_width"):
        assert column in frame.columns
    assert not frame["is_synthetic"].any()
    assert set(frame["wind_averaging"]) == {"1min"}
    assert set(frame["native_height"]) == {366}
    assert frame["grey_ir"].all()
    summary = summarize(frame)
    assert summary["rows"] == 5
    assert summary["unique_storms"] == 3
    assert summary["storms_on_both_source_sides"] == 1
    assert summary["native_sizes"] == ["366x366"]


def test_missing_image_fails(tmp_path: Path) -> None:
    root = _fake_root(tmp_path / "tc")
    (root / "train" / "bbb_000.jpg").unlink()
    with pytest.raises(FileNotFoundError):
        build_manifest(root, tmp_path, tables=csv_tables)


def test_load_ir_keeps_native_size_and_rejects_colour(tmp_path: Path) -> None:
    grey = np.stack([np.full((50, 70), 10, np.uint8)] * 3, axis=-1)
    Image.fromarray(grey).save(tmp_path / "g.png")
    assert load_ir(tmp_path / "g.png").shape == (50, 70)
    colour = grey.copy()
    colour[..., 0] = 200
    Image.fromarray(colour).save(tmp_path / "c.png")
    with pytest.raises(ValueError, match="channels differ"):
        load_ir(tmp_path / "c.png")
