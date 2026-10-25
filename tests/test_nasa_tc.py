"""NASA TC manifest builder on tiny fake tables. Never downloads data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.nasa_tc import (
    build_manifest,
    csv_tables,
    exclusions_for,
    load_ir,
    storm_split_check,
    summarize,
    to_working_grid,
)
from src.splits import assert_no_group_leakage, assign_storms


def _fake_root(root: Path, size: int = 366, extra: dict | None = None) -> Path:
    rows = {"train": [("aaa_000", "aaa", 0, 1, 30), ("aaa_001", "aaa", 1800, 1, 35), ("bbb_000", "bbb", 0, 2, 90)],
            "test": [("aaa_002", "aaa", 3600, 1, 40), ("ccc_000", "ccc", 0, 2, 120)]}
    for split, items in (extra or {}).items():
        rows[split] = rows[split] + items
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


def test_orig_size_and_working_resample_flag(tmp_path: Path) -> None:
    root = _fake_root(tmp_path / "tc")
    Image.fromarray(np.full((1093, 1093), 90, dtype=np.uint8)).convert("RGB").save(root / "train" / "bbb_000.jpg")
    frame = build_manifest(root, tmp_path / "m", tables=csv_tables)
    big = frame.set_index("sample_id").loc["bbb_000"]
    assert big["orig_size"] == 1093 and big["native_height"] == 1093
    assert big["working_resample"] == "area_average"
    assert (frame.loc[frame["sample_id"] != "bbb_000", "working_resample"] == "none").all()
    assert summarize(frame)["frames_to_area_average_to_366"] == 1


def test_to_working_grid_area_averages_and_keeps_366() -> None:
    grid = np.zeros((1093, 1093), np.float32)
    grid[:, 547:] = 300.0
    out, resampled = to_working_grid(grid)
    assert resampled and out.shape == (366, 366)
    assert abs(float(out.mean()) - float(grid.mean())) < 1.0  # area average keeps the mean
    same, again = to_working_grid(np.ones((366, 366), np.float32))
    assert not again and same.shape == (366, 366)


def test_colour_frames_are_excluded_with_reason(tmp_path: Path) -> None:
    root = _fake_root(tmp_path / "tc", extra={"train": [("npf_000", "npf", 0, 2, 50), ("npf_001", "npf", 1800, 2, 55)]})
    colour = np.zeros((366, 366, 3), np.uint8)
    colour[..., 0] = 200
    Image.fromarray(colour).save(root / "train" / "npf_001.jpg")  # npf_000 stays grey: excluded by id
    raw = build_manifest(root, tmp_path / "m", tables=csv_tables, keep_excluded=True)
    excluded = exclusions_for(raw)
    assert list(excluded.columns) == ["sample_id", "storm_id", "source_split", "reason"]
    assert set(excluded["sample_id"]) == {"npf_000", "npf_001"}
    assert excluded["reason"].str.len().gt(0).all()
    kept = build_manifest(root, tmp_path / "m", tables=csv_tables)
    assert not kept["sample_id"].isin(["npf_000", "npf_001"]).any()
    assert len(kept) == 5


def test_no_split_column_and_storm_split_has_no_leakage(tmp_path: Path) -> None:
    root = _fake_root(
        tmp_path / "tc",
        extra={"test": [(f"s{i}_000", f"s{i}", 0, 1 + i % 2, 30 + 10 * i) for i in range(8)]},
    )
    frame = build_manifest(root, tmp_path / "m", tables=csv_tables)
    assert "split" not in frame.columns  # official side lives only in source_split
    assert set(frame["source_split"]) == {"train", "test"}
    assigned = assign_storms(frame, "storm_wise", seed=42)
    assert_no_group_leakage(*[assigned.loc[assigned["split"] == s] for s in ("train", "val", "test")])
    # storm "aaa" spans both official sides but lands in one split.
    assert assigned.loc[assigned["storm_id"] == "aaa", "split"].nunique() == 1
    assert storm_split_check(frame)["leakage"] == "none"
