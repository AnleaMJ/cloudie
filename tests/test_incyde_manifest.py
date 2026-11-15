"""INCYDE manifest builder on tiny files. Never downloads the Zenodo zip."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from incyde.manifest import (
    RAW_3MIN_METRICS,
    ZIP_MD5,
    IncydeDataError,
    build_manifest,
    verify_zip_md5,
)
from src.schema import wind_averaging_seconds


def _write_tree(root: Path) -> tuple[Path, Path, Path]:
    images = root / "Cropped_cyclone_images"
    images.mkdir(parents=True)
    # 12NOV2013_0530 IST is 12 Nov 2013 00:00 UTC, a synoptic fix.
    # 12NOV2013_0600 IST is 00:30 UTC, not a fix.
    names = [
        ("12NOV2013_0530_1.jpg", 1, 90, "Train"),
        ("12NOV2013_0600_1.jpg", 1, 95, "Train"),
        ("12NOV2013_0530_1_flip_h.jpg", 1, 90, "Train"),
        ("13NOV2013_0830_1.jpg", 2, 40, "Validation"),
        ("14NOV2013_0000_1.jpg", 3, 30, "Test"),
        ("10SEP2017_0530_1.jpg", 42, 50, "Train"),
        ("15NOV2013_0300_1.jpg", 4, 55, "Train"),
    ]
    for name, _seq, _wind, _split in names:
        Image.fromarray(np.full((16, 16), 80, dtype=np.uint8), mode="L").save(images / name)
    labels = root / "Cropped_cyclone_labels.csv"
    pd.DataFrame(
        {
            "Sequence": [item[1] for item in names],
            "Name": ["A", "A", "A", "B", "C", "S42", "FLAG"],
            "MSWS kts": [item[2] for item in names],
            "Category": ["ESCS", "ESCS", "ESCS", "CS", "D", "SCS", "SCS"],
            "filename": [item[0] for item in names],
            "Set": [item[3] for item in names],
        }
    ).to_csv(labels, index=False)
    split = root / "storm_split.csv"
    pd.DataFrame(
        {
            "Sequence": [1, 2, 3, 4, 42],
            "storm_id": ["S1", "S2", "S3", "S4", "S42"],
            "name": ["A", "B", "C", "FLAG", "S42"],
            "basin": ["BOB", "BOB", "ARB", "BOB", "BOB"],
            "peak_imd_class": ["ESCS", "CS", "D", "SCS", "SCS"],
            "proposed_split": ["train", "val", "test", "train", "train"],
            "cooccurrence_group": [1, 2, 3, 4, 5],
        }
    ).to_csv(split, index=False)
    exclusions = root / "exclusions.csv"
    pd.DataFrame(
        {
            "filename": [
                "10SEP2017_0530_1.jpg",
                "15NOV2013_0300_1.jpg",
                "15NOV2013_0300_1_flip_h.jpg",
            ],
            "qc_flag": [
                "wrong_date_imd_daymonth_swap",
                "label_ne_imd_interp",
                "label_ne_imd_interp",
            ],
            "default_exclude": ["true", "false", "false"],
        }
    ).to_csv(exclusions, index=False)
    return labels, split, exclusions


def test_default_manifest_keeps_flagged_rows_and_raw_labels(tmp_path: Path) -> None:
    labels, split, exclusions = _write_tree(tmp_path / "src")
    frame, summary = build_manifest(
        tmp_path / "src",
        tmp_path / "manifest.csv",
        storm_split_path=split,
        exclusions_path=exclusions,
        labels_path=labels,
    )
    assert "12NOV2013_0530_1_flip_h" not in set(frame["sample_id"])
    assert "10SEP2017_0530_1" not in set(frame["sample_id"])
    assert "15NOV2013_0300_1" in set(frame["sample_id"])
    assert set(frame["wind_averaging"]) == {"3-min"}
    assert set(frame["wind_averaging_s"]) == {180}
    assert wind_averaging_seconds("3-min") == 180
    assert summary["default_excluded_originals"] == 1
    assert summary["flagged_not_default_originals"] == 1
    assert summary["augmented_dropped"] == 1
    assert summary["metrics_use"] == RAW_3MIN_METRICS
    assert "no 1-minute conversion" in str(summary["metrics_use"])
    assert summary["year_holdout_2019"] == "not_built"
    by_id = frame.set_index("sample_id")
    assert bool(by_id.loc["12NOV2013_0530_1", "on_fix"]) is True
    assert bool(by_id.loc["12NOV2013_0600_1", "on_fix"]) is False
    assert by_id.loc["12NOV2013_0530_1", "split"] == "train"
    assert by_id.loc["13NOV2013_0830_1", "split"] == "val"
    assert by_id.loc["14NOV2013_0000_1", "split"] == "test"
    assert by_id.loc["12NOV2013_0530_1", "cv_group"] == "1"
    assert by_id.loc["12NOV2013_0530_1", "source_split"] == "train"


def test_drop_flagged_removes_the_optional_flags(tmp_path: Path) -> None:
    labels, split, exclusions = _write_tree(tmp_path / "src")
    frame, summary = build_manifest(
        tmp_path / "src",
        tmp_path / "manifest.csv",
        storm_split_path=split,
        exclusions_path=exclusions,
        labels_path=labels,
        drop_flagged=True,
    )
    assert "15NOV2013_0300_1" not in set(frame["sample_id"])
    assert summary["drop_flagged"] is True


def test_bad_md5_stops_before_a_manifest(tmp_path: Path) -> None:
    path = tmp_path / "Cropped_cyclone_data.zip"
    path.write_bytes(b"not the dataset")
    with pytest.raises(IncydeDataError, match=ZIP_MD5):
        verify_zip_md5(path)
    assert not (tmp_path / "manifest.csv").exists()


def test_loader_source_does_not_build_year_holdout() -> None:
    text = Path(__file__).resolve().parents[1].joinpath("incyde", "manifest.py").read_text(encoding="utf-8")
    assert "def year_holdout" not in text
    assert "download=True" not in text
    assert "requests.get" not in text
    assert "urlretrieve" not in text
