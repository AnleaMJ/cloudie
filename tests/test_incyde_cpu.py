"""Interim INCYDE CPU pipeline (scripts/incyde_cpu). No images, no downloads."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from scripts.incyde_cpu import prep_incyde, train_cpu  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _manifest(tmp_path: Path) -> Path:
    excl = pd.read_csv(ROOT / "incyde/audit/exclusions.csv")
    bad = excl.loc[excl["default_exclude"].astype(str).str.lower() == "true", "filename"].iloc[0]
    flag = excl.loc[excl["default_exclude"].astype(str).str.lower() == "false", "filename"].iloc[0]
    rows = []
    for name, aug in [(bad, None), (flag, None), ("01JAN2015_0000_1.jpg", None),
                      ("01JAN2015_0000_1_rot_180.jpg", "rot_180"), ("01JAN2015_0000_1_flip_h.jpg", "flip_h")]:
        rows.append({
            "file": f"Cropped_cyclone_data/Cropped_cyclone_images/{name}",
            "storm_id": "S", "incyde_sequence": 1, "label_kt": 55, "aug_type": aug,
            "source_frame": "Cropped_cyclone_data/Cropped_cyclone_images/01JAN2015_0000_1.jpg",
            "proposed_split": "train",
        })
    path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_prep_drops_only_default_exclude_and_keeps_labels(tmp_path: Path) -> None:
    frame, stats = prep_incyde.build_index(_manifest(tmp_path))
    assert stats["dropped_default_exclude"] == 1
    assert "flip_h" not in set(frame["aug_type"].dropna())
    assert set(frame["aug_type"].dropna()) == {"rot_180"}
    # INCYDE labels are IMD 3-minute winds and pass through unchanged
    assert (frame["wind_kt_3min"] == frame["label_kt"].astype(float)).all()
    assert (frame["wind_averaging_s"] == 180).all()
    assert not frame["wind_converted"].any()
    assert frame["qc_flag_kept"].notna().sum() == 1


def test_cv_folds_follow_groups_and_skip_test() -> None:
    table = train_cpu.build_cv_folds(5, 42)
    storms = pd.read_csv(ROOT / "incyde/audit/storm_split.csv")
    merged = storms.merge(table, on="storm_id", how="left")
    assert merged.loc[merged["proposed_split"] == "test", "fold"].isna().all()
    assert merged.loc[merged["proposed_split"] != "test", "fold"].notna().all()
    pool = merged[merged["proposed_split"] != "test"]
    assert (pool.groupby("cooccurrence_group")["fold"].nunique() == 1).all()
    assert table.equals(train_cpu.build_cv_folds(5, 42))


def test_imd7_thresholds() -> None:
    names = train_cpu.imd7_of(np.array([10, 17, 27, 28, 33, 34, 47, 48, 63, 64, 89, 90, 119, 120]))
    assert names.tolist() == ["below_D", "D", "D", "DD", "DD", "CS", "CS", "SCS", "SCS",
                              "VSCS", "VSCS", "ESCS", "ESCS", "SuCS"]


def test_augment_has_no_flips_and_rot180_is_exact() -> None:
    gen = torch.Generator().manual_seed(0)
    batch = torch.arange(2 * 16, dtype=torch.float32).view(2, 1, 4, 4).repeat(8, 1, 1, 1)
    out = train_cpu.augment(batch, None, {"rot_180": True, "masked_rotation": False}, gen)
    for original, augmented in zip(batch, out):
        assert torch.equal(augmented, original) or torch.equal(augmented, torch.rot90(original, 2, dims=(-2, -1)))
        assert not torch.equal(augmented, torch.flip(original, dims=(-1,))) or torch.equal(original, torch.flip(original, dims=(-1,)))


def test_circle_mask_and_rotation_keep_the_disc() -> None:
    mask = train_cpu.circle_mask(32)
    assert mask[0, 0] == 0 and mask[16, 16] == 1
    image = (torch.ones(1, 1, 32, 32) * mask)
    turned = train_cpu.rotate_batch(image, torch.tensor([0.7])) * mask
    inner = train_cpu.circle_mask(32).bool()
    assert torch.allclose(turned[0, 0][inner].mean(), torch.tensor(1.0), atol=0.05)
