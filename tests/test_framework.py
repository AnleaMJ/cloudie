from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cloudie.config import load_config
from cloudie.data_loader import load_image, load_manifest, resolve_image
from cloudie.make_smoke_data import write_smoke
from cloudie.metrics import mean_baseline, regression_metrics
from cloudie.preprocessing import fit_norm_stats, prepare_image
from cloudie.schema import SchemaError, require_valid, wind_to_imd5
from cloudie.splits import LeakageError, assign_storms, null_cross_split_links

ROOT = Path(__file__).resolve().parents[1]


def test_project_bins_follow_configured_edges() -> None:
    assert wind_to_imd5(16) is None
    assert wind_to_imd5(33) == "D_DD"
    assert wind_to_imd5(34) == "CS"
    assert wind_to_imd5(90) == "VSCS"
    assert wind_to_imd5(91) == "ESCS_SUCS"
    assert wind_to_imd5(140) == "ESCS_SUCS"


def test_mixed_wind_averaging_is_rejected() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "image_path": ["a.npy", "b.npy"],
            "storm_id": ["s1", "s2"],
            "wind_speed_kt": [40, 50],
            "wind_averaging": ["1-min", "10-min"],
        }
    )
    with pytest.raises(SchemaError, match="wind_averaging"):
        require_valid(frame)


def test_random_split_rejected(tmp_path: Path) -> None:
    frame = load_manifest(write_smoke(tmp_path))
    with pytest.raises(LeakageError, match="random"):
        assign_storms(frame, "random")


def test_storm_sets_are_disjoint(tmp_path: Path) -> None:
    frame = load_manifest(write_smoke(tmp_path))
    split = assign_storms(frame, "storm_wise", test_fraction=0.25, val_fraction=0.25, seed=7)
    sets = [set(split.loc[split["split"] == name, "storm_id"]) for name in ("train", "val", "test")]
    assert sets[0].isdisjoint(sets[1])
    assert sets[0].isdisjoint(sets[2])
    assert sets[1].isdisjoint(sets[2])
    assert all(sets)


def test_year_boundary_storm_is_not_cut(tmp_path: Path) -> None:
    frame = load_manifest(write_smoke(tmp_path))
    split = assign_storms(frame, "chronological", holdout_years=[2021], seed=7)
    spanned = split[split["storm_id"] == "S06"]
    assert set(spanned["year"]) == {2020, 2021}
    assert set(spanned["split"]) == {"test"}


def test_holdout_basin_keeps_storms_whole(tmp_path: Path) -> None:
    frame = load_manifest(write_smoke(tmp_path))
    split = assign_storms(
        frame, "holdout_basin", holdout_basin="NIO", val_fraction=0.25, seed=7
    )
    nio = split[split["basin"] == "NIO"]
    assert set(nio["split"]) == {"test"}
    assert "test" not in set(split.loc[split["basin"] == "NA", "split"])


def test_cross_split_link_is_removed() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "storm_id": ["s", "s"],
            "split": ["train", "test"],
            "prev_sample_id": [pd.NA, "a"],
        }
    )
    cleaned, dropped = null_cross_split_links(frame)
    assert dropped == 1
    assert pd.isna(cleaned.loc[cleaned["sample_id"] == "b", "prev_sample_id"]).all()


def test_norm_stats_refuse_test_split() -> None:
    with pytest.raises(LeakageError):
        fit_norm_stats([np.zeros((1, 8, 8), dtype=np.float32)], "test")


def test_prepare_image_rejects_mostly_missing() -> None:
    image = np.full((32, 32), np.nan, dtype=np.float32)
    image[0, 0] = 1.0
    with pytest.raises(ValueError, match="missing fraction"):
        prepare_image(image, 16, None, max_missing_fraction=0.30)


def test_smoke_image_round_trip(tmp_path: Path) -> None:
    manifest = write_smoke(tmp_path, size=32)
    frame = load_manifest(manifest)
    path = resolve_image(manifest, frame.iloc[0]["image_path"])
    prepared = prepare_image(load_image(path), 16, None)
    assert prepared.shape == (1, 16, 16)
    assert np.isfinite(prepared).all()


def test_metrics_and_mean_baseline() -> None:
    scores = regression_metrics([10.0, 20.0], [12.0, 16.0])
    assert scores["mae_kt"] == pytest.approx(3.0)
    assert scores["bias_kt"] == pytest.approx(-1.0)
    assert mean_baseline([10.0, 20.0], 3).tolist() == [15.0, 15.0, 15.0]


def test_config_names_the_project_cloudie() -> None:
    smoke = load_config(ROOT / "config" / "config_smoke.yaml")
    full = load_config(ROOT / "config" / "config.yaml")
    assert smoke["project"]["name"] == "cloudie"
    assert full["project"]["name"] == "cloudie"
    assert "wind_speed_kt" in full["forbidden_features"]


def test_simple_cnn_forward_if_torch_present() -> None:
    torch = pytest.importorskip("torch")
    from cloudie.model import SimpleCNN

    output = SimpleCNN()(torch.zeros(2, 1, 32, 32))
    assert tuple(output.shape) == (2, 1)
