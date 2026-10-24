"""Feature values, temporal links, and train-only selection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import FeatureSpec, extract_feature_table, extract_image_features, model_matrix, select_features
from src.schema import FORBIDDEN_FEATURES
from src.splits import LeakageError


def _spec() -> FeatureSpec:
    return FeatureSpec()


def test_default_feature_count_and_names() -> None:
    spec = _spec()
    names = spec.feature_names()
    assert len(names) == len(set(names))
    assert 40 <= len(names) <= 55
    assert len(names) == 47
    assert names[0] == "pixel_min"
    assert "has_prev" in names
    assert "dav_simplified_r00" in names
    assert set(names).isdisjoint(FORBIDDEN_FEATURES)


def test_calibrated_prefix_is_explicit() -> None:
    names = FeatureSpec(calibrated=True).feature_names()
    assert names[0] == "tb_min"
    assert "delta_tb_mean" in names
    assert not any(name.startswith("pixel_") for name in names)


def test_bad_eye_geometry_is_rejected() -> None:
    with pytest.raises(ValueError, match="eye radius"):
        FeatureSpec(eye_radius_fraction=0.5, cdo_outer_fraction=0.2)


def test_constant_image_features_are_finite() -> None:
    image = np.full((16, 16), 5.0, dtype=np.float32)
    features = extract_image_features(image, _spec(), max_missing_fraction=0.30)
    assert features["pixel_mean"] == pytest.approx(5.0)
    assert features["pixel_std"] == pytest.approx(0.0)
    assert features["cold_frac_200"] == pytest.approx(1.0)
    assert features["eye_present"] == 0.0
    assert features["eye_cdo_contrast"] == pytest.approx(0.0)
    assert features["glcm_contrast"] == pytest.approx(0.0)
    assert features["glcm_energy"] == pytest.approx(1.0)
    assert features["haar_energy_ll"] == pytest.approx(1.0)
    assert np.isnan(features["dav_simplified_r00"])


def test_warm_center_sets_eye_flag() -> None:
    image = np.zeros((32, 32), dtype=np.float32)
    yy, xx = np.mgrid[0:32, 0:32]
    radius = np.hypot(yy - 15.5, xx - 15.5)
    image[radius <= 1.5] = 20.0
    features = extract_image_features(image, _spec(), max_missing_fraction=0.30)
    assert features["eye_cdo_contrast"] > 2.0
    assert features["eye_present"] == 1.0


def test_cold_core_is_not_called_an_eye() -> None:
    image = np.full((32, 32), 30.0, dtype=np.float32)
    yy, xx = np.mgrid[0:32, 0:32]
    radius = np.hypot(yy - 15.5, xx - 15.5)
    image[radius <= 3] = 5.0
    features = extract_image_features(image, _spec(), max_missing_fraction=0.30)
    assert features["eye_cdo_contrast"] < 0.0
    assert features["eye_present"] == 0.0


def test_radial_field_has_low_simplified_dav() -> None:
    yy, xx = np.mgrid[0:32, 0:32]
    image = np.hypot(yy - 15.5, xx - 15.5).astype(np.float32)
    features = extract_image_features(image, _spec(), max_missing_fraction=0.30)
    assert features["dav_simplified_r01"] < 0.2


def test_mostly_missing_image_is_rejected() -> None:
    image = np.full((8, 8), np.nan, dtype=np.float32)
    image[0, 0] = 1.0
    with pytest.raises(ValueError, match="missing fraction"):
        extract_image_features(image, _spec(), max_missing_fraction=0.30)


def _write_pair(tmp_path: Path, first: float, second: float) -> Path:
    np.save(tmp_path / "a.npy", np.full((16, 16), first, dtype=np.float32))
    np.save(tmp_path / "b.npy", np.full((16, 16), second, dtype=np.float32))
    return tmp_path / "manifest.csv"


def _pair_frame(split_b: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": "a",
                "image_path": "a.npy",
                "storm_id": "s",
                "timestamp": "2020-01-01T00:00:00Z",
                "wind_speed_kt": 40.0,
                "split": "train",
            },
            {
                "sample_id": "b",
                "image_path": "b.npy",
                "storm_id": "s",
                "timestamp": "2020-01-01T01:00:00Z",
                "wind_speed_kt": 42.0,
                "split": split_b,
            },
        ]
    )


def test_temporal_delta_stays_inside_the_split(tmp_path: Path) -> None:
    manifest = _write_pair(tmp_path, 0.0, 100.0)
    table, dropped = extract_feature_table(_pair_frame("train"), manifest, _spec())
    assert dropped == 0
    later = table.loc[table["sample_id"] == "b"].iloc[0]
    assert later["has_prev"] == 1.0
    assert later["delta_pixel_mean"] == pytest.approx(100.0)


def test_cross_split_temporal_link_is_not_used(tmp_path: Path) -> None:
    manifest = _write_pair(tmp_path, 0.0, 100.0)
    table, dropped = extract_feature_table(_pair_frame("test"), manifest, _spec())
    assert dropped == 1
    later = table.loc[table["sample_id"] == "b"].iloc[0]
    assert later["has_prev"] == 0.0
    assert np.isnan(later["delta_pixel_mean"])


def test_model_matrix_rejects_labels() -> None:
    frame = pd.DataFrame({"wind_speed_kt": [1.0, 2.0], "pixel_mean": [3.0, 4.0]})
    with pytest.raises(LeakageError, match="forbidden"):
        model_matrix(frame, ["wind_speed_kt"])


def _selection_frame() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    signal = rng.normal(size=30)
    frame = pd.DataFrame(
        {
            "split": ["train"] * 30,
            "f1": signal,
            "f2": signal * 1.0001,
            "f3": rng.normal(size=30),
            "fconst": np.ones(30),
        }
    )
    return frame, rng.normal(size=30)


def test_selection_drops_duplicates_and_constants() -> None:
    frame, target = _selection_frame()
    selected = select_features(
        frame,
        ["f1", "f2", "f3", "fconst"],
        target,
        source_split="train",
        correlation_threshold=0.95,
        max_features=10,
        method="correlation_rf",
        n_estimators=15,
        seed=0,
    )
    assert "f1" in selected.names
    assert "f2" not in selected.names
    assert "fconst" in selected.dropped_variance
    assert selected.source_split == "train"
    again = select_features(
        frame,
        ["f1", "f2", "f3", "fconst"],
        target,
        source_split="train",
        correlation_threshold=0.95,
        max_features=10,
        method="correlation_rf",
        n_estimators=15,
        seed=0,
    )
    assert again.names == selected.names


def test_selection_refuses_non_train_rows() -> None:
    frame, target = _selection_frame()
    with pytest.raises(LeakageError, match="train split only"):
        select_features(
            frame,
            ["f1", "f3"],
            target,
            source_split="test",
            method="none",
        )
    mixed = frame.copy()
    mixed.loc[0, "split"] = "val"
    with pytest.raises(LeakageError, match="selection frame"):
        select_features(
            mixed,
            ["f1", "f3"],
            target,
            source_split="train",
            method="none",
        )


def test_rfe_respects_the_cap() -> None:
    frame, target = _selection_frame()
    selected = select_features(
        frame,
        ["f1", "f2", "f3", "fconst"],
        target,
        source_split="train",
        max_features=2,
        method="rfe",
        n_estimators=12,
        seed=1,
    )
    assert 1 <= len(selected.names) <= 2
    assert set(selected.names).isdisjoint(FORBIDDEN_FEATURES)
