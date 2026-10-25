"""Feature values, temporal links, and train-only selection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.features import (
    FeatureSpec,
    extract_feature_table,
    extract_image_features,
    model_matrix,
    native_shape,
    select_features,
)
from src.schema import FORBIDDEN_FEATURES
from src.splits import LeakageError


def _spec() -> FeatureSpec:
    return FeatureSpec()


def test_default_feature_count_and_names() -> None:
    spec = _spec()
    groups = spec.feature_groups()
    names = spec.feature_names()
    flattened = tuple(name for group in groups.values() for name in group)
    assert names == flattened
    assert len(names) == len(set(names))
    assert 40 <= len(names) <= 50
    assert len(names) == 47
    assert names[0] == "pixel_min"
    assert all(name.startswith("pixel_") for name in groups["pixel_stats"])
    assert "has_prev" in groups["temporal"]
    assert "dav_simplified_r00" in groups["dav_simplified"]
    assert groups["glcm"]
    assert groups["haar"]
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
    """Synthetic manifest rows. The arrays are not satellite imagery."""
    return pd.DataFrame(
        [
            {
                "sample_id": "a",
                "image_path": "a.npy",
                "storm_id": "s",
                "timestamp": "2020-01-01T00:00:00Z",
                "wind_speed_kt": 40.0,
                "split": "train",
                "is_synthetic": True,
            },
            {
                "sample_id": "b",
                "image_path": "b.npy",
                "storm_id": "s",
                "timestamp": "2020-01-01T01:00:00Z",
                "wind_speed_kt": 42.0,
                "split": split_b,
                "is_synthetic": True,
            },
        ]
    )


def test_temporal_delta_stays_inside_the_split(tmp_path: Path) -> None:
    """Synthetic frames. The delta uses the same-split previous frame only."""
    manifest = _write_pair(tmp_path, 0.0, 100.0)
    table, dropped = extract_feature_table(_pair_frame("train"), manifest, _spec())
    assert dropped == 0
    later = table.loc[table["sample_id"] == "b"].iloc[0]
    assert later["has_prev"] == 1.0
    assert later["delta_pixel_mean"] == pytest.approx(100.0)


def test_cross_split_temporal_link_is_not_used(tmp_path: Path) -> None:
    """Synthetic frames. A previous frame in another split is nulled."""
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
            "storm_id": [f"s{index}" for index in range(30)],
            "is_synthetic": [True] * 30,
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
    assert selected.train_storms == frozenset(frame["storm_id"])
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


def test_selection_refuses_non_train_source_and_leaked_storms() -> None:
    frame, target = _selection_frame()
    with pytest.raises(LeakageError, match="train split only"):
        select_features(
            frame,
            ["f1", "f3"],
            target,
            source_split="test",
            method="none",
        )
    bare = frame.drop(columns=["storm_id"])
    with pytest.raises(LeakageError, match="storm_id"):
        select_features(
            bare,
            ["f1", "f3"],
            target,
            source_split="train",
            method="none",
        )
    leaked = frame.copy()
    leaked.loc[0, "storm_id"] = leaked.loc[1, "storm_id"]
    leaked.loc[0, "split"] = "val"
    with pytest.raises(LeakageError, match="s1"):
        select_features(
            leaked,
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
    assert selected.train_storms == frozenset(frame["storm_id"])


def test_pixel_stats_on_synthetic_image() -> None:
    """Synthetic array. Pixel stats are the stored values, not brightness temperature."""
    image = np.arange(16, dtype=np.float32).reshape(4, 4)
    features = extract_image_features(image, _spec(), max_missing_fraction=0.30)
    finite = image.ravel()
    percentiles = np.percentile(finite, [10, 25, 50, 75, 90])
    assert features["pixel_min"] == pytest.approx(float(finite.min()))
    assert features["pixel_max"] == pytest.approx(float(finite.max()))
    assert features["pixel_mean"] == pytest.approx(float(finite.mean()))
    assert features["pixel_std"] == pytest.approx(float(finite.std()))
    assert features["pixel_p10"] == pytest.approx(float(percentiles[0]))
    assert features["pixel_p50"] == pytest.approx(float(percentiles[2]))
    assert features["pixel_p90"] == pytest.approx(float(percentiles[4]))
    assert features["pixel_iqr"] == pytest.approx(float(percentiles[3] - percentiles[1]))
    assert not any(name.startswith("tb_") for name in features)


def test_cold_cloud_fractions_on_synthetic_image() -> None:
    """Synthetic array. Cold fraction is the share of pixels below each threshold."""
    image = np.full((8, 8), 250.0, dtype=np.float32)
    image[:2, :] = 100.0
    features = extract_image_features(image, _spec(), max_missing_fraction=0.30)
    assert features["cold_frac_200"] == pytest.approx(0.25)
    assert features["cold_frac_240"] == pytest.approx(0.25)
    warm = extract_image_features(np.full((8, 8), 300.0, dtype=np.float32), _spec())
    assert warm["cold_frac_200"] == pytest.approx(0.0)
    assert warm["cold_frac_240"] == pytest.approx(0.0)


def test_radial_profile_on_synthetic_image() -> None:
    """Synthetic array. The inner annulus is warmer than the outer one."""
    image = np.zeros((32, 32), dtype=np.float32)
    yy, xx = np.mgrid[0:32, 0:32]
    image[np.hypot(yy - 15.5, xx - 15.5) <= 4] = 50.0
    features = extract_image_features(image, _spec(), max_missing_fraction=0.30)
    assert features["radial_mean_00"] > features["radial_mean_07"]
    assert features["radial_mean_00"] == pytest.approx(50.0)


def test_eye_cdo_contrast_on_synthetic_image() -> None:
    """Synthetic array. A warm center is flagged; a cold center is not an eye."""
    spec = _spec()
    warm = _synthetic_core(32, 32, 30.0, 0.0)
    cold = _synthetic_core(32, 32, 0.0, 30.0)
    warm_features = extract_image_features(warm, spec)
    cold_features = extract_image_features(cold, spec)
    assert warm_features["eye_center_mean"] > warm_features["cdo_annulus_mean"]
    assert warm_features["eye_cdo_contrast"] >= spec.eye_contrast_min
    assert warm_features["eye_present"] == 1.0
    assert cold_features["eye_cdo_contrast"] < 0.0
    assert cold_features["eye_present"] == 0.0


def test_simplified_dav_on_synthetic_image() -> None:
    """Synthetic arrays. Simplified DAV is low on a radial field and high on a shear."""
    yy, xx = np.mgrid[0:48, 0:48]
    radial = np.hypot(yy - 23.5, xx - 23.5).astype(np.float32)
    horizontal = xx.astype(np.float32)
    radial_features = extract_image_features(radial, _spec())
    shear_features = extract_image_features(horizontal, _spec())
    assert radial_features["dav_simplified_r01"] < 0.2
    assert shear_features["dav_simplified_r01"] > 0.5


def test_glcm_texture_on_synthetic_image() -> None:
    """Synthetic arrays. A checkerboard is less homogeneous than a flat field."""
    smooth = np.full((32, 32), 4.0, dtype=np.float32)
    yy, xx = np.indices((32, 32))
    checker = ((yy + xx) % 2).astype(np.float32)
    smooth_features = extract_image_features(smooth, _spec())
    checker_features = extract_image_features(checker, _spec())
    assert smooth_features["glcm_contrast"] == pytest.approx(0.0)
    assert smooth_features["glcm_energy"] == pytest.approx(1.0)
    assert checker_features["glcm_contrast"] > smooth_features["glcm_contrast"]
    assert checker_features["glcm_homogeneity"] < smooth_features["glcm_homogeneity"]
    assert checker_features["glcm_energy"] < smooth_features["glcm_energy"]


def test_haar_energies_on_synthetic_image() -> None:
    """Synthetic arrays. Column stripes put energy in the horizontal Haar detail."""
    flat = np.full((32, 32), 6.0, dtype=np.float32)
    stripes = np.zeros((32, 32), dtype=np.float32)
    stripes[:, 1::2] = 1.0
    flat_features = extract_image_features(flat, _spec())
    stripe_features = extract_image_features(stripes, _spec())
    assert flat_features["haar_energy_ll"] == pytest.approx(1.0)
    assert stripe_features["haar_energy_lh"] > stripe_features["haar_energy_hl"]
    assert stripe_features["haar_energy_hh"] == pytest.approx(0.0)
    assert stripe_features["haar_energy_ll"] == pytest.approx(stripe_features["haar_energy_lh"])
    assert sum(stripe_features[name] for name in ("haar_energy_ll", "haar_energy_lh", "haar_energy_hl", "haar_energy_hh")) == pytest.approx(1.0)


@pytest.mark.parametrize("image_size", [128, 201, 366, "native", None])
def test_features_use_native_shape_not_config_image_size(image_size: int | str | None) -> None:
    """Synthetic crops. Geometry follows the array, not ``data.image_size``."""
    config = load_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    config["data"]["image_size"] = image_size
    spec = FeatureSpec.from_config(config)
    spike = np.zeros((201, 173), dtype=np.float32)
    spike[100, 80] = 80.0
    features = extract_image_features(spike, spec)
    assert native_shape(spike) == (201, 173)
    assert features["pixel_max"] == pytest.approx(80.0)
    assert features["pixel_mean"] == pytest.approx(80.0 / (201 * 173))
    for height, width in ((17, 19), (64, 64), (201, 173)):
        core = _synthetic_core(height, width, 40.0, 0.0)
        assert native_shape(core) == (height, width)
        assert extract_image_features(core, spec)["eye_present"] == 1.0


def _synthetic_core(height: int, width: int, core: float, ambient: float) -> np.ndarray:
    """Synthetic array. Not a satellite crop."""
    image = np.full((height, width), ambient, dtype=np.float32)
    yy, xx = np.mgrid[0:height, 0:width]
    radius = np.hypot(yy - (height - 1) / 2.0, xx - (width - 1) / 2.0)
    image[radius <= 0.12 * min(height, width) / 2.0] = core
    return image


def _selection_block(
    split: str,
    storm_prefix: str,
    columns: dict[str, np.ndarray],
    target: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    count = len(target)
    frame = pd.DataFrame(columns)
    frame["split"] = split
    frame["storm_id"] = [f"{storm_prefix}{index}" for index in range(count)]
    frame["is_synthetic"] = True
    return frame, np.asarray(target, dtype=np.float64)


def test_val_and_test_rows_do_not_influence_selection() -> None:
    """Synthetic rows. Val and test storms do not change correlation, RF, or RFE."""
    rng = np.random.default_rng(0)
    n_train = 40
    signal = np.linspace(-2.0, 2.0, n_train)
    orthogonal = rng.normal(size=n_train)
    orthogonal = orthogonal - signal * np.dot(signal, orthogonal) / np.dot(signal, signal)
    train, y_train = _selection_block(
        "train",
        "tr",
        {
            "f_a": signal,
            "f_b": orthogonal,
            "f_const": np.ones(n_train),
            "f_signal": signal,
            "f_decoy": rng.normal(size=n_train),
        },
        signal,
    )
    n_held = 80
    held_axis = np.linspace(-50.0, 50.0, n_held)
    val, y_val = _selection_block(
        "val",
        "va",
        {
            "f_a": held_axis[:40],
            "f_b": held_axis[:40],
            "f_const": np.arange(40, dtype=float),
            "f_signal": rng.normal(size=40),
            "f_decoy": held_axis[:40],
        },
        held_axis[:40],
    )
    test, y_test = _selection_block(
        "test",
        "te",
        {
            "f_a": held_axis[40:],
            "f_b": held_axis[40:],
            "f_const": np.arange(40, dtype=float) + 100.0,
            "f_signal": rng.normal(size=40),
            "f_decoy": held_axis[40:],
        },
        held_axis[40:],
    )
    mixed = pd.concat([train, val, test], ignore_index=True)
    y_control = np.concatenate([y_train, y_val, y_test])
    # Wild held-out labels would dominate a fit that used them.
    y_wild = y_control.copy()
    y_wild[len(train) :] = 10_000.0
    as_train = pd.concat([val, test], ignore_index=True)
    as_train["split"] = "train"
    as_train["storm_id"] = [f"leak{index}" for index in range(len(as_train))]
    contaminated = pd.concat([train, as_train], ignore_index=True)
    correlation_names = ["f_a", "f_b", "f_const"]
    rank_names = ["f_signal", "f_decoy"]
    for method, names, cap in (
        ("none", correlation_names, 10),
        ("correlation_rf", rank_names, 1),
        ("rfe", rank_names, 1),
    ):
        alone = select_features(
            train,
            names,
            y_train,
            source_split="train",
            correlation_threshold=0.95,
            max_features=cap,
            method=method,
            n_estimators=20,
            seed=0,
        )
        with_held_out = select_features(
            mixed,
            names,
            y_wild,
            source_split="train",
            correlation_threshold=0.95,
            max_features=cap,
            method=method,
            n_estimators=20,
            seed=0,
        )
        assert with_held_out.names == alone.names
        assert with_held_out.importances == alone.importances
        assert with_held_out.dropped_variance == alone.dropped_variance
        assert with_held_out.dropped_correlation == alone.dropped_correlation
        assert with_held_out.train_storms == frozenset(train["storm_id"])
        assert with_held_out.train_storms.isdisjoint(set(val["storm_id"]) | set(test["storm_id"]))
        influenced = select_features(
            contaminated,
            names,
            y_control,
            source_split="train",
            correlation_threshold=0.95,
            max_features=cap,
            method=method,
            n_estimators=20,
            seed=0,
        )
        assert influenced.names != alone.names
        assert not influenced.train_storms <= frozenset(train["storm_id"])
    filtered = select_features(
        contaminated,
        correlation_names,
        y_control,
        source_split="train",
        method="none",
    )
    assert "f_b" not in filtered.names
    assert "f_const" in filtered.names
    ranked = select_features(
        train,
        rank_names,
        y_train,
        source_split="train",
        max_features=1,
        method="correlation_rf",
        n_estimators=20,
        seed=0,
    )
    assert ranked.names == ("f_signal",)
    decoy = select_features(
        contaminated,
        rank_names,
        y_control,
        source_split="train",
        max_features=1,
        method="correlation_rf",
        n_estimators=20,
        seed=0,
    )
    assert decoy.names == ("f_decoy",)
