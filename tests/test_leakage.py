"""Leak-free splits and forbidden inputs.

Fixtures in this module are synthetic. They are not cyclones, and nothing
here downloads imagery. Image size comes from the call, not a hardcoded 128.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cloudie.sequences import stack_windows
from cloudie.train import (
    assert_model_inputs,
    fit_split_normalization,
    split_frame,
    stack_split,
)
from src.baselines import run_baselines
from src.features import FeatureSpec, extract_feature_table, model_matrix, select_features
from src.preprocessing import fit_norm_stats

SPLIT_MODULES = ("src.splits",)
IMAGE_SIZE = 8


def _splits(module_name: str):
    return importlib.import_module(module_name)


def _boundary_frame() -> pd.DataFrame:
    """Synthetic rows. Storm CROSS sits on a year boundary and a basin boundary."""
    spec = [
        ("CROSS", "c0", 2020, "NA", 40.0, "2020-12-31T00:00:00Z"),
        ("CROSS", "c1", 2020, "NA", 42.0, "2020-12-31T06:00:00Z"),
        ("CROSS", "c2", 2021, "NIO", 44.0, "2021-01-01T00:00:00Z"),
        ("A", "a0", 2018, "NA", 30.0, "2018-06-01T00:00:00Z"),
        ("A", "a1", 2018, "NA", 31.0, "2018-06-01T03:00:00Z"),
        ("B", "b0", 2018, "WP", 55.0, "2018-07-01T00:00:00Z"),
        ("C", "cC0", 2019, "WP", 60.0, "2019-08-01T00:00:00Z"),
        ("D", "d0", 2019, "NA", 70.0, "2019-09-01T00:00:00Z"),
        ("H1", "h1", 2021, "NIO", 90.0, "2021-06-01T00:00:00Z"),
        ("H2", "h2", 2021, "NIO", 100.0, "2021-07-01T00:00:00Z"),
    ]
    rows = []
    for storm_id, sample_id, year, basin, wind, timestamp in spec:
        rows.append(
            {
                "sample_id": sample_id,
                "storm_id": storm_id,
                "year": year,
                "basin": basin,
                "wind_speed_kt": wind,
                "timestamp": timestamp,
                "wind_averaging": "synthetic",
                "is_synthetic": True,
            }
        )
    return pd.DataFrame(rows)


def _assign(module, frame: pd.DataFrame, strategy: str) -> pd.DataFrame:
    kwargs: dict = {"test_fraction": 0.25, "val_fraction": 0.25, "seed": 7}
    if strategy == "chronological":
        kwargs["holdout_years"] = [2021]
    if strategy == "holdout_basin":
        kwargs["holdout_basin"] = "NIO"
    return module.assign_storms(frame, strategy, **kwargs)


def _assert_disjoint(module, assigned: pd.DataFrame) -> dict[str, frozenset[str]]:
    groups = module.storm_ids_by_split(assigned)
    assert groups["train"]
    assert groups["val"]
    assert groups["test"]
    assert groups["train"].isdisjoint(groups["val"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["val"].isdisjoint(groups["test"])
    assert set().union(*groups.values()) == set(assigned["storm_id"].astype(str))
    counts = assigned.groupby(assigned["storm_id"].astype(str))["split"].nunique()
    assert int(counts.max()) == 1
    return groups


@pytest.mark.parametrize("module_name", SPLIT_MODULES)
@pytest.mark.parametrize("strategy", ("storm_wise", "chronological", "holdout_basin"))
def test_strategies_keep_storms_whole_and_disjoint(module_name: str, strategy: str) -> None:
    module = _splits(module_name)
    frame = _boundary_frame()
    assigned = _assign(module, frame, strategy)
    again = _assign(module, frame, strategy)
    assert assigned["split"].tolist() == again["split"].tolist()
    groups = _assert_disjoint(module, assigned)
    cross = assigned[assigned["storm_id"] == "CROSS"]
    assert set(cross["year"]) == {2020, 2021}
    assert set(cross["basin"]) == {"NA", "NIO"}
    assert set(cross["split"]) == {cross["split"].iloc[0]}
    if strategy in {"chronological", "holdout_basin"}:
        assert set(cross["split"]) == {"test"}
        assert "CROSS" in groups["test"]
        held_out = assigned[assigned["storm_id"].astype(str).isin(groups["test"])]
        assert set(held_out["split"]) == {"test"}
        assert set(cross.loc[cross["year"] == 2020, "split"]) == {"test"}
        assert set(cross.loc[cross["basin"] == "NA", "split"]) == {"test"}


@pytest.mark.parametrize("module_name", SPLIT_MODULES)
def test_previous_frame_follows_relative_time_when_there_is_no_timestamp(module_name: str) -> None:
    module = _splits(module_name)
    frame = pd.DataFrame(
        {
            "sample_id": ["b", "a", "c"],
            "storm_id": ["s", "s", "s"],
            "relative_time_s": [10, 0, 20],
        }
    )
    linked = module.add_previous_frame_id(frame).set_index("sample_id")
    assert pd.isna(linked.loc["a", "prev_sample_id"])
    assert linked.loc["b", "prev_sample_id"] == "a"
    assert linked.loc["c", "prev_sample_id"] == "b"
    stamped = frame.assign(timestamp=["2020-01-01T03:00:00Z", "2020-01-01T00:00:00Z", "2020-01-01T01:00:00Z"])
    by_stamp = module.add_previous_frame_id(stamped).set_index("sample_id")
    assert pd.isna(by_stamp.loc["a", "prev_sample_id"])
    assert by_stamp.loc["c", "prev_sample_id"] == "a"
    assert by_stamp.loc["b", "prev_sample_id"] == "c"


@pytest.mark.parametrize("module_name", SPLIT_MODULES)
def test_random_splits_raise(module_name: str) -> None:
    module = _splits(module_name)
    frame = _boundary_frame()
    for name in ("random", "Random", "frame", "shuffle", "random_frame"):
        with pytest.raises(module.LeakageError, match="random"):
            module.assign_storms(frame, name)


@pytest.mark.parametrize("module_name", SPLIT_MODULES)
def test_cross_split_prev_links_are_nulled(module_name: str) -> None:
    module = _splits(module_name)
    frame = pd.DataFrame(
        {
            "sample_id": [1, 2, 3],
            "storm_id": ["s", "s", "s"],
            "split": ["train", "train", "test"],
            "prev_sample_id": [pd.NA, 1, 2],
            "is_synthetic": [True, True, True],
        }
    )
    cleaned, dropped = module.null_cross_split_links(frame)
    assert dropped == 1
    same_split = cleaned.loc[cleaned["sample_id"] == 2, "prev_sample_id"].iloc[0]
    assert str(same_split) == "1"
    assert pd.isna(cleaned.loc[cleaned["sample_id"] == 3, "prev_sample_id"]).all()

    unknown = frame.copy()
    unknown.loc[unknown["sample_id"] == 2, "prev_sample_id"] = "missing-frame"
    cleaned, dropped = module.null_cross_split_links(unknown)
    assert dropped == 2
    assert pd.isna(cleaned.loc[cleaned["sample_id"] == 2, "prev_sample_id"]).all()


def test_split_frame_nulls_cross_split_links_and_keeps_same_storm_links() -> None:
    frame = _boundary_frame()
    config = {
        "split": {
            "strategy": "chronological",
            "test_fraction": 0.25,
            "val_fraction": 0.25,
            "seed": 7,
            "holdout_years": [2021],
            "holdout_basin": None,
        },
        "project": {"seed": 7},
    }
    linked = split_frame(frame, config)
    cross = linked[linked["storm_id"] == "CROSS"].sort_values("timestamp")
    assert list(cross["sample_id"]) == ["c0", "c1", "c2"]
    assert set(cross["split"]) == {"test"}
    assert pd.isna(cross.iloc[0]["prev_sample_id"])
    assert cross.iloc[1]["prev_sample_id"] == "c0"
    assert cross.iloc[2]["prev_sample_id"] == "c1"

    peek = _assign(_splits("src.splits"), frame, "chronological")
    train_ids = peek.loc[peek["split"] == "train", "sample_id"].tolist()
    test_id = peek.loc[peek["split"] == "test", "sample_id"].iloc[0]
    assert len(train_ids) >= 2
    dirty = frame.copy()
    dirty["prev_sample_id"] = pd.NA
    dirty.loc[dirty["sample_id"] == train_ids[0], "prev_sample_id"] = test_id
    dirty.loc[dirty["sample_id"] == train_ids[1], "prev_sample_id"] = "missing-frame"
    cleaned = split_frame(dirty, config)
    for sample_id in train_ids[:2]:
        assert pd.isna(cleaned.loc[cleaned["sample_id"] == sample_id, "prev_sample_id"]).all()


def _write_constant(root: Path, sample_id: str, value: float) -> str:
    folder = root / "images"
    folder.mkdir(parents=True, exist_ok=True)
    relative = f"images/{sample_id}.npy"
    array = np.full((IMAGE_SIZE, IMAGE_SIZE), value, dtype=np.float32)
    np.save(root / relative, array)
    return relative


def _image_config(model_name: str) -> dict:
    return {
        "data": {"image_size": IMAGE_SIZE, "max_missing_fraction": 0.30},
        "model": {"name": model_name, "n_frames": 4},
    }


def test_norm_stats_use_train_pixels_only(tmp_path: Path) -> None:
    rows = []
    plan = [
        ("tr0", "T0", "train", 0.0),
        ("tr1", "T1", "train", 4.0),
        ("va0", "V0", "val", 100.0),
        ("te0", "X0", "test", 80.0),
    ]
    for sample_id, storm_id, split, pixel in plan:
        rows.append(
            {
                "sample_id": sample_id,
                "image_path": _write_constant(tmp_path, sample_id, pixel),
                "storm_id": storm_id,
                "split": split,
                "timestamp": "2020-01-01T00:00:00Z",
                "wind_speed_kt": 35.0,
                "_class": "CS",
                "wind_averaging": "synthetic",
                "is_synthetic": True,
            }
        )
    frame = pd.DataFrame(rows)
    manifest = tmp_path / "manifest.csv"
    config = _image_config("simple_cnn")
    stats = fit_split_normalization(frame, manifest, config)
    assert stats.source_split == "train"
    assert stats.mean == pytest.approx(2.0)
    leaked = fit_norm_stats(
        [np.full((1, IMAGE_SIZE, IMAGE_SIZE), value, dtype=np.float32) for value in (0.0, 4.0, 100.0)],
        "train",
    )
    assert leaked.mean != pytest.approx(stats.mean)

    np.save(tmp_path / "images" / "va0.npy", np.full((IMAGE_SIZE, IMAGE_SIZE), 250.0, dtype=np.float32))
    np.save(tmp_path / "images" / "te0.npy", np.full((IMAGE_SIZE, IMAGE_SIZE), -40.0, dtype=np.float32))
    again = fit_split_normalization(frame, manifest, config)
    assert again.mean == pytest.approx(stats.mean)
    assert again.std == pytest.approx(stats.std)

    with pytest.raises(Exception, match="train split only"):
        fit_norm_stats([np.zeros((1, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)], "val")
    with pytest.raises(Exception, match="train split only"):
        fit_norm_stats([np.zeros((1, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)], "test")


def test_sequence_norm_stats_ignore_other_splits(tmp_path: Path) -> None:
    rows = []
    for index in range(4):
        rows.append(
            {
                "sample_id": f"tr{index}",
                "image_path": _write_constant(tmp_path, f"tr{index}", 1.0),
                "storm_id": "T0",
                "split": "train",
                "timestamp": f"2020-01-01T0{index}:00:00Z",
                "wind_speed_kt": 40.0 + index,
                "wind_averaging": "synthetic",
                "is_synthetic": True,
            }
        )
    rows.append(
        {
            "sample_id": "va0",
            "image_path": _write_constant(tmp_path, "va0", 50.0),
            "storm_id": "V0",
            "split": "val",
            "timestamp": "2020-02-01T00:00:00Z",
            "wind_speed_kt": 90.0,
            "wind_averaging": "synthetic",
            "is_synthetic": True,
        }
    )
    frame = pd.DataFrame(rows)
    stats = fit_split_normalization(frame, tmp_path / "manifest.csv", _image_config("convlstm"))
    assert stats.source_split == "train"
    assert stats.mean == pytest.approx(1.0)
    np.save(tmp_path / "images" / "va0.npy", np.full((IMAGE_SIZE, IMAGE_SIZE), 9.0, dtype=np.float32))
    assert fit_split_normalization(frame, tmp_path / "manifest.csv", _image_config("convlstm")).mean == pytest.approx(
        1.0
    )


def test_temporal_delta_does_not_cross_splits(tmp_path: Path) -> None:
    _write_constant(tmp_path, "a", 0.0)
    _write_constant(tmp_path, "b", 5.0)
    manifest = tmp_path / "manifest.csv"
    base = {
        "storm_id": "s",
        "wind_speed_kt": 40.0,
        "wind_averaging": "synthetic",
        "is_synthetic": True,
        "r34_nmi": 20.0,
        "rmw_nmi": 8.0,
        "imd5_class": "CS",
        "intensity_bin": 1,
    }
    same = pd.DataFrame(
        [
            {"sample_id": "a", "image_path": "images/a.npy", "timestamp": "2020-01-01T00:00:00Z", "split": "train", **base},
            {"sample_id": "b", "image_path": "images/b.npy", "timestamp": "2020-01-01T01:00:00Z", "split": "train", **base},
        ]
    )
    table, dropped = extract_feature_table(same, manifest, FeatureSpec())
    assert dropped == 0
    later = table.loc[table["sample_id"] == "b"].iloc[0]
    assert later["has_prev"] == 1.0
    assert later["delta_pixel_mean"] == pytest.approx(5.0)

    crossed = same.copy()
    crossed.loc[crossed["sample_id"] == "b", "split"] = "test"
    table, dropped = extract_feature_table(crossed, manifest, FeatureSpec())
    assert dropped == 1
    later = table.loc[table["sample_id"] == "b"].iloc[0]
    assert later["has_prev"] == 0.0
    assert np.isnan(later["delta_pixel_mean"])


def test_feature_matrix_is_invariant_to_labels(tmp_path: Path) -> None:
    _write_constant(tmp_path, "a", 0.0)
    _write_constant(tmp_path, "b", 5.0)
    manifest = tmp_path / "manifest.csv"
    frame = pd.DataFrame(
        [
            {
                "sample_id": "a",
                "image_path": "images/a.npy",
                "storm_id": "s",
                "timestamp": "2020-01-01T00:00:00Z",
                "split": "train",
                "wind_speed_kt": 40.0,
                "r34_nmi": 20.0,
                "rmw_nmi": 8.0,
                "imd5_class": "CS",
                "intensity_bin": 1,
                "wind_averaging": "synthetic",
                "is_synthetic": True,
            },
            {
                "sample_id": "b",
                "image_path": "images/b.npy",
                "storm_id": "s",
                "timestamp": "2020-01-01T01:00:00Z",
                "split": "train",
                "wind_speed_kt": 42.0,
                "r34_nmi": 22.0,
                "rmw_nmi": 9.0,
                "imd5_class": "CS",
                "intensity_bin": 1,
                "wind_averaging": "synthetic",
                "is_synthetic": True,
            },
        ]
    )
    spec = FeatureSpec()
    original, _dropped = extract_feature_table(frame, manifest, spec)
    shifted = frame.copy()
    shifted["wind_speed_kt"] = [90.0, 110.0]
    shifted["r34_nmi"] = [80.0, 81.0]
    shifted["rmw_nmi"] = [1.0, 2.0]
    shifted["imd5_class"] = ["ESCS_SUCS", "D_DD"]
    shifted["intensity_bin"] = [4, 0]
    changed, _dropped = extract_feature_table(shifted, manifest, spec)
    names = list(spec.feature_names())
    pd.testing.assert_frame_equal(original[names], changed[names])
    assert original["wind_speed_kt"].tolist() != changed["wind_speed_kt"].tolist()
    for module_name in SPLIT_MODULES:
        module = _splits(module_name)
        assert module.forbidden_input_names(names) == []
    matrix = model_matrix(original, names)
    assert matrix.shape == (2, len(names))
    assert not np.array_equal(matrix[:, 0], original["wind_speed_kt"].to_numpy())


def test_label_names_and_derived_names_cannot_be_inputs(tmp_path: Path) -> None:
    derived = [
        "wind_speed_kt",
        "r34_nmi",
        "rmw_nmi",
        "imd5_class",
        "intensity_bin",
        "delta_wind_speed_kt",
        "prev_r34_nmi",
        "log_rmw_nmi",
        "imd5_class_CS",
        "D_DD",
        "CS",
        "SCS",
        "VSCS",
        "ESCS_SUCS",
        "wind",
        "class",
        "prev_wind",
        "class_D_DD",
        "intensity",
    ]
    for module_name in SPLIT_MODULES:
        module = _splits(module_name)
        assert module.forbidden_input_names(derived) == sorted(set(derived))
        assert module.forbidden_input_names(FeatureSpec().feature_names()) == []
        with pytest.raises(module.LeakageError, match="forbidden"):
            module.assert_no_forbidden_inputs(derived, "forbidden features in model matrix")
    frame = pd.DataFrame({"delta_wind_speed_kt": [1.0, 2.0], "pixel_mean": [3.0, 4.0]})
    with pytest.raises(Exception, match="forbidden"):
        model_matrix(frame, ["delta_wind_speed_kt"])
    assert model_matrix(frame, ["pixel_mean"]).shape == (2, 1)
    train = pd.DataFrame({"split": ["train"], "pixel_mean": [1.0], "delta_r34_nmi": [2.0]})
    with pytest.raises(Exception, match="forbidden"):
        select_features(
            train,
            ["delta_r34_nmi"],
            np.array([1.0]),
            source_split="train",
            method="none",
        )
    assert_model_inputs(("image",))
    for name in ("wind_speed_kt", "r34_nmi", "rmw_nmi", "imd5_class", "intensity_bin", "prev_wind"):
        with pytest.raises(Exception, match=name):
            assert_model_inputs(["image", name])
    protocol = pd.DataFrame(
        {
            "split": ["train"],
            "storm_id": ["s"],
            "wind_speed_kt": [40.0],
            "imd5_class": ["CS"],
            "sample_id": ["a"],
            "is_synthetic": [True],
        }
    )
    with pytest.raises(Exception, match="log_rmw_nmi"):
        run_baselines(protocol, ["log_rmw_nmi"], {}, tmp_path)


def _tensor_frame(root: Path) -> tuple[pd.DataFrame, Path]:
    rows = []
    for index in range(4):
        rows.append(
            {
                "sample_id": f"tr{index}",
                "image_path": _write_constant(root, f"tr{index}", 10.0),
                "storm_id": "T0",
                "split": "train",
                "timestamp": f"2020-01-01T0{index}:00:00Z",
                "wind_speed_kt": 77.0 + index,
                "r34_nmi": 55.0 + index,
                "rmw_nmi": 3.0,
                "imd5_class": "CS",
                "intensity_bin": 1,
                "_class": "CS",
                "hemisphere": "SH",
                "wind_averaging": "synthetic",
                "is_synthetic": True,
            }
        )
    for sample_id, storm_id, split, pixel in (
        ("va0", "V0", "val", 100.0),
        ("te0", "X0", "test", 80.0),
    ):
        rows.append(
            {
                "sample_id": sample_id,
                "image_path": _write_constant(root, sample_id, pixel),
                "storm_id": storm_id,
                "split": split,
                "timestamp": "2020-02-01T00:00:00Z",
                "wind_speed_kt": 120.0,
                "r34_nmi": 90.0,
                "rmw_nmi": 15.0,
                "imd5_class": "ESCS_SUCS",
                "intensity_bin": 4,
                "_class": "ESCS_SUCS",
                "wind_averaging": "synthetic",
                "is_synthetic": True,
            }
        )
    frame = pd.DataFrame(rows)
    manifest = root / "manifest.csv"
    return frame, manifest


def test_dl_tensors_exclude_wind_size_and_class(tmp_path: Path) -> None:
    frame, manifest = _tensor_frame(tmp_path)
    images, winds, labels = stack_split(frame, manifest, "train", IMAGE_SIZE, None, 0.30)
    assert images.shape == (4, 1, IMAGE_SIZE, IMAGE_SIZE)
    assert set(np.unique(images).tolist()) == {10.0}
    assert not np.isin([77.0, 55.0, 100.0, 80.0], images).any()

    shifted = frame.copy()
    shifted["wind_speed_kt"] = shifted["wind_speed_kt"] + 25
    shifted["r34_nmi"] = shifted["r34_nmi"] + 10
    shifted["rmw_nmi"] = shifted["rmw_nmi"] + 4
    shifted["imd5_class"] = "D_DD"
    shifted["intensity_bin"] = 0
    shifted["_class"] = "D_DD"
    images_again, winds_again, labels_again = stack_split(
        shifted, manifest, "train", IMAGE_SIZE, None, 0.30
    )
    assert np.allclose(images, images_again)
    assert not np.allclose(winds, winds_again)
    assert labels_again != labels

    windows, window_winds, window_labels, size = stack_windows(
        frame, manifest, "train", IMAGE_SIZE, None, 0.30
    )
    assert windows.shape == (1, 4, 1, IMAGE_SIZE, IMAGE_SIZE)
    assert set(np.unique(windows).tolist()) == {10.0}
    assert size is not None
    assert float(size[0, 0]) == pytest.approx(58.0)
    assert float(size[0, 1]) == pytest.approx(3.0)
    windows_again, winds_again, labels_again, size_again = stack_windows(
        shifted, manifest, "train", IMAGE_SIZE, None, 0.30
    )
    assert np.allclose(windows, windows_again)
    assert not np.allclose(window_winds, winds_again)
    assert labels_again != window_labels
    assert not np.allclose(size, size_again)
