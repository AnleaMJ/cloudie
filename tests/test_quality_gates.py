"""Fail-closed manifest gates. Bad rows raise; they are not warnings."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.baselines import _synthetic_flag
from src.features import FeatureSpec, extract_feature_table
from src.preprocessing import prepare_image
from src.schema import REQUIRED_COLUMNS, SchemaError, parse_is_synthetic, require_valid

ROOT = Path(__file__).resolve().parents[1]


def _frame(rows: list[dict] | None = None, **overrides: object) -> pd.DataFrame:
    base = {
        "sample_id": "a",
        "image_path": "a.npy",
        "storm_id": "s1",
        "wind_speed_kt": 40.0,
        "wind_averaging": "3-min",
        "is_synthetic": False,
    }
    if rows is None:
        base.update(overrides)
        return pd.DataFrame([base])
    return pd.DataFrame(rows)


def _pair(second: dict) -> pd.DataFrame:
    first = {
        "sample_id": "a",
        "image_path": "a.npy",
        "storm_id": "s1",
        "wind_speed_kt": 40.0,
        "wind_averaging": "3-min",
        "is_synthetic": False,
    }
    second = {**first, "sample_id": "b", "image_path": "b.npy", "storm_id": "s2", **second}
    return pd.DataFrame([first, second])


SCHEMAS = [
    pytest.param(require_valid, SchemaError, id="src"),
]


@pytest.mark.parametrize("require,error", SCHEMAS)
def test_required_columns_and_bounds_are_explicit(require, error) -> None:
    assert REQUIRED_COLUMNS == ("sample_id", "image_path", "storm_id", "wind_speed_kt")
    out = require(_frame(wind_speed_kt=0.0))
    assert out["wind_speed_kt"].iloc[0] == 0.0
    out = require(_frame(wind_speed_kt=200.0))
    assert out["imd5_class"].iloc[0] == "ESCS_SUCS"
    low = require(_frame(wind_speed_kt=10.0))
    assert pd.isna(low["imd5_class"].iloc[0])
    for wind in (-0.1, 200.1, 250):
        with pytest.raises(error, match="outside"):
            require(_frame(wind_speed_kt=wind))


@pytest.mark.parametrize("require,error", SCHEMAS)
def test_duplicate_sample_id_raises(require, error) -> None:
    with pytest.raises(error, match="duplicate sample_id"):
        require(_pair({"sample_id": "a"}))
    with pytest.raises(error, match="duplicate sample_id"):
        require(_pair({}).assign(sample_id=["1", 1]))
    with pytest.raises(error, match="duplicate sample_id"):
        require(_pair({}).assign(sample_id=[np.int64(7), "7"]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(error, match="duplicate sample_id"):
            require(_pair({"sample_id": "a"}))


@pytest.mark.parametrize("require,error", SCHEMAS)
def test_wind_averaging_must_be_one_recorded_period(require, error) -> None:
    with pytest.raises(error, match="wind_averaging"):
        require(_frame().drop(columns=["wind_averaging"]))
    with pytest.raises(error, match="wind_averaging"):
        require(_pair({"wind_averaging": "10-min"}))
    with pytest.raises(error, match="nulls"):
        require(_frame(wind_averaging=None))
    with pytest.raises(error, match="nulls"):
        require(_frame(wind_averaging="  "))
    out = require(_frame(wind_averaging="  3-min  "))
    assert set(out["wind_averaging"]) == {"3-min"}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(error, match="mixed wind_averaging"):
            require(_pair({"wind_averaging": "1-min"}))


@pytest.mark.parametrize("require,error", SCHEMAS)
def test_synthetic_rows_stay_explicitly_flagged(require, error) -> None:
    with pytest.raises(error, match="is_synthetic"):
        require(_frame().drop(columns=["is_synthetic"]))
    with pytest.raises(error, match="is_synthetic"):
        require(_frame(is_synthetic=None))
    with pytest.raises(error, match="is_synthetic"):
        require(_frame(is_synthetic="maybe"))
    with pytest.raises(error, match="is_synthetic"):
        require(_frame(is_synthetic=float("nan")))
    flagged = require(_pair({"is_synthetic": "true"}))
    assert flagged["is_synthetic"].tolist() == [False, True]
    assert flagged["is_synthetic"].dtype == bool
    real = require(_frame(is_synthetic="false"))
    assert real["is_synthetic"].tolist() == [False]
    numeric = require(_frame(is_synthetic="1.0"))
    assert numeric["is_synthetic"].tolist() == [True]
    numpy_flag = require(_frame(is_synthetic=np.bool_(True)))
    assert numpy_flag["is_synthetic"].tolist() == [True]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(error, match="is_synthetic"):
            require(_frame(is_synthetic="maybe"))


def test_parse_is_synthetic_does_not_treat_ambiguous_values_as_real() -> None:
    assert parse_is_synthetic(True) is True
    assert parse_is_synthetic(np.bool_(False)) is False
    assert parse_is_synthetic(1) is True
    assert parse_is_synthetic(0) is False
    assert parse_is_synthetic(1.0) is True
    assert parse_is_synthetic("1.0") is True
    assert parse_is_synthetic(" yes ") is True
    assert parse_is_synthetic("false") is False
    for value in (None, float("nan"), "maybe", "", "synthetic"):
        with pytest.raises(SchemaError):
            parse_is_synthetic(value)


def test_manifest_flag_tracks_any_synthetic_row() -> None:
    from cloudie.train import manifest_is_synthetic, project_labels

    mixed = _pair({"is_synthetic": True})
    validated = require_valid(mixed)
    assert manifest_is_synthetic(validated) is True
    assert validated["is_synthetic"].tolist() == [False, True]
    assert manifest_is_synthetic(require_valid(_frame(is_synthetic=False))) is False
    with pytest.raises(SchemaError, match="is_synthetic"):
        manifest_is_synthetic(_frame().drop(columns=["is_synthetic"]))
    with pytest.raises(SchemaError, match="is_synthetic"):
        manifest_is_synthetic(_frame(is_synthetic=None))
    config = {
        "data": {
            "imd5_edges_kt": [17, 34, 48, 64, 91],
            "imd5_names": ["D_DD", "CS", "SCS", "VSCS", "ESCS_SUCS"],
        }
    }
    assert project_labels(require_valid(_frame()), config) == ["CS"]
    with pytest.raises(SchemaError, match="wind_averaging"):
        project_labels(_pair({"wind_averaging": "10-min"}), config)


def test_baseline_tracker_does_not_call_a_synthetic_row_real() -> None:
    assert _synthetic_flag(pd.DataFrame({"is_synthetic": ["true", "true"]})) == "yes"
    assert _synthetic_flag(pd.DataFrame({"is_synthetic": ["false"]})) == "no"
    assert _synthetic_flag(pd.DataFrame({"is_synthetic": [True, False]})) == "mixed"
    assert _synthetic_flag(pd.DataFrame({"wind_speed_kt": [1.0]})) == "unknown"
    with pytest.raises(SchemaError, match="is_synthetic"):
        _synthetic_flag(pd.DataFrame({"is_synthetic": [None]}))
    with pytest.raises(SchemaError, match="is_synthetic"):
        _synthetic_flag(pd.DataFrame({"is_synthetic": ["maybe"]}))


def test_feature_table_keeps_a_string_false_flag(tmp_path: Path) -> None:
    np.save(tmp_path / "a.npy", np.zeros((8, 8), dtype=np.float32))
    frame = _frame(
        sample_id="a",
        image_path="a.npy",
        split="train",
        is_synthetic="false",
        wind_speed_kt=40.0,
    )
    table, _dropped = extract_feature_table(
        frame, str(tmp_path / "manifest.csv"), FeatureSpec()
    )
    assert table["is_synthetic"].tolist() == [False]
    assert table["pixel_mean"].iloc[0] == pytest.approx(0.0)
    assert "tb_mean" not in table.columns


def test_missing_pixel_fraction_above_0_30_raises() -> None:
    image = np.zeros((10, 10), dtype=np.float32)
    image.ravel()[:30] = np.nan
    prepared = prepare_image(image, 8, None, max_missing_fraction=0.30)
    assert prepared.shape == (1, 8, 8)
    assert np.isfinite(prepared).all()
    image.ravel()[:31] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="missing fraction"):
            prepare_image(image, 8, None, max_missing_fraction=0.30)


def test_smoke_manifests_keep_the_synthetic_flag(tmp_path: Path) -> None:
    from src.data_loader import load_manifest
    from src.make_smoke_data import write_smoke

    frame = load_manifest(write_smoke(tmp_path / "smoke"))
    assert set(REQUIRED_COLUMNS).issubset(frame.columns)
    assert set(frame["wind_averaging"]) == {"synthetic"}
    assert frame["is_synthetic"].dtype == bool
    assert bool(frame["is_synthetic"].all())
    assert not frame["sample_id"].duplicated().any()


def test_schema_wording_matches_the_project_rules() -> None:
    for relative in (
        "src/schema.py",
        "README.md",
        "docs/PROJECT_DOCUMENTATION.md",
        "docs/DATA_SOURCES.md",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "not official IMD classes" in text
    text = (ROOT / "src" / "schema.py").read_text(encoding="utf-8")
    assert "pixel stats" in text
    assert "Raw digital numbers are not named" in text
    assert "IMD associated maximum sustained wind" not in text
    doc = (ROOT / "docs" / "PROJECT_DOCUMENTATION.md").read_text(encoding="utf-8")
    assert "Pixel stats" in doc
    assert "Raw digital numbers stay `pixel_*`" in doc
    assert "brightness-temperature statistics" not in doc
