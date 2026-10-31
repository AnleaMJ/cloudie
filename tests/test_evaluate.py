"""Held-out intensity evaluation. Fixtures are synthetic arrays, not cyclones."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image

from src.baselines import baseline_params, fit_numeric_pipeline
from src.config import load_config
from src.evaluate import (
    BINNING_LABEL,
    assign_bins,
    evaluate,
    interval_status,
    main,
    mc_dropout_coverage,
    mc_dropout_intervals,
    persistence_predictions,
    residual_quantile_coverage,
    score_winds,
)
from src.metrics import mean_baseline, regression_metrics
from src.schema import DEFAULT_EDGES_KT, DEFAULT_NAMES, SchemaError
from src.splits import LeakageError

ROOT = Path(__file__).resolve().parents[1]
NATIVE_SIZES = (128, 201, 366)


def test_module_does_not_import_torch() -> None:
    source = (ROOT / "src" / "evaluate.py").read_text(encoding="utf-8")
    assert BINNING_LABEL in source
    assert "official IMD classes" not in source
    for line in source.splitlines():
        stripped = line.strip()
        assert not stripped.startswith("import torch")
        assert not stripped.startswith("from torch")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules.pop('torch', None); import src.evaluate; "
            "raise SystemExit(0 if 'torch' not in sys.modules else 1)",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_bins_follow_our_grouping_and_keep_empty_highlights() -> None:
    winds = [16, 17, 33, 34, 47, 48, 63, 64, 90, 91, 140]
    labels = assign_bins(winds)
    assert labels.tolist() == [
        None,
        "D_DD",
        "D_DD",
        "CS",
        "CS",
        "SCS",
        "SCS",
        "VSCS",
        "VSCS",
        "ESCS_SUCS",
        "ESCS_SUCS",
    ]
    scored = score_winds([20.0, 20.0], [21.0, 23.0])
    assert scored["overall"]["mae_kt"] == pytest.approx(2.0)
    assert scored["overall"]["bias_kt"] == pytest.approx(2.0)
    assert scored["by_bin"]["D_DD"]["n"] == pytest.approx(2.0)
    assert scored["by_bin"]["VSCS"]["n"] == pytest.approx(0.0)
    assert scored["by_bin"]["VSCS"]["mae_kt"] is None
    assert scored["highlight"]["VSCS"]["display_name"] == "VSCS"
    assert scored["highlight"]["ESCS_SUCS"]["display_name"] == "ESCS/SuCS"
    assert scored["highlight"]["ESCS_SUCS"]["mae_kt"] is None
    assert scored["n_below_lowest_edge"] == 0
    below = score_winds([10.0], [12.0], DEFAULT_EDGES_KT, DEFAULT_NAMES)
    assert below["n_below_lowest_edge"] == 1
    assert below["by_bin"]["ESCS_SUCS"]["rmse_kt"] is None


def test_interval_status_uses_a_strict_gap() -> None:
    assert interval_status(0.8, 0.8, 0.0) == "within_tolerance"
    assert interval_status(1.0, 0.8, 0.2) == "within_tolerance"
    assert interval_status(1.0, 0.8, 0.05) == "uncalibrated"
    assert interval_status(0.5, 0.9, 0.0) == "uncalibrated"
    with pytest.raises(ValueError, match="tolerance"):
        interval_status(0.8, 0.8, -0.01)


def test_mc_dropout_coverage_reads_samples_only() -> None:
    truth = np.array([0.0, 0.0, 10.0, 10.0])
    draws = np.zeros((6, 4), dtype=np.float64)
    draws[:, 2:] = 10.0
    lower, upper = mc_dropout_intervals(draws, 0.8)
    assert lower.tolist() == [0.0, 0.0, 10.0, 10.0]
    assert upper.tolist() == [0.0, 0.0, 10.0, 10.0]
    covered = mc_dropout_coverage(truth, draws, levels=(0.8, 0.9), tolerance=0.05)
    assert covered["0.80"]["empirical_coverage"] == pytest.approx(1.0)
    assert covered["0.80"]["status"] == "uncalibrated"
    wide = mc_dropout_coverage(truth, draws, levels=(0.8,), tolerance=0.25)
    assert wide["0.80"]["status"] == "within_tolerance"
    outside = np.full(4, 50.0)
    missed = mc_dropout_coverage(outside, draws, levels=(0.9,), tolerance=0.05)
    assert missed["0.90"]["empirical_coverage"] == pytest.approx(0.0)
    assert missed["0.90"]["status"] == "uncalibrated"
    swapped = np.zeros((4, 6))
    with pytest.raises(ValueError, match="axis 0"):
        mc_dropout_coverage(truth, swapped)
    ragged = np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]])
    level = 0.5
    expected_lo = np.quantile(ragged, 0.25, axis=0, method="linear")
    expected_hi = np.quantile(ragged, 0.75, axis=0, method="linear")
    lo, hi = mc_dropout_intervals(ragged, level)
    assert lo == pytest.approx(expected_lo)
    assert hi == pytest.approx(expected_hi)


def test_residual_quantile_fits_on_validation_and_can_be_uncalibrated() -> None:
    val_truth = np.zeros(4)
    val_pred = np.array([0.0, 2.0, 4.0, 10.0])
    test_truth = np.zeros(4)
    test_pred = np.array([5.0, 2.0, 10.0, 10.0])
    report = residual_quantile_coverage(
        val_truth, val_pred, test_truth, test_pred, levels=(0.8, 0.9), tolerance=0.05
    )
    assert report["0.80"]["absolute_residual_quantile_kt"] == pytest.approx(
        float(np.quantile(np.abs(val_pred - val_truth), 0.8, method="linear"))
    )
    assert report["0.80"]["empirical_coverage"] == pytest.approx(0.5)
    assert report["0.80"]["status"] == "uncalibrated"
    assert report["0.80"]["fit_split"] == "val"
    assert report["0.80"]["eval_split"] == "test"
    shifted = residual_quantile_coverage(
        val_truth,
        val_pred,
        test_truth,
        np.zeros(4),
        levels=(0.8,),
        tolerance=0.05,
    )
    assert shifted["0.80"]["absolute_residual_quantile_kt"] == pytest.approx(
        report["0.80"]["absolute_residual_quantile_kt"]
    )
    assert shifted["0.80"]["empirical_coverage"] == pytest.approx(1.0)
    inside = residual_quantile_coverage(
        np.zeros(5),
        np.ones(5),
        np.zeros(10),
        np.array([1.0] * 8 + [3.0, 3.0]),
        levels=(0.8,),
        tolerance=0.05,
    )
    assert inside["0.80"]["empirical_coverage"] == pytest.approx(0.8)
    assert inside["0.80"]["status"] == "within_tolerance"


def test_persistence_uses_the_previous_wind_of_the_same_storm() -> None:
    frame = pd.DataFrame(
        {
            "storm_id": ["s1", "s2", "s1", "s2"],
            "wind_speed_kt": [10.0, 99.0, 12.0, 98.0],
            "timestamp": [
                "2020-01-01T00:00:00Z",
                "2020-01-01T00:00:00Z",
                "2020-01-01T03:00:00Z",
                "2020-01-01T03:00:00Z",
            ],
        }
    )
    predicted = persistence_predictions(frame)
    assert np.isnan(predicted[0])
    assert np.isnan(predicted[1])
    assert predicted[2] == pytest.approx(10.0)
    assert predicted[3] == pytest.approx(99.0)
    flipped = frame.iloc[::-1].reset_index(drop=True)
    reversed_pred = persistence_predictions(flipped)
    assert reversed_pred[0] == pytest.approx(99.0)
    assert reversed_pred[1] == pytest.approx(10.0)
    assert np.isnan(reversed_pred[2])
    assert np.isnan(reversed_pred[3])
    no_time = pd.DataFrame({"storm_id": ["s1", "s1"], "wind_speed_kt": [10.0, 12.0]})
    by_row = persistence_predictions(no_time)
    assert np.isnan(by_row[0])
    assert by_row[1] == pytest.approx(10.0)


def _config(size: int) -> dict:
    """Synthetic held-out case built from ``config/config.yaml``.

    ``baseline_params`` rejects a partial ``baselines`` block. The case
    overrides chip size and seeds, and keeps a single-frame CNN so the
    fixture images are not read as sequences.
    """
    config = load_config(ROOT / "config" / "config.yaml")
    config["project"]["seed"] = 0
    config["data"]["image_size"] = size
    config["split"]["seed"] = 0
    config["baselines"]["seed"] = 0
    config["model"]["name"] = "simple_cnn"
    return config


def _write_image(root: Path, name: str, size: int, value: float) -> str:
    folder = root / "images"
    folder.mkdir(parents=True, exist_ok=True)
    relative = f"images/{name}.npy"
    array = np.full((size, size), value, dtype=np.float32)
    np.save(root / relative, array)
    return relative


def _case(tmp_path: Path, size: int) -> tuple[pd.DataFrame, Path, dict, np.ndarray]:
    """One synthetic storm split. Pixels are constants, not satellite imagery."""
    plan = [
        ("T1", "train", "2020-01-01T00:00:00Z", 30.0, 30.0, 0.0, 1.0),
        ("T1", "train", "2020-01-01T03:00:00Z", 32.0, 32.0, 1.0, 5.0),
        ("T2", "train", "2020-01-02T00:00:00Z", 40.0, 40.0, 2.0, 1.0),
        ("T2", "train", "2020-01-02T03:00:00Z", 42.0, 42.0, 3.0, 5.0),
        ("V1", "val", "2020-02-01T00:00:00Z", 50.0, 50.0, 4.0, 80.0),
        ("V1", "val", "2020-02-01T03:00:00Z", 55.0, 65.0, 5.0, 80.0),
        ("X1", "test", "2020-03-01T00:00:00Z", 90.0, 95.0, 6.0, 90.0),
        ("X1", "test", "2020-03-01T03:00:00Z", 70.0, 75.0, 7.0, 90.0),
        ("X2", "test", "2020-03-02T00:00:00Z", 91.0, 81.0, 8.0, 90.0),
        ("X2", "test", "2020-03-02T03:00:00Z", 110.0, 100.0, 9.0, 90.0),
    ]
    rows = []
    for index, (storm, split, timestamp, wind, pred, feature, pixel) in enumerate(plan):
        sample_id = f"{storm}_{index}"
        rows.append(
            {
                "sample_id": sample_id,
                "image_path": _write_image(tmp_path, sample_id, size, pixel),
                "storm_id": storm,
                "split": split,
                "timestamp": timestamp,
                "wind_speed_kt": wind,
                "y_pred": pred,
                "f1": feature,
                "wind_averaging": "1min",
                "is_synthetic": True,
            }
        )
    frame = pd.DataFrame(rows)
    samples = np.tile(frame.loc[frame["split"] == "test", "y_pred"].to_numpy(dtype=np.float64), (5, 1))
    return frame, tmp_path / "manifest.csv", _config(size), samples


def _required_frame(**overrides) -> pd.DataFrame:
    row = {
        "sample_id": "a",
        "image_path": "images/a.npy",
        "storm_id": "S",
        "wind_speed_kt": 40.0,
        "y_pred": 41.0,
        "f1": 1.0,
        "wind_averaging": "1min",
        "is_synthetic": True,
        "split": "train",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_mixed_wind_averaging_is_refused(tmp_path: Path) -> None:
    frame = pd.concat(
        [
            _required_frame(sample_id="a", storm_id="T", split="train"),
            _required_frame(sample_id="b", storm_id="V", split="val", wind_averaging="3-min"),
            _required_frame(sample_id="c", storm_id="X", split="test"),
        ],
        ignore_index=True,
    )
    with pytest.raises(SchemaError, match="wind_averaging"):
        evaluate(frame, tmp_path / "manifest.csv", _config(8), feature_names=["f1"], out_dir=tmp_path / "out")


def test_random_frame_split_is_rejected(tmp_path: Path) -> None:
    rows = [
        _required_frame(sample_id=f"s{index}", storm_id=f"S{index}").drop(columns=["split"])
        for index in range(3)
    ]
    frame = pd.concat(rows, ignore_index=True)
    config = _config(8)
    config["split"]["strategy"] = "random"
    with pytest.raises(LeakageError, match="random"):
        evaluate(frame, tmp_path / "manifest.csv", config, feature_names=["f1"], out_dir=tmp_path / "out")


def test_frame_level_split_is_rejected(tmp_path: Path) -> None:
    frame = pd.concat(
        [
            _required_frame(sample_id="a", storm_id="S", split="train"),
            _required_frame(sample_id="b", storm_id="S", split="test"),
            _required_frame(sample_id="c", storm_id="V", split="val"),
        ],
        ignore_index=True,
    )
    with pytest.raises(LeakageError):
        evaluate(frame, tmp_path / "manifest.csv", _config(8), feature_names=["f1"], out_dir=tmp_path / "out")


def test_wind_feature_is_forbidden(tmp_path: Path) -> None:
    frame = pd.concat(
        [
            _required_frame(sample_id="t", storm_id="T", split="train"),
            _required_frame(sample_id="v", storm_id="V", split="val"),
            _required_frame(sample_id="x", storm_id="X", split="test", wind_speed_kt=90.0),
        ],
        ignore_index=True,
    )
    with pytest.raises(LeakageError, match="wind_speed_kt"):
        evaluate(
            frame,
            tmp_path / "manifest.csv",
            _config(8),
            feature_names=["wind_speed_kt"],
            out_dir=tmp_path / "out",
        )


def test_train_storms_never_appear_in_the_evaluated_set(tmp_path: Path) -> None:
    frame, manifest, config, samples = _case(tmp_path, 8)
    report = evaluate(
        frame,
        manifest,
        config,
        feature_names=["f1"],
        out_dir=tmp_path / "reports",
        samples=samples,
    )
    from src.splits import storm_ids_by_split

    assert storm_ids_by_split(frame)["test"] == frozenset(report["evaluated_storms"])
    evaluated = set(report["evaluated_storms"])
    train = set(report["train_storms"])
    assert evaluated.isdisjoint(train)
    assert evaluated == {"X1", "X2"}
    assert train == {"T1", "T2"}
    assert set(report["val_storms"]) == {"V1"}
    assert set(report["baselines"]["fitted_storms"]) == train
    assert evaluated.isdisjoint(report["baselines"]["fitted_storms"])
    assert report["binning"] == "our own grouping, IMD-style names"
    assert report["reported_especially"] == ["VSCS", "ESCS/SuCS"]
    assert report["wind_averaging"] == "1min"
    assert "3-minute" in report["wind_averaging_note"]
    assert "1-minute" in report["wind_averaging_note"]
    assert "official IMD classes" not in json.dumps(report)
    assert report["quote_as_result"] == "no"
    assert report["present_as_forecast"] is False
    assert report["synthetic"] == "yes"
    assert report["normalization"]["source_split"] == "train"
    assert report["normalization"]["mean"] == pytest.approx(3.0)
    assert report["normalization"]["std"] == pytest.approx(2.0)
    assert report["edges_kt"] == [17.0, 34.0, 48.0, 64.0, 91.0]

    model = report["model"]
    assert model["overall"]["mae_kt"] == pytest.approx(7.5)
    assert model["overall"]["bias_kt"] == pytest.approx(-2.5)
    assert model["highlight"]["VSCS"]["mae_kt"] == pytest.approx(5.0)
    assert model["highlight"]["VSCS"]["bias_kt"] == pytest.approx(5.0)
    assert model["highlight"]["ESCS_SUCS"]["display_name"] == "ESCS/SuCS"
    assert model["highlight"]["ESCS_SUCS"]["mae_kt"] == pytest.approx(10.0)
    assert model["highlight"]["ESCS_SUCS"]["bias_kt"] == pytest.approx(-10.0)
    assert model["by_bin"]["D_DD"]["n"] == pytest.approx(0.0)
    assert model["by_bin"]["D_DD"]["mae_kt"] is None

    train_winds = frame.loc[frame["split"] == "train", "wind_speed_kt"]
    test_winds = frame.loc[frame["split"] == "test", "wind_speed_kt"]
    expected_mean = regression_metrics(test_winds, mean_baseline(train_winds, len(test_winds)))
    assert report["baselines"]["mean"]["overall"]["mae_kt"] == pytest.approx(expected_mean["mae_kt"])
    assert report["baselines"]["mean"]["overall"]["bias_kt"] == pytest.approx(expected_mean["bias_kt"])
    fitted: set[str] = set()
    ridge_hat, _pipeline, note = fit_numeric_pipeline(
        frame.loc[frame["split"] == "train", ["f1"]].to_numpy(dtype=np.float64),
        train_winds.to_numpy(dtype=np.float64),
        frame.loc[frame["split"] == "test", ["f1"]].to_numpy(dtype=np.float64),
        task="regression",
        model_name="ridge",
        params=baseline_params(config),
        seed=0,
        train_storms=frame.loc[frame["split"] == "train", "storm_id"].astype(str),
        fitted_storms=fitted,
    )
    assert note == "no_smote"
    assert fitted <= {"T1", "T2"}
    expected_ridge = regression_metrics(test_winds, ridge_hat)
    assert report["baselines"]["ridge"]["overall"]["mae_kt"] == pytest.approx(expected_ridge["mae_kt"])
    assert report["baselines"]["ridge_fit_notes"] == "no_smote"
    persistence = report["baselines"]["persistence"]
    assert persistence["overall"]["n"] == pytest.approx(2.0)
    assert persistence["overall"]["mae_kt"] == pytest.approx(19.5)
    assert persistence["overall"]["bias_kt"] == pytest.approx(0.5)
    assert set(persistence["storms"]) == {"X1", "X2"}
    paired = report["baselines"]["paired"]
    assert paired["model"]["overall"]["n"] == pytest.approx(2.0)
    assert paired["persistence"]["overall"]["mae_kt"] == pytest.approx(19.5)

    residual = report["uncertainty"]["residual_quantile"]["levels"]["0.80"]
    val_abs = np.abs(
        frame.loc[frame["split"] == "val", "y_pred"].to_numpy()
        - frame.loc[frame["split"] == "val", "wind_speed_kt"].to_numpy()
    )
    assert residual["absolute_residual_quantile_kt"] == pytest.approx(
        float(np.quantile(val_abs, 0.8, method="linear"))
    )
    assert residual["status"] == "uncalibrated"
    assert residual["empirical_coverage"] == pytest.approx(0.5)
    mc = report["uncertainty"]["mc_dropout"]["levels"]["0.80"]
    assert mc["empirical_coverage"] == pytest.approx(0.0)
    assert mc["status"] == "uncalibrated"
    assert report["uncertainty"]["mc_dropout"]["imports_torch"] is False

    loaded = json.loads((tmp_path / "reports" / "metrics.json").read_text(encoding="utf-8"))
    assert loaded["evaluated_storms"] == ["X1", "X2"]
    assert set(loaded["evaluated_storms"]).isdisjoint(loaded["train_storms"])
    assert loaded["wind_averaging"] == "1min"
    table = pd.read_csv(tmp_path / "reports" / "metrics.csv")
    assert set(table.loc[table["status"] == "uncalibrated", "series"]) >= {
        "mc_dropout",
        "residual_quantile",
    }
    assert (table["quote_as_result"] == "no").all()
    readme = (tmp_path / "reports" / "README.txt").read_text(encoding="utf-8").lower()
    assert "do not quote" in readme
    assert "not a forecast" in readme
    for name in ("parity.png", "error_by_bin.png", "coverage.png"):
        with Image.open(tmp_path / "reports" / name) as image:
            assert image.size[0] > 10
            assert image.size[1] > 10
    import matplotlib

    assert matplotlib.get_backend().lower() == "agg"

    np.save(tmp_path / "images" / "V1_4.npy", np.full((8, 8), -40.0, dtype=np.float32))
    np.save(tmp_path / "images" / "X1_6.npy", np.full((8, 8), 250.0, dtype=np.float32))
    again = evaluate(
        frame,
        manifest,
        config,
        feature_names=["f1"],
        out_dir=tmp_path / "reports_again",
        samples=samples,
    )
    assert again["normalization"]["mean"] == pytest.approx(report["normalization"]["mean"])
    assert again["normalization"]["std"] == pytest.approx(report["normalization"]["std"])
    assert set(again["evaluated_storms"]).isdisjoint(again["train_storms"])


@pytest.mark.parametrize("size", NATIVE_SIZES)
def test_native_image_sizes(tmp_path: Path, size: int) -> None:
    frame, manifest, config, _samples = _case(tmp_path, size)
    assert int(config["data"]["image_size"]) == size
    for relative in frame["image_path"]:
        assert np.load(tmp_path / relative).shape == (size, size)
    report = evaluate(
        frame,
        manifest,
        config,
        feature_names=["f1"],
        out_dir=tmp_path / "reports",
    )
    assert report["normalization"]["source_split"] == "train"
    assert report["normalization"]["mean"] == pytest.approx(3.0)
    assert report["normalization"]["std"] == pytest.approx(2.0)
    assert set(report["evaluated_storms"]).isdisjoint(report["train_storms"])
    assert set(report["evaluated_storms"]) == {"X1", "X2"}
    assert report["wind_averaging"] == "1min"
    assert (tmp_path / "reports" / "coverage.png").is_file()


def test_cli_writes_a_report(tmp_path: Path) -> None:
    frame, manifest, config, samples = _case(tmp_path, 8)
    frame_path = tmp_path / "frame.csv"
    frame.to_csv(frame_path, index=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    sample_path = tmp_path / "samples.npy"
    np.save(sample_path, samples)
    report = main(
        [
            "--config",
            str(config_path),
            "--frame",
            str(frame_path),
            "--manifest",
            str(manifest),
            "--features",
            "f1",
            "--out",
            str(tmp_path / "cli"),
            "--samples",
            str(sample_path),
            "--coverage-tolerance",
            "0.05",
        ]
    )
    assert report["evaluated_storms"] == ["X1", "X2"]
    assert set(report["train_storms"]).isdisjoint(report["evaluated_storms"])
    assert Path(report["outputs"]["metrics_json"]).is_file()
    assert Path(report["outputs"]["parity"]).is_file()
