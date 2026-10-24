"""Leak-free classical baselines. Smoke metrics are a software check."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.baselines import (
    BaselineDataError,
    fit_numeric_pipeline,
    iter_stratified_group_folds,
    resample_training_fold,
    run_baselines,
    xgboost_available,
)
from src.config import load_config
from src.data_loader import load_manifest
from src.features import FeatureSpec, extract_feature_table
from src.make_smoke_data import write_balanced_smoke
from src.metrics import regression_metrics
from src.schema import FORBIDDEN_FEATURES
from src.splits import LeakageError, assign_storms

ROOT = Path(__file__).resolve().parents[1]


def _params(**overrides):
    config = {
        "baselines": {
            "n_splits": 2,
            "seed": 0,
            "use_smote": False,
            "ridge_alpha": 1.0,
            "knn_neighbors": 3,
            "svr_c": 1.0,
            "svc_c": 1.0,
            "rf_estimators": 10,
            "rf_max_depth": 3,
            "xgb_estimators": 10,
            "xgb_max_depth": 2,
            "xgb_learning_rate": 0.1,
            "max_iter": 300,
            "regression_models": ["mean", "ridge"],
            "classification_models": ["most_frequent", "logreg"],
        },
        "features": {
            "selection": {
                "correlation_threshold": 0.95,
                "max_features": 5,
                "method": "correlation_rf",
                "n_estimators": 12,
                "seed": 0,
            }
        },
    }
    config["baselines"].update(overrides)
    return config


def _protocol_frame() -> pd.DataFrame:
    rows = []
    plan = [
        ("t1", "train", 30.0, "D_DD"),
        ("t2", "train", 31.0, "D_DD"),
        ("t3", "train", 40.0, "CS"),
        ("t4", "train", 41.0, "CS"),
        ("v1", "val", 32.0, "D_DD"),
        ("v2", "val", 42.0, "CS"),
        ("x1", "test", 55.0, "SCS"),
    ]
    for storm, split, wind, klass in plan:
        for time_index in range(4):
            rows.append(
                {
                    "sample_id": f"{storm}_{time_index}",
                    "storm_id": storm,
                    "split": split,
                    "wind_speed_kt": wind + time_index * 0.1,
                    "imd5_class": klass,
                    "is_synthetic": True,
                    "f1": wind + time_index * 0.1,
                    "f2": 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_group_folds_keep_storms_intact() -> None:
    labels = np.array(["A"] * 4 + ["B"] * 4)
    groups = np.array(["a1", "a1", "a2", "a2", "b1", "b1", "b2", "b2"])
    folds = list(iter_stratified_group_folds(labels, groups, n_splits=2, seed=0))
    assert len(folds) == 2
    for train_index, val_index in folds:
        assert set(groups[train_index]).isdisjoint(set(groups[val_index]))
    again = list(iter_stratified_group_folds(labels, groups, n_splits=2, seed=0))
    assert [index.tolist() for pair in folds for index in pair] == [
        index.tolist() for pair in again for index in pair
    ]


def test_group_folds_refuse_rare_classes() -> None:
    with pytest.raises(BaselineDataError, match="n_splits"):
        list(iter_stratified_group_folds(["A", "A", "B"], ["s1", "s1", "s2"], n_splits=2, seed=0))


def test_scaler_is_fit_on_train_only() -> None:
    X_train = np.zeros((4, 1))
    X_val = np.full((3, 1), 100.0)
    target = np.array([10.0, 10.0, 12.0, 14.0])
    fitted: set[str] = set()
    predicted, pipeline, note = fit_numeric_pipeline(
        X_train,
        target,
        X_val,
        task="regression",
        model_name="mean",
        params=_params()["baselines"],
        seed=0,
        train_storms=["s", "s", "s", "s"],
        fitted_storms=fitted,
    )
    assert note == "no_smote"
    assert pipeline.named_steps["scaler"].mean_[0] == pytest.approx(0.0)
    assert predicted == pytest.approx(np.full(3, target.mean()))
    assert fitted == {"s"}


def test_smote_does_not_change_validation_length() -> None:
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(12, 3))
    y_train = np.array(["A"] * 8 + ["B"] * 4)
    X_val = rng.normal(size=(5, 3))
    params = _params(use_smote=True)["baselines"]
    predicted, _pipeline, note = fit_numeric_pipeline(
        X_train,
        y_train,
        X_val,
        task="classification",
        model_name="logreg",
        params=params,
        seed=0,
        train_storms=["t"] * 12,
        fitted_storms=set(),
    )
    assert note == "smote"
    assert len(predicted) == len(X_val)
    resampled_x, resampled_y, how = resample_training_fold(X_train, y_train, seed=0, enabled=True)
    assert how == "smote"
    assert len(resampled_y) > len(y_train)
    assert len(resampled_x) == len(resampled_y)


def test_regression_does_not_resample() -> None:
    predicted, _pipeline, note = fit_numeric_pipeline(
        np.zeros((6, 2)),
        np.arange(6, dtype=float),
        np.ones((2, 2)),
        task="regression",
        model_name="ridge",
        params=_params(use_smote=True)["baselines"],
        seed=0,
        train_storms=["s"] * 6,
        fitted_storms=set(),
    )
    assert note == "no_smote"
    assert len(predicted) == 2


def test_protocol_locks_the_test_split(tmp_path: Path) -> None:
    frame = _protocol_frame()
    report = run_baselines(frame, ["f1", "f2"], _params(), tmp_path)
    assert report.fitted_storms == report.train_storms
    assert report.fitted_storms.isdisjoint(report.val_storms)
    assert report.fitted_storms.isdisjoint(report.test_storms)
    assert report.test_storms == frozenset({"x1"})
    assert report.n_rows_used_for_selection == int((frame["split"] == "train").sum())
    assert set(report.selected_features).isdisjoint(FORBIDDEN_FEATURES)
    reloaded = pd.read_csv(tmp_path / "baseline_comparison.csv")
    assert reloaded["adt_mae_kt"].isna().all()
    assert reloaded["satcon_mae_kt"].isna().all()
    assert reloaded["dvorak_mae_kt"].isna().all()
    assert (reloaded["quote_as_result"] == "no").all()
    assert (reloaded["synthetic"] == "yes").all()
    banner = (tmp_path / "README.txt").read_text(encoding="utf-8").lower()
    assert "do not quote" in banner
    assert "not a forecast" in banner
    payload = json.loads((tmp_path / "selected_features.json").read_text(encoding="utf-8"))
    assert payload["source_split"] == "train"
    assert payload["quote_as_result"] == "no"
    train = frame[frame["split"] == "train"]
    val = frame[frame["split"] == "val"]
    expected = regression_metrics(
        val["wind_speed_kt"], np.full(len(val), train["wind_speed_kt"].mean())
    )
    mean_row = reloaded[(reloaded["task"] == "regression") & (reloaded["model"] == "mean")].iloc[0]
    assert mean_row["mae_kt"] == pytest.approx(expected["mae_kt"])
    assert mean_row["bias_kt"] == pytest.approx(expected["bias_kt"])
    assert (tmp_path / "scatter_mean.png").stat().st_size > 0
    assert (tmp_path / "confusion_logreg.png").stat().st_size > 0
    assert (tmp_path / "feature_importance.png").stat().st_size > 0


def test_leaked_storm_is_refused(tmp_path: Path) -> None:
    frame = _protocol_frame()
    frame.loc[frame["storm_id"] == "x1", "storm_id"] = "t1"
    frame.loc[frame["sample_id"].str.startswith("x1"), "split"] = "test"
    with pytest.raises(LeakageError):
        run_baselines(frame, ["f1", "f2"], _params(), tmp_path)


def test_missing_xgboost_is_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.baselines.xgboost_available", lambda: False)
    config = _params(
        regression_models=["mean", "xgboost"],
        classification_models=["most_frequent"],
    )
    report = run_baselines(_protocol_frame(), ["f1", "f2"], config, tmp_path)
    skipped = report.comparison[
        (report.comparison["model"] == "xgboost") & (report.comparison["task"] == "regression")
    ].iloc[0]
    assert skipped["fit_notes"] == "not_installed"
    assert pd.isna(skipped["mae_kt"])
    assert skipped["quote_as_result"] == "no"


def test_smoke_images_run_the_configured_baselines(tmp_path: Path) -> None:
    manifest = write_balanced_smoke(tmp_path / "smoke", storms_per_class=5, size=32, seed=7)
    config = load_config(ROOT / "config" / "config_smoke.yaml")
    frame = load_manifest(manifest)
    split = assign_storms(
        frame,
        strategy=config["split"]["strategy"],
        test_fraction=float(config["split"]["test_fraction"]),
        val_fraction=float(config["split"]["val_fraction"]),
        seed=int(config["split"]["seed"]),
    )
    spec = FeatureSpec.from_config(config)
    table, dropped = extract_feature_table(
        split,
        manifest,
        spec,
        max_missing_fraction=float(config["data"]["max_missing_fraction"]),
    )
    assert dropped == 0
    assert set(spec.feature_names()).isdisjoint(FORBIDDEN_FEATURES)
    assert (table.loc[table["split"] == "train", "has_prev"] == 1).any()
    report = run_baselines(table, spec.feature_names(), config, tmp_path / "out")
    assert report.fitted_storms <= report.train_storms
    assert report.fitted_storms.isdisjoint(report.val_storms | report.test_storms)
    comparison = report.comparison
    assert comparison["quote_as_result"].eq("no").all()
    assert comparison["adt_mae_kt"].isna().all()
    regression = comparison[comparison["task"] == "regression"]
    assert regression["mae_kt"].notna().all()
    assert np.isfinite(regression["mae_kt"]).all()
    classification = comparison[comparison["task"] == "classification"]
    scored = classification[classification["fit_notes"] != "not_installed"]
    assert scored["macro_f1"].notna().all()
    if xgboost_available():
        assert "xgboost" in set(regression["model"])
        assert "xgboost" in set(classification["model"])
    models = set(regression["model"])
    for name in ("mean", "ridge", "knn", "svr", "rf"):
        assert name in models
