"""Leak-free classical baselines. Smoke metrics are a software check."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LinearRegression

from src.baselines import (
    BaselineDataError,
    fit_numeric_pipeline,
    frame_for_recorded_split,
    frame_for_storm_split,
    iter_stratified_group_folds,
    persistence_class_predictions,
    persistence_predictions,
    resample_training_fold,
    round_robin_group_folds,
    run_baselines,
    run_from_config,
    write_comparison_files,
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
            "bootstrap_resamples": 32,
            "per_storm_breakdown": True,
            "cv_pool": "train",
            "cv_group_column": "storm_id",
            "cv_collapse_rare": False,
            "cv_assign": "stratified",
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
    assert payload["train_storms"] == sorted(report.train_storms)
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
    for name in ("mean", "persistence", "ridge", "lr", "knn", "svr", "rf"):
        assert name in models
    classified = set(classification["model"])
    for name in ("mean", "persistence", "logreg", "knn", "svc", "rf"):
        assert name in classified
    persist = regression[regression["model"] == "persistence"].iloc[0]
    assert persist["mae_kt"] == persist["mae_kt"]
    assert "not_a_feature" in persist["fit_notes"]
    assert "no_pipeline" in persist["fit_notes"]
    assert "reference_baseline" in persist["fit_notes"]
    assert persist["role"] == "reference_baseline"
    class_persist = classification[classification["model"] == "persistence"].iloc[0]
    assert class_persist["macro_f1"] == class_persist["macro_f1"]
    assert (tmp_path / "out" / "baseline_comparison.md").stat().st_size > 0
    assert (tmp_path / "out" / "confusion_persistence.png").stat().st_size > 0
    assert (tmp_path / "out" / "scatter_persistence.png").stat().st_size > 0
    per_class = pd.read_csv(tmp_path / "out" / "per_class_metrics.csv")
    assert set(per_class["split"]) == {"val"}
    assert set(per_class["subset"]) == {"all"}
    assert set(per_class["resample"]) == {"storm_id"}
    assert set(per_class["quote_as_result"]) == {"no"}
    assert set(per_class["wind_averaging"]) == {"synthetic"}
    assert set(per_class["n_resamples"]) == {1000}
    assert "test" not in set(per_class["split"].astype(str))
    small = per_class[
        (per_class["row_kind"] == "class")
        & (per_class["n_storms"] > 0)
        & (per_class["n_storms"] < 3)
    ]
    if len(small):
        assert small["status"].eq("indicative (n_storms<3)").all()
    text = (tmp_path / "out" / "per_class_metrics.md").read_text(encoding="utf-8")
    assert "validation" in text.lower()
    assert "storm" in text.lower()


def test_persistence_follows_relative_time_and_stays_inside_the_storm() -> None:
    frame = pd.DataFrame(
        {
            "storm_id": ["s1", "s1", "s2", "s1"],
            "wind_speed_kt": [30.0, 80.0, 10.0, 40.0],
            "relative_time_s": [0, 20, 0, 10],
        }
    )
    predicted = persistence_predictions(frame)
    assert np.isnan(predicted[0])
    assert predicted[1] == pytest.approx(40.0)
    assert np.isnan(predicted[2])
    assert predicted[3] == pytest.approx(30.0)
    classes = persistence_class_predictions(frame)
    assert classes[0] is None
    assert classes[1] == "CS"
    assert classes[3] == "D_DD"


def test_linear_regression_is_a_pipeline(tmp_path: Path) -> None:
    predicted, pipeline, note = fit_numeric_pipeline(
        np.arange(8, dtype=float).reshape(4, 2),
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.ones((2, 2)),
        task="regression",
        model_name="lr",
        params=_params()["baselines"],
        seed=0,
        train_storms=["s"] * 4,
        fitted_storms=set(),
    )
    assert note == "no_smote"
    assert isinstance(pipeline.named_steps["model"], LinearRegression)
    assert len(predicted) == 2
    assert np.isfinite(predicted).all()


def test_protocol_persistence_matches_the_previous_row(tmp_path: Path) -> None:
    config = _params(
        regression_models=["persistence"],
        classification_models=["persistence"],
    )
    report = run_baselines(_protocol_frame(), ["f1", "f2"], config, tmp_path)
    assert report.fitted_storms == frozenset()
    assert report.fitted_storms.isdisjoint(report.val_storms | report.test_storms)
    regression = report.comparison[
        (report.comparison["task"] == "regression") & (report.comparison["model"] == "persistence")
    ].iloc[0]
    classification = report.comparison[
        (report.comparison["task"] == "classification")
        & (report.comparison["model"] == "persistence")
    ].iloc[0]
    assert regression["mae_kt"] == pytest.approx(0.1)
    assert classification["macro_f1"] == pytest.approx(1.0)
    text = (tmp_path / "baseline_comparison.md").read_text(encoding="utf-8")
    assert "macro_f1" in text
    assert "not an operational-readiness claim" in text
    assert "reference baseline" in text.lower()
    assert "best-track" in text
    assert "not the imagery" in text
    run_note = (tmp_path / "RUN.md").read_text(encoding="utf-8") if (tmp_path / "RUN.md").is_file() else ""
    assert "reference baseline" in text.lower()
    assert "quote_as_result" in text


def test_official_side_is_not_the_modeling_split() -> None:
    rows = []
    for index, storm in enumerate(["a", "b", "c", "d", "e", "f"]):
        rows.append(
            {
                "sample_id": storm,
                "image_path": f"{storm}.npy",
                "storm_id": storm,
                "wind_speed_kt": 30.0 + index * 15.0,
                "split": "train",
                "source_split": "test" if index % 2 else "train",
                "wind_averaging": "1min",
                "is_synthetic": False,
            }
        )
    frame = pd.DataFrame(rows)
    config = load_config(ROOT / "config" / "config.yaml")
    assigned = frame_for_storm_split(frame, config)
    assert list(assigned["source_split"]) == [row["source_split"] for row in rows]
    assert set(assigned["split"]) == {"train", "val", "test"}
    assert not assigned["split"].eq("train").all()
    refused = load_config(ROOT / "config" / "config.yaml")
    refused["split"]["strategy"] = "chronological"
    with pytest.raises(BaselineDataError, match="storm-wise"):
        frame_for_storm_split(frame, refused)


def test_per_class_table_scores_validation_rows_only(tmp_path: Path, monkeypatch) -> None:
    seen: list[set[str]] = []
    from src.evaluate import intensity_class_evaluation as real

    def spy(*args, **kwargs):
        splits = kwargs.get("splits", args[4])
        seen.append(set(np.asarray(splits).astype(str)))
        return real(*args, **kwargs)

    monkeypatch.setattr("src.evaluate.intensity_class_evaluation", spy)
    frame = _protocol_frame()
    run_baselines(frame, ["f1", "f2"], _params(), tmp_path)
    assert seen
    assert all(item == {"val"} for item in seen)
    table = pd.read_csv(tmp_path / "per_class_metrics.csv")
    assert set(table["split"]) == {"val"}
    assert set(table["subset"]) == {"all"}
    assert set(table["resample"]) == {"storm_id"}
    assert set(table["n_resamples"]) == {32}
    assert set(table["seed"]) == {0}
    assert table["ci_level"].to_numpy() == pytest.approx(0.95)
    assert (table["quote_as_result"] == "no").all()
    assert table["wind_averaging"].isna().all()
    train = frame[frame["split"] == "train"]
    val = frame[frame["split"] == "val"]
    expected = regression_metrics(
        val["wind_speed_kt"], np.full(len(val), train["wind_speed_kt"].mean())
    )
    overall = table[
        (table["model"] == "mean")
        & (table["row_kind"] == "overall")
        & (table["intensity_class"] == "overall")
    ].iloc[0]
    assert overall["n_rows"] == len(val)
    assert overall["n_storms"] == val["storm_id"].nunique()
    assert overall["mae_kt"] == pytest.approx(expected["mae_kt"])
    assert overall["rmse_kt"] == pytest.approx(expected["rmse_kt"])
    assert overall["bias_kt"] == pytest.approx(expected["bias_kt"])
    assert overall["status"] == "indicative (n_storms<3)"
    dd = table[
        (table["model"] == "mean")
        & (table["row_kind"] == "class")
        & (table["intensity_class"] == "D_DD")
    ].iloc[0]
    assert dd["n_rows"] == 4
    assert dd["n_storms"] == 1
    assert dd["status"] == "indicative (n_storms<3)"
    storms = table[
        (table["model"] == "mean")
        & (table["row_kind"] == "storm")
        & (table["intensity_class"] == "D_DD")
    ]
    assert list(storms["storm_id"]) == ["v1"]
    empty = table[
        (table["model"] == "mean")
        & (table["row_kind"] == "class")
        & (table["intensity_class"] == "SCS")
    ].iloc[0]
    assert empty["n_rows"] == 0
    assert empty["n_storms"] == 0
    assert pd.isna(empty["rmse_kt"])
    assert pd.isna(empty["status"])
    note = (tmp_path / "per_class_metrics.md").read_text(encoding="utf-8")
    assert "indicative (n_storms<3)" in note
    assert "validation split only" in note.lower()
    comparison = (tmp_path / "baseline_comparison.md").read_text(encoding="utf-8")
    assert "per_class_metrics.csv" in comparison


def test_per_class_groups_wind_averaging_and_on_fix(tmp_path: Path) -> None:
    frame = _protocol_frame()
    frame["wind_averaging"] = "1min"
    frame.loc[frame["storm_id"] == "v2", "wind_averaging"] = "3-min"
    frame["on_fix"] = False
    frame.loc[frame["sample_id"].isin(["v1_0", "v1_1"]), "on_fix"] = True
    config = _params(
        regression_models=["mean"],
        classification_models=["most_frequent"],
        bootstrap_resamples=8,
    )
    run_baselines(frame, ["f1", "f2"], config, tmp_path)
    table = pd.read_csv(tmp_path / "per_class_metrics.csv")
    mean = table[table["model"] == "mean"]
    assert set(mean["wind_averaging"]) == {"1min", "3-min"}
    assert set(mean["subset"]) == {"all", "on_fix"}

    def _count(subset: str, period: str, kind: str, name: str) -> int:
        match = mean[
            (mean["subset"] == subset)
            & (mean["wind_averaging"] == period)
            & (mean["row_kind"] == kind)
            & (mean["intensity_class"] == name)
        ]
        assert len(match) == 1
        return int(match.iloc[0]["n_rows"])

    assert _count("all", "1min", "overall", "overall") == 4
    assert _count("all", "1min", "class", "D_DD") == 4
    assert _count("on_fix", "1min", "class", "D_DD") == 2
    assert _count("on_fix", "1min", "overall", "overall") == 2
    assert _count("all", "3-min", "class", "CS") == 4
    assert _count("on_fix", "3-min", "class", "CS") == 0
    assert _count("all", "3-min", "overall", "overall") == 4


def test_per_storm_breakdown_can_be_turned_off(tmp_path: Path) -> None:
    config = _params(
        regression_models=["mean"],
        classification_models=["most_frequent"],
        bootstrap_resamples=8,
        per_storm_breakdown=False,
    )
    run_baselines(_protocol_frame(), ["f1", "f2"], config, tmp_path)
    table = pd.read_csv(tmp_path / "per_class_metrics.csv")
    assert "storm" not in set(table["row_kind"])
    marked = table[(table["model"] == "mean") & (table["intensity_class"] == "D_DD")]
    assert marked["status"].eq("indicative (n_storms<3)").all()


def test_recorded_split_is_not_redrawn() -> None:
    rows = []
    for storm, split in (("a", "train"), ("b", "train"), ("c", "val"), ("d", "test")):
        rows.append(
            {
                "sample_id": storm,
                "storm_id": storm,
                "split": split,
                "cv_group": "g-" + storm,
                "source_split": "train",
                "wind_speed_kt": 40.0,
            }
        )
    frame = pd.DataFrame(rows)
    kept = frame_for_recorded_split(frame)
    assert list(kept["split"]) == ["train", "train", "val", "test"]
    assert list(kept["source_split"]) == ["train"] * 4


def test_year_holdout_is_refused(tmp_path: Path) -> None:
    config = load_config(ROOT / "config" / "config.yaml")
    config["split"]["strategy"] = "year_holdout_2019"
    frame = pd.DataFrame(
        {
            "sample_id": ["a"],
            "image_path": ["a.npy"],
            "storm_id": ["a"],
            "wind_speed_kt": [30.0],
            "wind_averaging": ["1min"],
            "is_synthetic": [False],
            "split": ["train"],
        }
    )
    with pytest.raises(BaselineDataError, match="year_holdout"):
        frame_for_storm_split(frame, config)
    manifest = tmp_path / "manifest.csv"
    frame.to_csv(manifest, index=False)
    config_path = tmp_path / "config.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(BaselineDataError, match="year_holdout"):
        run_from_config(config_path, manifest, tmp_path / "out")


def test_run_md_states_raw_three_minute_labels(tmp_path: Path) -> None:
    from src.baselines import _empty_metric_row

    row = _empty_metric_row("regression", "mean", "note", "no", 1)
    write_comparison_files(
        tmp_path,
        [row],
        {"wind_averaging": "3-min", "wind_averaging_s": "180"},
    )
    text = (tmp_path / "RUN.md").read_text(encoding="utf-8")
    assert "All metrics use raw 3-minute IMD labels." in text
    assert "wind_averaging_s is 180" in text
    assert "no 1-minute conversion" in text


def test_train_val_cv_keeps_cooccurrence_groups_together(tmp_path: Path, monkeypatch) -> None:
    rows = []
    plan = [
        ("a", "train", "g1", 30.0, "D_DD"),
        ("c", "train", "g2", 32.0, "D_DD"),
        ("d", "train", "g3", 50.0, "CS"),
        ("h", "train", "g6", 52.0, "CS"),
        ("b", "val", "g1", 31.0, "D_DD"),
        ("e", "val", "g4", 51.0, "CS"),
        ("f", "test", "g5", 80.0, "VSCS"),
    ]
    for storm, split, group, wind, klass in plan:
        for time_index in range(3):
            rows.append(
                {
                    "sample_id": f"{storm}_{time_index}",
                    "storm_id": storm,
                    "split": split,
                    "cv_group": group,
                    "wind_speed_kt": wind + time_index,
                    "imd5_class": klass,
                    "is_synthetic": True,
                    "f1": wind + time_index,
                    "f2": 1.0,
                }
            )
    seen_groups: list[np.ndarray] = []
    real = iter_stratified_group_folds

    def spy(labels, groups, n_splits, seed):
        folds = list(real(labels, groups, n_splits=n_splits, seed=seed))
        seen_groups.append(np.asarray(groups))
        for train_index, val_index in folds:
            assert set(groups[train_index]).isdisjoint(set(groups[val_index]))
        return folds

    monkeypatch.setattr("src.baselines.iter_stratified_group_folds", spy)
    fitted_calls: list[set[str]] = []
    real_fit = fit_numeric_pipeline

    def spy_fit(*args, **kwargs):
        fitted_calls.append({str(item) for item in kwargs["train_storms"]})
        return real_fit(*args, **kwargs)

    monkeypatch.setattr("src.baselines.fit_numeric_pipeline", spy_fit)
    config = _params(
        cv_pool="train_val",
        cv_group_column="cv_group",
        cv_collapse_rare=True,
        n_splits=2,
        regression_models=["mean"],
        classification_models=["most_frequent"],
        bootstrap_resamples=8,
        per_storm_breakdown=False,
    )
    report = run_baselines(pd.DataFrame(rows), ["f1", "f2"], config, tmp_path)
    assert seen_groups
    assert report.fitted_storms <= report.train_storms
    assert report.fitted_storms.isdisjoint(report.val_storms | report.test_storms)
    assert all("f" not in call for call in fitted_calls)
    assert report.comparison["fit_notes"].astype(str).str.contains("cv_pool:train_val").all()
    assert report.comparison["fit_notes"].astype(str).str.contains("cv_group:cv_group").all()
    assert set(pd.read_csv(tmp_path / "per_class_metrics.csv")["split"]) == {"val"}


def test_round_robin_cv_folds_file_matches_seed_42(tmp_path: Path, monkeypatch) -> None:
    storms = pd.read_csv(ROOT / "incyde" / "audit" / "storm_split.csv")
    rows = []
    for record in storms.itertuples(index=False):
        for time_index in range(2):
            rows.append(
                {
                    "sample_id": f"{record.storm_id}_{time_index}",
                    "storm_id": record.storm_id,
                    "split": record.proposed_split,
                    "cv_group": str(int(record.cooccurrence_group)),
                    "wind_speed_kt": 40.0 + time_index,
                    "imd5_class": "CS",
                    "is_synthetic": True,
                    "f1": float(time_index),
                    "f2": 1.0,
                }
            )
    frame = pd.DataFrame(rows)
    params = _params(
        cv_pool="train_val",
        cv_group_column="cv_group",
        cv_assign="round_robin",
        n_splits=5,
        seed=42,
    )["baselines"]
    folds = round_robin_group_folds(frame, params)
    assert list(folds.columns) == ["storm_id", "fold"]
    pool = storms.loc[storms["proposed_split"].isin(["train", "val"])]
    groups = sorted({int(value) for value in pool["cooccurrence_group"]})
    order = np.random.default_rng(42).permutation(len(groups))
    shuffled = [groups[int(index)] for index in order]
    expected = {group: index % 5 for index, group in enumerate(shuffled)}
    got = {
        storm_id: int(fold)
        for storm_id, fold in zip(folds["storm_id"], folds["fold"], strict=True)
    }
    for record in pool.itertuples(index=False):
        assert got[record.storm_id] == expected[int(record.cooccurrence_group)]
    test_ids = set(storms.loc[storms["proposed_split"] == "test", "storm_id"])
    assert set(folds["storm_id"]).isdisjoint(test_ids)
    assert len(folds) == len(pool)

    def refuse(*_args, **_kwargs):
        raise AssertionError("round-robin CV must not call StratifiedGroupKFold")

    monkeypatch.setattr("src.baselines.iter_stratified_group_folds", refuse)
    tiny = frame.groupby("storm_id", as_index=False).head(1)
    # Two rows per storm keeps persistence defined. Keep the full tiny table.
    config = _params(
        cv_pool="train_val",
        cv_group_column="cv_group",
        cv_assign="round_robin",
        n_splits=5,
        seed=42,
        regression_models=["mean"],
        classification_models=["most_frequent"],
        bootstrap_resamples=4,
        per_storm_breakdown=False,
        use_smote=False,
    )
    run_baselines(frame, ["f1", "f2"], config, tmp_path)
    written = pd.read_csv(tmp_path / "cv_folds.csv")
    assert list(written.columns) == ["storm_id", "fold"]
    assert written["storm_id"].tolist() == folds["storm_id"].tolist()
    assert written["fold"].tolist() == folds["fold"].tolist()
    assert "StratifiedGroupKFold" not in (tmp_path / "baseline_comparison.csv").read_text(encoding="utf-8")
    del tiny
