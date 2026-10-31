"""Classical intensity baselines on storm-grouped, train-selected features.

``run_from_config`` assigns storms with ``src.splits.assign_storms`` and
accepts only ``storm_wise``. An official competition side stays in
``source_split`` and is never the modeling split. Models are fit on train
storms and scored on val storms. The test split is never passed to a fit call.

Learned models are sklearn ``Pipeline`` objects. Inner ``StratifiedGroupKFold``
groups on ``storm_id`` and is a comparison diagnostic on the training storms
only. It is not a license to tune on the locked test storms.

SMOTE, when enabled, resamples the current training fold and never the
validation rows. Otherwise those classifiers use class weights inside the
fold. Regression does not use SMOTE. Persistence is the previous same-storm
wind. It is not a feature and it is not fit. ADT, SATCON, and Dvorak columns
are left empty until those products are actually joined.
"""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR

from src.config import load_config
from src.data_loader import load_manifest
from src.features import FeatureSpec, extract_feature_table, model_matrix, select_features
from src.metrics import classification_metrics, metrics_by_class, regression_metrics
from src.plots import save_confusion_matrix, save_feature_importance, save_regression_scatter
from src.schema import DEFAULT_NAMES, FORBIDDEN_FEATURES, parse_is_synthetic, wind_to_imd5
from src.splits import (
    LeakageError,
    assert_no_forbidden_inputs,
    assert_no_group_leakage,
    assign_storms,
    observation_order,
)

logger = logging.getLogger(__name__)

BANNER = (
    "Software check only. Do not quote these numbers as experimental results. "
    "Operational baselines (ADT, SATCON, Dvorak) are not joined. "
    "The locked test split was not used for fitting, selection, or SMOTE. "
    "This is not a forecast or a warning."
)
OPERATIONAL_COLUMNS = ("adt_mae_kt", "satcon_mae_kt", "dvorak_mae_kt")
COMPARISON_COLUMNS = (
    "task",
    "model",
    "split_protocol",
    "inner_cv",
    "n_train_rows",
    "n_val_rows",
    "n_train_storms",
    "n_val_storms",
    "n_features",
    "mae_kt",
    "rmse_kt",
    "bias_kt",
    "cv_mae_kt_mean",
    "cv_mae_kt_std",
    *[f"mae_{name}" for name in DEFAULT_NAMES],
    "macro_f1",
    "cv_macro_f1_mean",
    "cv_macro_f1_std",
    *[f"recall_{name}" for name in DEFAULT_NAMES],
    *OPERATIONAL_COLUMNS,
    "fit_notes",
    "quote_as_result",
    "synthetic",
)


class BaselineDataError(ValueError):
    """The frame cannot support a leak-free baseline comparison."""


class MeanRegressor(BaseEstimator, RegressorMixin):
    """Predict the mean of the labels it was fit on."""

    def fit(self, X, y):
        target = np.asarray(y, dtype=np.float64).ravel()
        self.mean_ = float(np.mean(target))
        self.n_features_in_ = int(np.asarray(X).shape[1])
        return self

    def predict(self, X):
        return np.full(np.asarray(X).shape[0], self.mean_, dtype=np.float64)


class MostFrequentClassifier(BaseEstimator, ClassifierMixin):
    """Predict the most common training class. Ties take the sorted label."""

    def fit(self, X, y):
        labels, counts = np.unique(np.asarray(y).ravel(), return_counts=True)
        self.classes_ = labels
        self.constant_ = labels[int(np.argmax(counts))]
        self.n_features_in_ = int(np.asarray(X).shape[1])
        return self

    def predict(self, X):
        return np.full(np.asarray(X).shape[0], self.constant_)


_MAJORITY_CLASSIFIERS = frozenset({"most_frequent", "mean"})
_MARKDOWN_COLUMNS = (
    "task",
    "model",
    "n_train_rows",
    "n_val_rows",
    "n_train_storms",
    "n_val_storms",
    "mae_kt",
    "rmse_kt",
    "bias_kt",
    "macro_f1",
    "fit_notes",
    "quote_as_result",
    "synthetic",
)


def persistence_predictions(frame: pd.DataFrame) -> np.ndarray:
    """Previous ``wind_speed_kt`` inside the same storm.

    Order is ``timestamp`` when that column exists, otherwise
    ``relative_time_s``, otherwise the given row order. The first row of
    each storm is NaN. Wind from another storm is never used. This is a
    baseline, not a model feature.
    """
    if "storm_id" not in frame.columns or "wind_speed_kt" not in frame.columns:
        raise ValueError("persistence needs storm_id and wind_speed_kt")
    winds = pd.to_numeric(frame["wind_speed_kt"], errors="coerce").to_numpy(dtype=np.float64)
    table = pd.DataFrame(
        {
            "storm_id": frame["storm_id"].astype(str).to_numpy(),
            "wind": winds,
            "order": observation_order(frame).to_numpy(),
            "row": np.arange(len(frame)),
        }
    )
    table = table.sort_values(["storm_id", "order", "row"], kind="mergesort")
    table["prediction"] = table.groupby("storm_id", sort=False)["wind"].shift(1)
    table = table.sort_values("row", kind="mergesort")
    return table["prediction"].to_numpy(dtype=np.float64)


def persistence_class_predictions(frame: pd.DataFrame) -> np.ndarray:
    """Previous same-storm wind, mapped with this project's five bins."""
    winds = persistence_predictions(frame)
    classes = np.empty(len(winds), dtype=object)
    for index, value in enumerate(winds):
        if not np.isfinite(value):
            classes[index] = None
        else:
            classes[index] = wind_to_imd5(float(value))
    return classes


def _known_class_mask(truth: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    known = set(DEFAULT_NAMES)
    return np.asarray(
        [item in known and guess in known for item, guess in zip(truth, predicted, strict=True)],
        dtype=bool,
    )


@dataclass(frozen=True)
class BaselineReport:
    comparison: pd.DataFrame
    selected_features: tuple[str, ...]
    train_storms: frozenset[str]
    val_storms: frozenset[str]
    test_storms: frozenset[str]
    fitted_storms: frozenset[str]
    n_rows_used_for_selection: int
    out_dir: Path


def xgboost_available() -> bool:
    try:
        import xgboost  # noqa: F401
    except ImportError:
        return False
    return True


_BASELINE_KEYS = (
    "n_splits",
    "seed",
    "regression_models",
    "classification_models",
    "use_smote",
    "ridge_alpha",
    "knn_neighbors",
    "svr_c",
    "svc_c",
    "rf_estimators",
    "rf_max_depth",
    "xgb_estimators",
    "xgb_max_depth",
    "xgb_learning_rate",
    "max_iter",
)


def baseline_params(config: dict) -> dict:
    """Hyperparameters from ``baselines`` in the YAML config. No fallbacks."""
    raw = config.get("baselines")
    if not isinstance(raw, dict):
        raise KeyError("config baselines is required")
    missing = [key for key in _BASELINE_KEYS if key not in raw]
    if missing:
        raise KeyError(f"config baselines missing {missing}")
    return {key: raw[key] for key in _BASELINE_KEYS}


def selection_params(config: dict) -> dict:
    """Feature-selection hyperparameters from ``features.selection``."""
    raw = config["features"]["selection"]
    return {
        "correlation_threshold": float(raw["correlation_threshold"]),
        "max_features": int(raw["max_features"]),
        "method": str(raw["method"]),
        "n_estimators": int(raw["n_estimators"]),
        "seed": int(raw["seed"]),
    }


def iter_stratified_group_folds(y_strat, groups, n_splits: int, seed: int):
    """Yield train/val indices. A storm never sits on both sides of a fold."""
    labels = np.asarray(y_strat).ravel()
    group_ids = np.asarray(groups).ravel()
    if len(labels) != len(group_ids):
        raise BaselineDataError("stratify labels and groups must align")
    if n_splits < 2:
        raise BaselineDataError("n_splits must be at least 2")
    table = pd.DataFrame({"y": labels.astype(str), "g": group_ids.astype(str)})
    for label, part in table.groupby("y"):
        n_groups = int(part["g"].nunique())
        if n_groups < n_splits:
            raise BaselineDataError(
                f"class {label} is in {n_groups} storms but n_splits={n_splits}"
            )
    from sklearn.model_selection import StratifiedGroupKFold

    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    yield from splitter.split(np.zeros(len(labels)), table["y"], table["g"])


def impute_train_val(X_train: np.ndarray, X_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median imputer fit on the training fold only."""
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train = np.asarray(imputer.fit_transform(X_train), dtype=np.float64)
    val = np.asarray(imputer.transform(X_val), dtype=np.float64)
    return (
        np.where(np.isfinite(train), train, 0.0),
        np.where(np.isfinite(val), val, 0.0),
    )


def resample_training_fold(
    X_train: np.ndarray, y_train: np.ndarray, seed: int, enabled: bool
) -> tuple[np.ndarray, np.ndarray, str]:
    """SMOTE the training fold only. Validation rows are not an argument."""
    if not enabled:
        return X_train, y_train, "class_weight"
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        return X_train, y_train, "class_weight_imblearn_missing"
    labels, counts = np.unique(y_train, return_counts=True)
    if len(labels) < 2:
        return X_train, y_train, "class_weight_single_class"
    minority = int(counts.min())
    if minority < 2:
        return X_train, y_train, "class_weight_minority_lt_2"
    sampler = SMOTE(k_neighbors=min(5, minority - 1), random_state=seed)
    try:
        resampled_x, resampled_y = sampler.fit_resample(X_train, y_train)
    except ValueError:
        return X_train, y_train, "class_weight_smote_failed"
    return np.asarray(resampled_x), np.asarray(resampled_y), "smote"


def _balanced_weights(y: np.ndarray) -> np.ndarray:
    labels, counts = np.unique(y, return_counts=True)
    total = len(y)
    n_class = len(labels)
    table = {
        label: total / (n_class * count) for label, count in zip(labels, counts, strict=True)
    }
    return np.asarray([table[label] for label in y], dtype=np.float64)


def _make_estimator(name: str, task: str, params: dict, class_weight: str | None, n_train: int, seed: int):
    weight = "balanced" if class_weight == "balanced" else None
    neighbors = max(1, min(int(params["knn_neighbors"]), int(n_train)))
    if task == "regression":
        if name == "mean":
            return MeanRegressor()
        if name == "lr":
            return LinearRegression()
        if name == "ridge":
            return Ridge(alpha=float(params["ridge_alpha"]))
        if name == "knn":
            return KNeighborsRegressor(n_neighbors=neighbors)
        if name == "svr":
            return SVR(C=float(params["svr_c"]), kernel="rbf")
        if name == "rf":
            return RandomForestRegressor(
                n_estimators=int(params["rf_estimators"]),
                max_depth=int(params["rf_max_depth"]),
                random_state=seed,
                n_jobs=1,
            )
        if name == "xgboost":
            from xgboost import XGBRegressor

            return XGBRegressor(
                n_estimators=int(params["xgb_estimators"]),
                max_depth=int(params["xgb_max_depth"]),
                learning_rate=float(params["xgb_learning_rate"]),
                random_state=seed,
                n_jobs=1,
                verbosity=0,
                tree_method="hist",
            )
    else:
        if name in _MAJORITY_CLASSIFIERS:
            return MostFrequentClassifier()
        if name == "logreg":
            return LogisticRegression(max_iter=int(params["max_iter"]), class_weight=weight)
        if name == "knn":
            return KNeighborsClassifier(n_neighbors=neighbors)
        if name == "svc":
            return SVC(C=float(params["svc_c"]), kernel="rbf", class_weight=weight)
        if name == "rf":
            return RandomForestClassifier(
                n_estimators=int(params["rf_estimators"]),
                max_depth=int(params["rf_max_depth"]),
                random_state=seed,
                n_jobs=1,
                class_weight=weight,
            )
        if name == "xgboost":
            from xgboost import XGBClassifier

            return XGBClassifier(
                n_estimators=int(params["xgb_estimators"]),
                max_depth=int(params["xgb_max_depth"]),
                learning_rate=float(params["xgb_learning_rate"]),
                random_state=seed,
                n_jobs=1,
                verbosity=0,
                tree_method="hist",
            )
    raise BaselineDataError(f"unknown model {name} for {task}")


def fit_numeric_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    *,
    task: str,
    model_name: str,
    params: dict,
    seed: int,
    train_storms,
    fitted_storms: set[str],
) -> tuple[np.ndarray, Pipeline, str]:
    """Fit on training rows and predict validation rows.

    ``train_storms`` is recorded so a caller can prove val and test storms
    never entered a fit. SMOTE sees only the imputed training matrix.
    """
    fitted_storms.update(str(storm) for storm in np.asarray(train_storms).ravel())
    X_train_imp, X_val_imp = impute_train_val(X_train, X_val)
    note = "no_smote"
    X_fit, y_fit = X_train_imp, np.asarray(y_train).ravel()
    if task == "classification" and model_name not in _MAJORITY_CLASSIFIERS:
        X_fit, y_fit, note = resample_training_fold(
            X_train_imp,
            y_fit,
            seed,
            enabled=bool(params["use_smote"]),
        )
        if model_name == "knn" and note != "smote":
            note = "unweighted_knn"
    elif task == "classification":
        note = "majority_train_class"

    class_weight = None
    if (
        task == "classification"
        and model_name not in _MAJORITY_CLASSIFIERS | {"knn"}
        and note != "smote"
    ):
        class_weight = "balanced"
    estimator = _make_estimator(
        model_name, task, params, class_weight, n_train=len(X_fit), seed=seed
    )
    sample_weight = None
    if model_name == "xgboost" and class_weight == "balanced":
        sample_weight = _balanced_weights(y_fit)
        estimator = _make_estimator(
            model_name, task, params, None, n_train=len(X_fit), seed=seed
        )
        note = f"{note},xgb_sample_weight"
    encoder = None
    y_model = y_fit
    if task == "classification" and model_name == "xgboost":
        encoder = LabelEncoder()
        y_model = encoder.fit_transform(np.asarray(y_fit).ravel())
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
    if sample_weight is None:
        pipeline.fit(X_fit, y_model)
    else:
        pipeline.fit(X_fit, y_model, model__sample_weight=sample_weight)
    raw = np.asarray(pipeline.predict(X_val_imp)).ravel()
    if encoder is not None:
        raw = encoder.inverse_transform(raw.astype(int))
    elif task == "regression":
        raw = raw.astype(np.float64)
    return raw, pipeline, note


def _synthetic_flag(frame: pd.DataFrame) -> str:
    """Track synthetic rows. A missing column is unknown, never a silent no.

    Ambiguous values raise. They are not coerced to real observations.
    """
    if "is_synthetic" not in frame.columns:
        return "unknown"
    flags = [parse_is_synthetic(value) for value in frame["is_synthetic"].tolist()]
    if not flags:
        return "unknown"
    if all(flags):
        return "yes"
    if any(flags):
        return "mixed"
    return "no"


def _empty_metric_row(task: str, model: str, note: str, synthetic: str, n_features: int) -> dict:
    row = {column: np.nan for column in COMPARISON_COLUMNS}
    row.update(
        {
            "task": task,
            "model": model,
            "split_protocol": "train_fit_val_score_test_locked",
            "inner_cv": "StratifiedGroupKFold",
            "n_features": n_features,
            "fit_notes": note,
            "quote_as_result": "no",
            "synthetic": synthetic,
        }
    )
    for column in OPERATIONAL_COLUMNS:
        row[column] = np.nan
    return row


def _cross_val(
    train: pd.DataFrame,
    selected: tuple[str, ...],
    *,
    task: str,
    model_name: str,
    params: dict,
    fitted_storms: set[str],
) -> tuple[float, float, str]:
    matrix = model_matrix(train, selected)
    groups = train["storm_id"].astype(str).to_numpy()
    if task == "regression":
        target = train["wind_speed_kt"].to_numpy(dtype=np.float64)
        stratify = (
            train["imd5_class"]
            .astype(object)
            .where(train["imd5_class"].notna(), "below")
            .astype(str)
            .to_numpy()
        )
    else:
        target = train["imd5_class"].astype(str).to_numpy()
        stratify = target
    folds = list(
        iter_stratified_group_folds(
            stratify, groups, n_splits=int(params["n_splits"]), seed=int(params["seed"])
        )
    )
    scores: list[float] = []
    notes: list[str] = []
    for fold_index, (train_index, val_index) in enumerate(folds):
        predicted, _pipeline, note = fit_numeric_pipeline(
            matrix[train_index],
            target[train_index],
            matrix[val_index],
            task=task,
            model_name=model_name,
            params=params,
            seed=int(params["seed"]) + fold_index + 1,
            train_storms=groups[train_index],
            fitted_storms=fitted_storms,
        )
        notes.append(note)
        if task == "regression":
            scores.append(regression_metrics(target[val_index], predicted)["mae_kt"])
        else:
            scores.append(
                classification_metrics(target[val_index], predicted, DEFAULT_NAMES)["macro_f1"]
            )
    return float(np.mean(scores)), float(np.std(scores)), "inner_cv:" + "|".join(notes)


def _persistence_metric(frame: pd.DataFrame, task: str) -> float | None:
    if frame.empty:
        return None
    if task == "regression":
        predicted = persistence_predictions(frame)
        mask = np.isfinite(predicted)
        if int(mask.sum()) == 0:
            return None
        truth = frame["wind_speed_kt"].to_numpy(dtype=np.float64)[mask]
        return regression_metrics(truth, predicted[mask])["mae_kt"]
    truth = frame["imd5_class"].astype(object).to_numpy()
    predicted = persistence_class_predictions(frame)
    mask = _known_class_mask(truth, predicted)
    if int(mask.sum()) == 0:
        return None
    return classification_metrics(truth[mask], predicted[mask], DEFAULT_NAMES)["macro_f1"]


def _persistence_cv(frame: pd.DataFrame, task: str, params: dict) -> tuple[float, float, str]:
    groups = frame["storm_id"].astype(str).to_numpy()
    stratify = (
        frame["imd5_class"]
        .astype(object)
        .where(frame["imd5_class"].notna(), "below")
        .astype(str)
        .to_numpy()
    )
    folds = list(
        iter_stratified_group_folds(
            stratify, groups, n_splits=int(params["n_splits"]), seed=int(params["seed"])
        )
    )
    scores: list[float] = []
    for _train_index, val_index in folds:
        score = _persistence_metric(frame.iloc[val_index], task)
        if score is not None:
            scores.append(score)
    if not scores:
        return float("nan"), float("nan"), "inner_cv:no_previous_frame"
    return (
        float(np.mean(scores)),
        float(np.std(scores)),
        "inner_cv:previous_same_storm_wind",
    )


def _format_cell(value: object) -> str:
    if isinstance(value, float):
        if not np.isfinite(value):
            return ""
        return f"{value:.4f}"
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).replace("|", "/")


def _markdown_table(frame: pd.DataFrame, columns: tuple[str, ...]) -> str:
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(_format_cell(row[column]) for column in columns) + " |"
        for _, row in frame.iterrows()
    ]
    return "\n".join([header, rule, *body])


def _comparison_frame(rows: list[dict]) -> pd.DataFrame:
    comparison = pd.DataFrame(rows, columns=list(COMPARISON_COLUMNS))
    for column in OPERATIONAL_COLUMNS:
        comparison[column] = np.nan
    comparison["quote_as_result"] = "no"
    return comparison


def write_comparison_files(
    destination: Path, rows: list[dict], metadata: dict | None
) -> pd.DataFrame:
    """Write the CSV and markdown comparison. Numbers stay `quote_as_result=no`."""
    comparison = _comparison_frame(rows)
    destination.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(destination / "baseline_comparison.csv", index=False)
    lines = [
        "# Baseline comparison",
        "",
        BANNER,
        "",
        "Validation scores. The test split was not used for fitting, selection, or SMOTE.",
        "Persistence is the previous same-storm wind, not a feature.",
        "Learned models are sklearn pipelines. SMOTE and class weights stay inside the training fold.",
        "The official competition side, when present, stays in `source_split`.",
        "`quote_as_result` is no. This is not a forecast and not an operational-readiness claim.",
        "",
    ]
    if metadata:
        lines.extend(["## Run", ""])
        for key, value in metadata.items():
            lines.append(f"- {key}: `{value}`")
        lines.append("")
    lines.extend([_markdown_table(comparison, _MARKDOWN_COLUMNS), ""])
    (destination / "baseline_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    (destination / "README.txt").write_text(BANNER + "\n", encoding="utf-8")
    if metadata:
        (destination / "run_record.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        record = [
            "# Baseline run",
            "",
            BANNER,
            "",
            "Official competition labels stay in `source_split`. The modeling split is storm-wise.",
            "The test split was not fit. `quote_as_result` is no.",
            "",
        ]
        for key, value in metadata.items():
            record.append(f"- {key}: `{value}`")
        record.append("")
        (destination / "RUN.md").write_text("\n".join(record), encoding="utf-8")
    return comparison


def _score_persistence(
    train: pd.DataFrame,
    val: pd.DataFrame,
    task: str,
    params: dict,
    synthetic: str,
    n_features: int,
    destination: Path,
) -> dict:
    """Score the previous same-storm wind. Nothing is fit and val storms stay out."""
    cv_mean, cv_std, cv_note = _persistence_cv(train, task, params)
    note = f"previous_same_storm_wind;not_a_feature;no_pipeline;{cv_note}"
    row = _empty_metric_row(task, "persistence", note, synthetic, n_features)
    if task == "regression":
        train_pred = persistence_predictions(train)
        val_pred = persistence_predictions(val)
        train_mask = np.isfinite(train_pred)
        val_mask = np.isfinite(val_pred)
        if int(val_mask.sum()) == 0:
            row["fit_notes"] = f"{note};no_previous_frame"
            return row
        truth = val["wind_speed_kt"].to_numpy(dtype=np.float64)[val_mask]
        predicted = val_pred[val_mask]
        scores = regression_metrics(truth, predicted)
        by_class = metrics_by_class(truth, predicted, val["imd5_class"].to_numpy()[val_mask])
        row.update(
            {
                "n_train_rows": int(train_mask.sum()),
                "n_val_rows": int(val_mask.sum()),
                "n_train_storms": int(train.iloc[np.flatnonzero(train_mask)]["storm_id"].nunique()),
                "n_val_storms": int(val.iloc[np.flatnonzero(val_mask)]["storm_id"].nunique()),
                "mae_kt": scores["mae_kt"],
                "rmse_kt": scores["rmse_kt"],
                "bias_kt": scores["bias_kt"],
                "cv_mae_kt_mean": cv_mean,
                "cv_mae_kt_std": cv_std,
            }
        )
        for class_name in DEFAULT_NAMES:
            class_scores = by_class.get(class_name)
            row[f"mae_{class_name}"] = np.nan if class_scores is None else class_scores["mae_kt"]
        save_regression_scatter(
            truth,
            predicted,
            destination / "scatter_persistence.png",
            title="persistence val scatter — not a forecast",
        )
        return row
    train_truth = train["imd5_class"].astype(object).to_numpy()
    val_truth = val["imd5_class"].astype(object).to_numpy()
    train_pred = persistence_class_predictions(train)
    val_pred = persistence_class_predictions(val)
    train_mask = _known_class_mask(train_truth, train_pred)
    val_mask = _known_class_mask(val_truth, val_pred)
    if int(val_mask.sum()) == 0:
        row["fit_notes"] = f"{note};no_previous_class"
        return row
    truth = val_truth[val_mask].astype(str)
    predicted = val_pred[val_mask].astype(str)
    scores = classification_metrics(truth, predicted, DEFAULT_NAMES)
    row.update(
        {
            "n_train_rows": int(train_mask.sum()),
            "n_val_rows": int(val_mask.sum()),
            "n_train_storms": int(train.iloc[np.flatnonzero(train_mask)]["storm_id"].nunique()),
            "n_val_storms": int(val.iloc[np.flatnonzero(val_mask)]["storm_id"].nunique()),
            "macro_f1": scores["macro_f1"],
            "cv_macro_f1_mean": cv_mean,
            "cv_macro_f1_std": cv_std,
        }
    )
    for class_name in DEFAULT_NAMES:
        row[f"recall_{class_name}"] = scores[f"recall_{class_name}"]
    save_confusion_matrix(
        truth,
        predicted,
        destination / "confusion_persistence.png",
        title="persistence val counts — not a forecast",
    )
    return row


def run_baselines(
    frame: pd.DataFrame,
    feature_names: tuple[str, ...] | list[str],
    config: dict,
    out_dir: str | Path,
    metadata: dict | None = None,
) -> BaselineReport:
    """Select on train, compare models on val, and leave the test split locked."""
    required = {"split", "storm_id", "wind_speed_kt", "imd5_class", "sample_id"}
    missing = required - set(frame.columns)
    if missing:
        raise BaselineDataError(f"baseline frame is missing {sorted(missing)}")
    unknown = set(frame["split"].astype(str).unique()) - {"train", "val", "test"}
    if unknown:
        raise BaselineDataError(f"unknown splits: {sorted(unknown)}")
    names = tuple(feature_names)
    assert_no_forbidden_inputs(names, "refusing to train on")
    parts = [frame[frame["split"] == name] for name in ("train", "val", "test")]
    assert_no_group_leakage(*parts)
    train = parts[0].copy()
    val = parts[1].copy()
    test = parts[2].copy()
    if train.empty or val.empty:
        raise BaselineDataError("train and val must both be non-empty")
    if train["storm_id"].nunique() < 3:
        raise BaselineDataError("need at least 3 training storms")

    params = baseline_params(config)
    choose = selection_params(config)
    selection = select_features(
        train,
        names,
        train["wind_speed_kt"].to_numpy(dtype=np.float64),
        source_split="train",
        correlation_threshold=choose["correlation_threshold"],
        max_features=choose["max_features"],
        method=choose["method"],
        n_estimators=choose["n_estimators"],
        seed=choose["seed"],
    )
    if selection.train_storms != frozenset(train["storm_id"].astype(str)):
        raise LeakageError("feature selection used storms outside the training split")
    selected = selection.names
    if not selected:
        raise BaselineDataError("feature selection returned no columns")
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    synthetic = _synthetic_flag(frame)
    fitted: set[str] = set()
    rows: list[dict] = []

    for model_name in list(params["regression_models"]):
        logger.info("Regression baseline %s", model_name)
        if model_name == "persistence":
            rows.append(
                _score_persistence(
                    train, val, "regression", params, synthetic, len(selected), destination
                )
            )
            write_comparison_files(destination, rows, metadata)
            continue
        if model_name == "xgboost" and not xgboost_available():
            rows.append(_empty_metric_row("regression", model_name, "not_installed", synthetic, len(selected)))
            write_comparison_files(destination, rows, metadata)
            continue
        cv_mean, cv_std, cv_note = _cross_val(
            train, selected, task="regression", model_name=model_name, params=params, fitted_storms=fitted
        )
        predicted, _pipeline, note = fit_numeric_pipeline(
            model_matrix(train, selected),
            train["wind_speed_kt"].to_numpy(dtype=np.float64),
            model_matrix(val, selected),
            task="regression",
            model_name=model_name,
            params=params,
            seed=int(params["seed"]),
            train_storms=train["storm_id"].astype(str),
            fitted_storms=fitted,
        )
        scores = regression_metrics(val["wind_speed_kt"], predicted)
        by_class = metrics_by_class(
            val["wind_speed_kt"].to_numpy(), predicted, val["imd5_class"].to_numpy()
        )
        row = _empty_metric_row("regression", model_name, f"{note};{cv_note}", synthetic, len(selected))
        row.update(
            {
                "n_train_rows": int(len(train)),
                "n_val_rows": int(len(val)),
                "n_train_storms": int(train["storm_id"].nunique()),
                "n_val_storms": int(val["storm_id"].nunique()),
                "mae_kt": scores["mae_kt"],
                "rmse_kt": scores["rmse_kt"],
                "bias_kt": scores["bias_kt"],
                "cv_mae_kt_mean": cv_mean,
                "cv_mae_kt_std": cv_std,
            }
        )
        for class_name in DEFAULT_NAMES:
            class_scores = by_class.get(class_name)
            row[f"mae_{class_name}"] = np.nan if class_scores is None else class_scores["mae_kt"]
        rows.append(row)
        save_regression_scatter(
            val["wind_speed_kt"].to_numpy(),
            predicted,
            destination / f"scatter_{model_name}.png",
            title=f"{model_name} val scatter — not a forecast",
        )
        write_comparison_files(destination, rows, metadata)

    train_cls = train[train["imd5_class"].notna()].copy()
    val_cls = val[val["imd5_class"].notna()].copy()
    if train_cls.empty or val_cls.empty:
        raise BaselineDataError("classification needs labeled rows in train and val")
    for model_name in list(params["classification_models"]):
        logger.info("Classification baseline %s", model_name)
        if model_name == "persistence":
            rows.append(
                _score_persistence(
                    train, val, "classification", params, synthetic, len(selected), destination
                )
            )
            write_comparison_files(destination, rows, metadata)
            continue
        if model_name == "xgboost" and not xgboost_available():
            rows.append(
                _empty_metric_row("classification", model_name, "not_installed", synthetic, len(selected))
            )
            write_comparison_files(destination, rows, metadata)
            continue
        cv_mean, cv_std, cv_note = _cross_val(
            train_cls,
            selected,
            task="classification",
            model_name=model_name,
            params=params,
            fitted_storms=fitted,
        )
        predicted, _pipeline, note = fit_numeric_pipeline(
            model_matrix(train_cls, selected),
            train_cls["imd5_class"].astype(str).to_numpy(),
            model_matrix(val_cls, selected),
            task="classification",
            model_name=model_name,
            params=params,
            seed=int(params["seed"]),
            train_storms=train_cls["storm_id"].astype(str),
            fitted_storms=fitted,
        )
        scores = classification_metrics(
            val_cls["imd5_class"].astype(str).to_numpy(), predicted, DEFAULT_NAMES
        )
        row = _empty_metric_row(
            "classification", model_name, f"{note};{cv_note}", synthetic, len(selected)
        )
        row.update(
            {
                "n_train_rows": int(len(train_cls)),
                "n_val_rows": int(len(val_cls)),
                "n_train_storms": int(train_cls["storm_id"].nunique()),
                "n_val_storms": int(val_cls["storm_id"].nunique()),
                "macro_f1": scores["macro_f1"],
                "cv_macro_f1_mean": cv_mean,
                "cv_macro_f1_std": cv_std,
            }
        )
        for class_name in DEFAULT_NAMES:
            row[f"recall_{class_name}"] = scores[f"recall_{class_name}"]
        rows.append(row)
        save_confusion_matrix(
            val_cls["imd5_class"].astype(str).to_numpy(),
            predicted,
            destination / f"confusion_{model_name}.png",
            title=f"{model_name} val counts — not a forecast",
        )
        write_comparison_files(destination, rows, metadata)

    comparison = write_comparison_files(destination, rows, metadata)
    payload = {
        "source_split": selection.source_split,
        "names": list(selection.names),
        "importances": selection.importances,
        "dropped_variance": list(selection.dropped_variance),
        "dropped_correlation": list(selection.dropped_correlation),
        "train_storms": sorted(selection.train_storms),
        "quote_as_result": "no",
    }
    (destination / "selected_features.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    save_feature_importance(
        selection.importances,
        destination / "feature_importance.png",
        title="Train-only selection scores — not a skill claim",
    )
    report = BaselineReport(
        comparison=comparison,
        selected_features=selected,
        train_storms=frozenset(train["storm_id"].astype(str)),
        val_storms=frozenset(val["storm_id"].astype(str)),
        test_storms=frozenset(test["storm_id"].astype(str)),
        fitted_storms=frozenset(fitted),
        n_rows_used_for_selection=int(len(train)),
        out_dir=destination,
    )
    leaked = report.fitted_storms & (report.val_storms | report.test_storms)
    if leaked:
        raise LeakageError(f"fit saw held-out storms: {sorted(leaked)}")
    if not report.fitted_storms <= report.train_storms:
        raise LeakageError("fit saw storms that are not in the training split")
    return report


def frame_for_storm_split(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Assign a storm-wise split. Drop any incoming ``split`` column first.

    ``source_split`` is left untouched. It is the official competition side
    and is not the modeling split.
    """
    split_cfg = config["split"]
    strategy = str(split_cfg["strategy"]).lower().strip()
    if strategy != "storm_wise":
        raise BaselineDataError(
            "baselines use storm-wise splits only via src.splits.assign_storms; "
            f"refusing split.strategy={strategy}"
        )
    work = frame.copy()
    if "split" in work.columns:
        work = work.drop(columns=["split"])
    holdout_basin = split_cfg["holdout_basin"] or None
    assigned = assign_storms(
        work,
        "storm_wise",
        test_fraction=float(split_cfg["test_fraction"]),
        val_fraction=float(split_cfg["val_fraction"]),
        seed=int(split_cfg["seed"]),
        holdout_years=[int(year) for year in (split_cfg["holdout_years"] or [])],
        holdout_basin=None if holdout_basin in {None, ""} else str(holdout_basin),
    )
    if "source_split" in frame.columns and "source_split" not in assigned.columns:
        raise BaselineDataError("source_split was dropped while assigning storms")
    return assigned


def _count_block(frame: pd.DataFrame, column: str) -> str:
    parts = []
    for name in sorted(frame[column].astype(str).unique()):
        part = frame.loc[frame[column].astype(str) == name]
        parts.append(f"{name}:rows={len(part)},storms={part['storm_id'].nunique()}")
    return "; ".join(parts)


def _run_metadata(
    assigned: pd.DataFrame,
    table: pd.DataFrame,
    config: dict,
    *,
    command: str | None,
    manifest_command: str | None,
    manifest: Path,
    dropped: int,
    workers: int,
) -> dict[str, str]:
    both = 0
    if "source_split" in assigned.columns:
        sides = [
            set(assigned.loc[assigned["source_split"].astype(str) == side, "storm_id"].astype(str))
            for side in sorted(assigned["source_split"].astype(str).unique())
        ]
        both = len(set.intersection(*sides)) if len(sides) > 1 else 0
    metadata = {
        "command": command or "",
        "manifest_command": manifest_command or "",
        "manifest": str(manifest),
        "split_strategy": "storm_wise",
        "split_seed": str(int(config["split"]["seed"])),
        "baseline_seed": str(int(config["baselines"]["seed"])),
        "feature_selection_seed": str(int(config["features"]["selection"]["seed"])),
        "test_fraction": str(float(config["split"]["test_fraction"])),
        "val_fraction": str(float(config["split"]["val_fraction"])),
        "workers": str(int(workers)),
        "rows": str(int(len(assigned))),
        "storms": str(int(assigned["storm_id"].nunique())),
        "rows_after_features": str(int(len(table))),
        "storms_after_features": str(int(table["storm_id"].nunique())),
        "modeling_split": _count_block(assigned, "split"),
        "dropped_cross_split_links": str(int(dropped)),
        "use_smote": str(bool(config["baselines"]["use_smote"])),
        "quote_as_result": "no",
        "official_split_used_for_training": "no",
    }
    if "source_split" in assigned.columns:
        metadata["source_split"] = _count_block(assigned, "source_split")
        metadata["storms_on_both_source_sides"] = str(both)
    return metadata


def _cache_key(manifest: Path, config: dict) -> dict[str, object]:
    return {
        "manifest": str(manifest.resolve()),
        "strategy": "storm_wise",
        "split_seed": int(config["split"]["seed"]),
        "test_fraction": float(config["split"]["test_fraction"]),
        "val_fraction": float(config["split"]["val_fraction"]),
        "feature_seed": int(config["features"]["selection"]["seed"]),
    }


def run_from_config(
    config_path: str | Path,
    manifest_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    workers: int = 1,
    feature_cache: str | Path | None = None,
    command: str | None = None,
    manifest_command: str | None = None,
) -> BaselineReport:
    """Run the Phase 2 comparison. Does not download imagery.

    The modeling split is always storm-wise. A ``source_split`` column is
    reference only.
    """
    config = load_config(config_path)
    manifest = Path(manifest_path or config["paths"]["manifest"])
    if not manifest.is_file():
        raise FileNotFoundError(
            f"missing manifest {manifest}. Point paths.manifest at a local file. "
            "This command does not download NASA, TCIR, or INSAT imagery."
        )
    destination = Path(out_dir) if out_dir else Path(config["paths"]["reports"]) / "baselines"
    frame = frame_for_storm_split(load_manifest(manifest), config)
    spec = FeatureSpec.from_config(config)
    cache = Path(feature_cache) if feature_cache else None
    meta_path = cache.with_suffix(cache.suffix + ".meta.json") if cache else None
    key = _cache_key(manifest, config)
    table = None
    dropped = 0
    if cache and cache.is_file() and meta_path and meta_path.is_file():
        saved = json.loads(meta_path.read_text(encoding="utf-8"))
        if saved.get("key") == key:
            table = pd.read_csv(cache)
            dropped = int(saved.get("dropped_cross_split_links", 0))
            logger.info("Loaded feature cache %s", cache)
    if table is None:
        table, dropped = extract_table(frame, manifest, spec, config, workers=workers)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(cache, index=False)
            meta_path.write_text(
                json.dumps({"key": key, "dropped_cross_split_links": int(dropped)}, indent=2),
                encoding="utf-8",
            )
    logger.info("Dropped %s cross-split temporal links", dropped)
    metadata = _run_metadata(
        frame,
        table,
        config,
        command=command,
        manifest_command=manifest_command,
        manifest=manifest,
        dropped=dropped,
        workers=workers,
    )
    return run_baselines(table, spec.feature_names(), config, destination, metadata=metadata)


def extract_table(frame, manifest, spec: FeatureSpec, config: dict, workers: int = 1):
    return extract_feature_table(
        frame,
        manifest,
        spec,
        max_missing_fraction=float(config["data"]["max_missing_fraction"]),
        workers=workers,
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--feature-cache", type=Path, default=None)
    parser.add_argument(
        "--manifest-command",
        default="",
        help="Exact command that built the manifest. Recorded in RUN.md. This command does not run it.",
    )
    args = parser.parse_args(argv)
    command = "python -m src.baselines " + " ".join(shlex.quote(arg) for arg in (argv if argv is not None else sys.argv[1:]))
    report = run_from_config(
        args.config,
        args.manifest,
        args.out,
        workers=args.workers,
        feature_cache=args.feature_cache,
        command=command,
        manifest_command=args.manifest_command or None,
    )
    print(BANNER)
    print(f"Wrote {report.out_dir / 'baseline_comparison.csv'}")
    print(f"Wrote {report.out_dir / 'baseline_comparison.md'}")
    print("quote_as_result=no for every row. Test storms were not fit.")


if __name__ == "__main__":
    main()
