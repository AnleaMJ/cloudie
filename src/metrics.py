"""Regression and classification metrics. No operational baseline is implied."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=np.float64).ravel()
    pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if truth.shape != pred.shape or truth.size == 0:
        raise ValueError("y_true and y_pred must be non-empty and the same shape")
    error = pred - truth
    return {
        "n": float(truth.size),
        "mae_kt": float(np.mean(np.abs(error))),
        "rmse_kt": float(np.sqrt(np.mean(error**2))),
        "bias_kt": float(np.mean(error)),
    }


def _as_seed(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _storm_index(storm_ids: np.ndarray) -> list[np.ndarray]:
    """Row positions of each storm, in order of first appearance."""
    order: list[str] = []
    buckets: dict[str, list[int]] = {}
    for index, storm in enumerate(np.asarray(storm_ids, dtype=object).ravel().tolist()):
        if storm is None or pd.isna(storm):
            raise ValueError("storm_ids must be present on every row")
        key = str(storm)
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(index)
    return [np.asarray(buckets[key], dtype=np.int64) for key in order]


def storm_bootstrap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    storm_ids: np.ndarray,
    *,
    n_resamples: int = 1000,
    seed: int = 0,
    ci_level: float = 0.95,
) -> dict[str, float | int | str]:
    """RMSE, MAE, and bias with a percentile interval that resamples storms.

    Bias is prediction minus true wind. Each resample draws as many storm
    IDs as there are storms, with replacement, and keeps every row of a
    drawn storm. A storm drawn twice contributes its rows twice. Rows are
    not drawn on their own. The point values are the full sample, not the
    mean of the resamples. Storms are ordered by first appearance. The
    generator is ``numpy.random.default_rng(seed)``.
    """
    truth = np.asarray(y_true, dtype=np.float64).ravel()
    pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if truth.shape != pred.shape or truth.size == 0:
        raise ValueError("y_true and y_pred must be non-empty and the same shape")
    if not np.isfinite(truth).all() or not np.isfinite(pred).all():
        raise ValueError("y_true and y_pred must be finite")
    ids = np.asarray(storm_ids, dtype=object).ravel()
    if ids.size != truth.size:
        raise ValueError("storm_ids must align with y_true")
    groups = _storm_index(ids)
    if not groups:
        raise ValueError("storm_ids must align with y_true")
    resamples = _as_seed(n_resamples, "n_resamples")
    if resamples < 1:
        raise ValueError("n_resamples must be at least 1")
    seed_value = _as_seed(seed, "seed")
    level = float(ci_level)
    if not 0.0 < level < 1.0:
        raise ValueError(f"ci_level must be between 0 and 1, got {level}")
    point = regression_metrics(truth, pred)
    error = pred - truth
    n_per = np.asarray([group.size for group in groups], dtype=np.float64)
    sum_abs = np.asarray([np.abs(error[group]).sum() for group in groups], dtype=np.float64)
    sum_sq = np.asarray([(error[group] ** 2).sum() for group in groups], dtype=np.float64)
    sum_err = np.asarray([error[group].sum() for group in groups], dtype=np.float64)
    drawn = np.random.default_rng(seed_value).integers(0, len(groups), size=(resamples, len(groups)))
    mae_s = np.empty(resamples, dtype=np.float64)
    rmse_s = np.empty(resamples, dtype=np.float64)
    bias_s = np.empty(resamples, dtype=np.float64)
    for index, draw in enumerate(drawn):
        counts = np.bincount(draw, minlength=len(groups)).astype(np.float64)
        total = float(counts @ n_per)
        mae_s[index] = float(counts @ sum_abs) / total
        rmse_s[index] = float(np.sqrt((counts @ sum_sq) / total))
        bias_s[index] = float(counts @ sum_err) / total
    alpha = (1.0 - level) / 2.0
    out: dict[str, float | int | str] = {
        "n_rows": int(truth.size),
        "n_storms": int(len(groups)),
        "mae_kt": float(point["mae_kt"]),
        "rmse_kt": float(point["rmse_kt"]),
        "bias_kt": float(point["bias_kt"]),
        "n_resamples": resamples,
        "seed": seed_value,
        "ci_level": level,
        "resample": "storm_id",
    }
    for name, samples in (
        ("mae_kt", mae_s),
        ("rmse_kt", rmse_s),
        ("bias_kt", bias_s),
    ):
        out[f"{name}_ci_low"] = float(np.quantile(samples, alpha, method="linear"))
        out[f"{name}_ci_high"] = float(np.quantile(samples, 1.0 - alpha, method="linear"))
    return out


def metrics_by_class(
    y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray
) -> dict[str, dict[str, float]]:
    truth = np.asarray(y_true, dtype=np.float64).ravel()
    pred = np.asarray(y_pred, dtype=np.float64).ravel()
    labels = np.asarray(classes)
    if len(labels) != truth.size:
        raise ValueError("classes must align with y_true")
    out: dict[str, dict[str, float]] = {}
    for name in pd.unique(labels):
        if pd.isna(name):
            continue
        mask = labels == name
        if np.asarray(mask).sum() == 0:
            continue
        out[str(name)] = regression_metrics(truth[mask], pred[mask])
    return out


def mean_baseline(y_train: np.ndarray, n_rows: int) -> np.ndarray:
    """Predict the training mean. Fit this on train storms only."""
    return np.full(n_rows, float(np.mean(np.asarray(y_train, dtype=np.float64))))


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: tuple[str, ...],
) -> dict[str, float]:
    """Macro-F1 over classes present in ``y_true``, plus per-class recall.

    Recall is NaN when the validation side has no samples of that class.
    Accuracy is omitted on purpose: a single headline rate hides the rare
    intense classes.
    """
    truth = np.asarray(y_true).ravel()
    pred = np.asarray(y_pred).ravel()
    if truth.shape != pred.shape or truth.size == 0:
        raise ValueError("y_true and y_pred must be non-empty and the same shape")
    present = [label for label in labels if np.any(truth == label)]
    if not present:
        present = [str(value) for value in pd.unique(truth)]
    macro = float(
        f1_score(truth, pred, labels=present, average="macro", zero_division=0)
    )
    recalls = recall_score(
        truth, pred, labels=list(labels), average=None, zero_division=0
    )
    out: dict[str, float] = {"n": float(truth.size), "macro_f1": macro}
    for label, recall in zip(labels, recalls, strict=True):
        if not np.any(truth == label):
            out[f"recall_{label}"] = float("nan")
        else:
            out[f"recall_{label}"] = float(recall)
    return out
