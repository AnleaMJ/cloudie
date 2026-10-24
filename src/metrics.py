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
