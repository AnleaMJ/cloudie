"""Regression metrics. No operational baseline is implied."""

from __future__ import annotations

import numpy as np
import pandas as pd


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
        mask = labels == name
        if np.asarray(mask).sum() == 0 or pd.isna(name):
            continue
        out[str(name)] = regression_metrics(truth[mask], pred[mask])
    return out


def mean_baseline(y_train: np.ndarray, n_rows: int) -> np.ndarray:
    """Predict the training mean. Fit this on train storms only."""
    return np.full(n_rows, float(np.mean(np.asarray(y_train, dtype=np.float64))))
