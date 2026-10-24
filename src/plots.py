"""Comparison figures for classical baselines.

Figures are software artifacts. A title that includes a number is not a
result to quote, and these plots are not warnings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.schema import DEFAULT_NAMES


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: str | Path,
    title: str,
    labels: tuple[str, ...] = DEFAULT_NAMES,
) -> Path:
    """Write a count matrix. Classes absent from both sides stay in the grid."""
    plt = _pyplot()
    truth = np.asarray(y_true).ravel()
    pred = np.asarray(y_pred).ravel()
    index = {label: position for position, label in enumerate(labels)}
    counts = np.zeros((len(labels), len(labels)), dtype=int)
    for actual, guess in zip(truth, pred, strict=True):
        if actual not in index or guess not in index:
            continue
        counts[index[actual], index[guess]] += 1
    figure, axis = plt.subplots(figsize=(6.2, 5.2))
    image = axis.imshow(counts, cmap="Blues")
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Best-track class (noisy label)")
    axis.set_title(title)
    for row in range(counts.shape[0]):
        for col in range(counts.shape[1]):
            axis.text(col, row, str(counts[row, col]), ha="center", va="center", fontsize=8)
    figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=100)
    plt.close(figure)
    return destination


def save_feature_importance(
    importances: dict[str, float],
    path: str | Path,
    title: str,
    limit: int = 15,
) -> Path | None:
    """Horizontal bars for a train-only selection score. Empty input writes nothing."""
    if not importances:
        return None
    plt = _pyplot()
    ordered = sorted(importances.items(), key=lambda item: (item[1], item[0]))
    ordered = ordered[-limit:]
    labels = [name for name, _score in ordered]
    scores = [score for _name, score in ordered]
    figure, axis = plt.subplots(figsize=(7.0, 5.0))
    axis.barh(labels, scores, color="#3d5a40")
    axis.set_xlabel("Train-split selection score")
    axis.set_title(title)
    figure.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=100)
    plt.close(figure)
    return destination


def save_regression_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: str | Path,
    title: str,
) -> Path:
    plt = _pyplot()
    truth = np.asarray(y_true, dtype=np.float64).ravel()
    pred = np.asarray(y_pred, dtype=np.float64).ravel()
    figure, axis = plt.subplots(figsize=(5.2, 5.0))
    axis.scatter(truth, pred, s=18, alpha=0.8, color="#1d3557")
    low = float(min(truth.min(), pred.min()))
    high = float(max(truth.max(), pred.max()))
    axis.plot([low, high], [low, high], color="#9b2226", linewidth=1)
    axis.set_xlabel("Label wind (kt)")
    axis.set_ylabel("Predicted wind (kt)")
    axis.set_title(title)
    figure.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=100)
    plt.close(figure)
    return destination
