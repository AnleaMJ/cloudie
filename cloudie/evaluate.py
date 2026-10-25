"""Evaluate a checkpoint against the train-set mean.

Operational baselines (ADT, SATCON, Dvorak) stay empty until those outputs
are actually joined. A synthetic checkpoint is not a skill score.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cloudie.config import checkpoint_path as configured_checkpoint
from cloudie.config import image_sizes_match, load_config, parse_image_size
from cloudie.data_loader import load_manifest
from cloudie.metrics import mean_baseline, metrics_by_class, regression_metrics
from cloudie.preprocessing import NormStats
from cloudie.sequences import stack_windows
from cloudie.train import (
    NOT_A_FORECAST,
    SMOKE_BANNER,
    assert_normalize,
    build_net,
    model_n_frames,
    project_labels,
    split_frame,
    stack_split,
    uses_sequences,
)

OPERATIONAL_BASELINES = {"ADT": None, "SATCON": None, "Dvorak": None}


@dataclass(frozen=True)
class Evaluation:
    model: dict[str, float]
    mean_baseline: dict[str, float]
    by_class: dict[str, dict[str, float]]
    operational_baselines: dict[str, None]
    residual_plot: Path
    is_synthetic: bool
    present_as_forecast: bool
    note: str


def write_residual_plot(
    y_true: np.ndarray, y_pred: np.ndarray, path: Path, synthetic: bool
) -> Path:
    """Scatter residual against the label. The file is a software artifact."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    truth = np.asarray(y_true, dtype=np.float64).ravel()
    pred = np.asarray(y_pred, dtype=np.float64).ravel()
    residual = pred - truth
    figure, axis = plt.subplots()
    axis.scatter(truth, residual, s=16)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Label wind (kt)")
    axis.set_ylabel("Prediction minus label (kt)")
    if synthetic:
        axis.set_title("Residuals on synthetic smoke data. Do not report scores.")
    else:
        axis.set_title("Residuals. Not an operational verification.")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
    return path


def _load_torch_checkpoint(path: Path) -> dict:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _predict(model, images: np.ndarray) -> np.ndarray:
    import torch

    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(np.ascontiguousarray(images))
        output = model(tensor)
        wind_hat = output[0] if isinstance(output, tuple) else output
    return wind_hat.detach().cpu().numpy().reshape(-1)


def evaluate(config: dict, checkpoint_path: str | Path | None = None) -> Evaluation:
    """Score the test split. Normalization stats come from the checkpoint."""
    path = Path(checkpoint_path) if checkpoint_path is not None else configured_checkpoint(config)
    if not path.is_file():
        raise FileNotFoundError(
            f"no checkpoint at {path}. Train first, or pass --checkpoint. "
            "A missing checkpoint is not a forecast."
        )
    payload = _load_torch_checkpoint(path)
    assert_normalize(config)
    if str(config["model"]["name"]) != str(payload["model_name"]):
        raise ValueError("config model.name does not match the checkpoint")
    if not image_sizes_match(config["data"]["image_size"], payload["image_size"]):
        raise ValueError("config image_size does not match the checkpoint")
    manifest = Path(config["paths"]["manifest"])
    frame = load_manifest(manifest)
    assigned = split_frame(frame, config)
    assigned["_class"] = project_labels(assigned, config)
    stats = NormStats(
        mean=float(payload["norm_mean"]),
        std=float(payload["norm_std"]),
        source_split=str(payload["norm_source_split"]),
    )
    if stats.source_split != "train":
        raise ValueError("checkpoint normalization was not fit on train")
    image_size = parse_image_size(payload["image_size"])
    max_missing = float(payload["max_missing_fraction"])
    if uses_sequences(config):
        if int(payload["n_frames"]) != model_n_frames(config):
            raise ValueError("config model.n_frames does not match the checkpoint")
        if payload.get("hidden") is not None and int(config["model"]["hidden"]) != int(
            payload["hidden"]
        ):
            raise ValueError("config model.hidden does not match the checkpoint")
        test_images, test_winds, test_labels, _size = stack_windows(
            assigned,
            manifest,
            "test",
            image_size,
            stats,
            max_missing,
            model_n_frames(config),
        )
    else:
        test_images, test_winds, test_labels = stack_split(
            assigned, manifest, "test", image_size, stats, max_missing
        )
    model = build_net(config, size_head=bool(payload.get("size_head", False)))
    model.load_state_dict(payload["state_dict"])
    predicted = _predict(model, test_images)
    baseline = mean_baseline([float(payload["mean_baseline_kt"])], len(test_winds))
    # Recompute the train mean from the manifest split. It must match the checkpoint.
    train_winds = assigned.loc[assigned["split"] == "train", "wind_speed_kt"].to_numpy()
    current_mean = float(np.mean(np.asarray(train_winds, dtype=np.float64)))
    if abs(current_mean - float(payload["mean_baseline_kt"])) > 1e-4:
        raise ValueError("train-set mean changed since the checkpoint was written")
    synthetic = bool(payload["is_synthetic"])
    note = SMOKE_BANNER if synthetic else NOT_A_FORECAST
    by_class: dict[str, dict[str, float]] = {}
    if payload.get("wind_averaging"):
        by_class = metrics_by_class(
            test_winds,
            predicted,
            np.asarray([label if label is not None else pd.NA for label in test_labels]),
        )
    report_dir = Path(config["paths"]["reports"])
    report_dir.mkdir(parents=True, exist_ok=True)
    plot_path = write_residual_plot(test_winds, predicted, report_dir / "residual.png", synthetic)
    result = Evaluation(
        model=regression_metrics(test_winds, predicted),
        mean_baseline=regression_metrics(test_winds, baseline),
        by_class=by_class,
        operational_baselines=dict(OPERATIONAL_BASELINES),
        residual_plot=plot_path,
        is_synthetic=synthetic,
        present_as_forecast=False,
        note=note,
    )
    report = {
        "note": result.note,
        "present_as_forecast": False,
        "is_synthetic": result.is_synthetic,
        "model": result.model,
        "mean_baseline": result.mean_baseline,
        "by_class": result.by_class,
        "operational_baselines": result.operational_baselines,
        "residual_plot": str(plot_path),
    }
    (report_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(note)
    print(f"Wrote {plot_path}")
    return result


def main(argv: list[str] | None = None) -> Evaluation:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    return evaluate(config, args.checkpoint)


if __name__ == "__main__":
    main()
