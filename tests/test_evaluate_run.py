"""Tests for standalone evaluation and residual quantile merging into metrics.json."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    _LEVELS,
    _TOLERANCE,
    compute_residual_quantiles,
    evaluate_run_directory,
    merge_residual_quantiles,
)
from src.inference import ConfidenceScore


def test_compute_residual_quantiles_exact_schema():
    # 50 val samples and 50 test samples with controlled residuals
    # Val absolute residuals: 1..50
    # Test absolute residuals: 1..50 (same distribution -> exact empirical coverage)
    y_true_val = np.zeros(50, dtype=np.float64)
    y_pred_val = np.arange(1, 51, dtype=np.float64)

    y_true_test = np.zeros(50, dtype=np.float64)
    y_pred_test = np.arange(1, 51, dtype=np.float64)

    val_storms = ["201901", "201902"]
    test_storms = ["202001", "202002"]

    block = compute_residual_quantiles(
        y_true_val=y_true_val,
        y_pred_val=y_pred_val,
        y_true_test=y_true_test,
        y_pred_test=y_pred_test,
        val_storms=val_storms,
        test_storms=test_storms,
        levels=[0.8, 0.9],
        tolerance=0.05,
        is_synthetic=False,
    )

    assert block["fit_split"] == "val"
    assert block["eval_split"] == "test"
    assert block["val_storms"] == sorted(val_storms)
    assert block["test_storms"] == sorted(test_storms)
    assert block["synthetic"] is False
    assert "0.80" in block["levels"]
    assert "0.90" in block["levels"]

    row_90 = block["levels"]["0.90"]
    assert row_90["nominal"] == 0.9
    assert row_90["n"] == 50.0
    assert row_90["status"] == "within_tolerance"
    assert row_90["eval_split"] == "test"


def test_merge_residual_quantiles_passes_held_out_storm_check():
    y_true_val = np.zeros(100, dtype=np.float64)
    y_pred_val = np.linspace(1, 20, 100, dtype=np.float64)

    y_true_test = np.zeros(100, dtype=np.float64)
    y_pred_test = np.linspace(1, 20, 100, dtype=np.float64)

    val_storms = ["storm_val_1"]
    test_storms = ["storm_test_1"]

    block = compute_residual_quantiles(
        y_true_val=y_true_val,
        y_pred_val=y_pred_val,
        y_true_test=y_true_test,
        y_pred_test=y_pred_test,
        val_storms=val_storms,
        test_storms=test_storms,
        levels=[0.8, 0.9],
        tolerance=0.05,
        is_synthetic=False,
    )

    metrics = {
        "run_id": "test_run_001",
        "model": "cloudie",
        "dataset": "nasa_tc",
        "is_synthetic": False,
    }

    merged = merge_residual_quantiles(metrics, block, tolerance=0.05, levels=[0.8, 0.9])
    assert ConfidenceScore._quantile_block(merged) is not None
    assert ConfidenceScore._held_out_storm_check(merged) is True


def test_synthetic_run_fails_held_out_storm_check():
    y_true_val = np.zeros(50, dtype=np.float64)
    y_pred_val = np.arange(1, 51, dtype=np.float64)
    y_true_test = np.zeros(50, dtype=np.float64)
    y_pred_test = np.arange(1, 51, dtype=np.float64)

    block = compute_residual_quantiles(
        y_true_val=y_true_val,
        y_pred_val=y_pred_val,
        y_true_test=y_true_test,
        y_pred_test=y_pred_test,
        val_storms=["val_1"],
        test_storms=["test_1"],
        levels=[0.8, 0.9],
        tolerance=0.05,
        is_synthetic=True,
    )

    metrics = {"is_synthetic": True}
    merged = merge_residual_quantiles(metrics, block, tolerance=0.05, levels=[0.8, 0.9])
    assert ConfidenceScore._held_out_storm_check(merged) is False


def test_evaluate_run_directory_with_existing_predictions_npz(tmp_path: Path):
    run_dir = tmp_path / "artifacts" / "runs" / "run_abc"
    run_dir.mkdir(parents=True)

    # Create manifest
    manifest_df = pd.DataFrame({
        "sample_id": ["s1", "s2", "s3", "s4"],
        "storm_id": ["val_storm", "val_storm", "test_storm", "test_storm"],
        "split": ["val", "val", "test", "test"],
        "wind_speed_kt": [35.0, 45.0, 55.0, 65.0],
    })
    manifest_df.to_csv(run_dir / "split_manifest.csv", index=False)

    # Create predictions.npz with 20 points
    val_true = np.zeros(20, dtype=np.float64)
    val_pred = np.linspace(1, 10, 20, dtype=np.float64)
    test_true = np.zeros(20, dtype=np.float64)
    test_pred = np.linspace(1, 10, 20, dtype=np.float64)

    np.savez(
        run_dir / "predictions.npz",
        y_true_val=val_true,
        y_pred_val=val_pred,
        y_true_test=test_true,
        y_pred_test=test_pred,
        mc_samples_test=test_pred[None, :],
    )

    # Initial metrics.json
    (run_dir / "metrics.json").write_text(
        json.dumps({"run_id": "run_abc", "is_synthetic": False}) + "\n",
        encoding="utf-8",
    )

    updated = evaluate_run_directory(run_dir, tolerance=0.05)
    assert (run_dir / "metrics.json").is_file()

    saved_metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert ConfidenceScore._held_out_storm_check(saved_metrics) is True
    assert saved_metrics["uncertainty"]["residual_quantile"]["test_storms"] == ["test_storm"]
    assert saved_metrics["uncertainty"]["residual_quantile"]["val_storms"] == ["val_storm"]


def test_evaluate_run_directory_from_manifest_predictions(tmp_path: Path):
    run_dir = tmp_path / "run_from_manifest"
    run_dir.mkdir(parents=True)

    # Manifest with precomputed y_pred column and linearly spaced residuals
    n_points = 50
    winds = np.zeros(2 * n_points, dtype=np.float64)
    preds = np.concatenate([
        np.linspace(1, 10, n_points, dtype=np.float64),
        np.linspace(1, 10, n_points, dtype=np.float64),
    ])

    manifest_df = pd.DataFrame({
        "sample_id": [f"s_{i}" for i in range(2 * n_points)],
        "storm_id": ["storm_v"] * n_points + ["storm_t"] * n_points,
        "split": ["val"] * n_points + ["test"] * n_points,
        "wind_speed_kt": winds,
        "y_pred": preds,
    })
    manifest_df.to_csv(run_dir / "split_manifest.csv", index=False)
    (run_dir / "metrics.json").write_text(
        json.dumps({"is_synthetic": False}) + "\n",
        encoding="utf-8",
    )

    evaluate_run_directory(run_dir, tolerance=0.05)

    # Verify predictions.npz was generated
    assert (run_dir / "predictions.npz").is_file()
    npz_data = dict(np.load(str(run_dir / "predictions.npz")))
    assert "y_true_test" in npz_data
    assert "y_pred_test" in npz_data
    assert "mc_samples_test" in npz_data
    assert len(npz_data["y_true_test"]) == n_points

    # Verify metrics.json updated and passes held-out storm check
    saved_metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert ConfidenceScore._held_out_storm_check(saved_metrics) is True


def test_scripts_evaluate_run_cli(tmp_path: Path):
    from scripts.evaluate_run import main as evaluate_main

    run_dir = tmp_path / "run_cli_test"
    run_dir.mkdir(parents=True)

    n_points = 40
    winds = np.zeros(2 * n_points, dtype=np.float64)
    preds = np.concatenate([
        np.linspace(1, 10, n_points, dtype=np.float64),
        np.linspace(1, 10, n_points, dtype=np.float64),
    ])

    manifest_df = pd.DataFrame({
        "sample_id": [f"s_{i}" for i in range(2 * n_points)],
        "storm_id": ["val_storm"] * n_points + ["test_storm"] * n_points,
        "split": ["val"] * n_points + ["test"] * n_points,
        "wind_speed_kt": winds,
        "y_pred": preds,
    })
    manifest_df.to_csv(run_dir / "split_manifest.csv", index=False)
    (run_dir / "metrics.json").write_text(
        json.dumps({"is_synthetic": False}) + "\n",
        encoding="utf-8",
    )

    exit_code = evaluate_main([f"--run-dir={run_dir}"])
    assert exit_code == 0

    saved_metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert ConfidenceScore._held_out_storm_check(saved_metrics) is True


def test_src_evaluate_cli_run_dir(tmp_path: Path):
    from src.evaluate import main as src_eval_main

    run_dir = tmp_path / "src_eval_run_dir"
    run_dir.mkdir(parents=True)

    n_points = 40
    winds = np.zeros(2 * n_points, dtype=np.float64)
    preds = np.concatenate([
        np.linspace(1, 10, n_points, dtype=np.float64),
        np.linspace(1, 10, n_points, dtype=np.float64),
    ])

    manifest_df = pd.DataFrame({
        "sample_id": [f"s_{i}" for i in range(2 * n_points)],
        "storm_id": ["v_storm"] * n_points + ["t_storm"] * n_points,
        "split": ["val"] * n_points + ["test"] * n_points,
        "wind_speed_kt": winds,
        "y_pred": preds,
    })
    manifest_df.to_csv(run_dir / "split_manifest.csv", index=False)
    (run_dir / "metrics.json").write_text(
        json.dumps({"is_synthetic": False}) + "\n",
        encoding="utf-8",
    )

    res = src_eval_main([f"--run-dir={run_dir}"])
    assert isinstance(res, dict)
    assert ConfidenceScore._held_out_storm_check(res) is True


def test_export_predictions_cli(tmp_path: Path):
    from scripts.export_predictions import main as export_main

    run_dir = tmp_path / "export_pred_cli_test"
    run_dir.mkdir(parents=True)

    n_points = 20
    winds = np.zeros(2 * n_points, dtype=np.float64)
    preds = np.concatenate([
        np.linspace(1, 10, n_points, dtype=np.float64),
        np.linspace(1, 10, n_points, dtype=np.float64),
    ])

    manifest_df = pd.DataFrame({
        "sample_id": [f"s_{i}" for i in range(2 * n_points)],
        "storm_id": ["v_storm"] * n_points + ["t_storm"] * n_points,
        "split": ["val"] * n_points + ["test"] * n_points,
        "wind_speed_kt": winds,
        "y_pred": preds,
    })
    manifest_df.to_csv(run_dir / "split_manifest.csv", index=False)
    (run_dir / "metrics.json").write_text(
        json.dumps({"is_synthetic": False}) + "\n",
        encoding="utf-8",
    )

    exit_code = export_main([f"--run-dir={run_dir}"])
    assert exit_code == 0
    assert (run_dir / "predictions.npz").is_file()
    saved = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert ConfidenceScore._held_out_storm_check(saved) is True
