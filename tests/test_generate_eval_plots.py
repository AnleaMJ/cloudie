"""Tests for scripts/generate_eval_plots.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.generate_eval_plots import generate_eval_plots, main


def test_generate_eval_plots_creates_all_three_pngs(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    out_dir = tmp_path / "plots"

    # Create synthetic test fixture
    rng = np.random.default_rng(123)
    n_test = 40
    n_val = 30
    n_draws = 10

    # True winds spanning multiple Saffir-Simpson categories (20 to 110 kt)
    y_true_test = rng.uniform(20.0, 110.0, size=n_test).astype(np.float64)
    mc_samples_test = y_true_test + rng.normal(0.0, 5.0, size=(n_draws, n_test))

    y_true_val = rng.uniform(20.0, 110.0, size=n_val).astype(np.float64)
    mc_samples_val = y_true_val + rng.normal(0.0, 5.0, size=(n_draws, n_val))

    np.savez(
        run_dir / "calibration.npz",
        y_true_test=y_true_test,
        mc_samples_test=mc_samples_test,
        y_true_val=y_true_val,
        mc_samples_val=mc_samples_val,
    )

    metrics = {
        "run_id": "test-plot-run",
        "dataset": "synthetic",
        "model": "simple_cnn",
        "test": {
            "mae_kt": 4.2,
            "rmse_kt": 5.8,
            "bias_kt": 0.1,
            "n": n_test,
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    result = generate_eval_plots(run_dir=run_dir, out_dir=out_dir)

    expected_files = [
        out_dir / "eval_pred_vs_actual.png",
        out_dir / "eval_error_by_category.png",
        out_dir / "eval_conformal_coverage.png",
    ]

    for p in expected_files:
        assert p.is_file(), f"Expected plot {p.name} was not created"
        assert p.stat().st_size > 1000, f"Plot {p.name} is unexpectedly small"

    assert result["pred_vs_actual"] == expected_files[0]
    assert result["error_by_category"] == expected_files[1]
    assert result["conformal_coverage"] == expected_files[2]


def test_generate_eval_plots_cli(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")

    run_dir = tmp_path / "cli_run"
    run_dir.mkdir(parents=True)
    out_dir = tmp_path / "cli_plots"

    rng = np.random.default_rng(456)
    n = 20
    y_test = rng.uniform(25.0, 80.0, size=n).astype(np.float64)
    mc_test = y_test + rng.normal(0.0, 4.0, size=(5, n))

    np.savez(
        run_dir / "calibration.npz",
        y_true_test=y_test,
        mc_samples_test=mc_test,
    )
    (run_dir / "metrics.json").write_text(json.dumps({"run_id": "cli-test"}), encoding="utf-8")

    code = main(["--run-dir", str(run_dir), "--out-dir", str(out_dir)])
    assert code == 0

    assert (out_dir / "eval_pred_vs_actual.png").is_file()
    assert (out_dir / "eval_error_by_category.png").is_file()
    assert (out_dir / "eval_conformal_coverage.png").is_file()
