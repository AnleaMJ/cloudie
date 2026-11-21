"""Tests for scripts/finalize_run.py post-training publishing pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.finalize_run import finalize_run

ROOT = Path(__file__).resolve().parents[1]


def test_finalize_run_refuses_synthetic_runs(tmp_path: Path) -> None:
    """Hard gate: finalize_run must refuse any run with is_synthetic=True."""
    run_dir = tmp_path / "runs" / "smoke-run"
    run_dir.mkdir(parents=True)
    metrics = {
        "run_id": "smoke-run-001",
        "is_synthetic": True,
        "quote_as_result": "yes",
        "val": {"mae_kt": 12.3},
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    with pytest.raises(ValueError, match="synthetic/smoke"):
        finalize_run(run_dir)


def test_finalize_run_refuses_unapproved_quote_as_result(tmp_path: Path) -> None:
    """Hard gate: finalize_run must refuse runs without quote_as_result='yes'."""
    run_dir = tmp_path / "runs" / "unapproved-run"
    run_dir.mkdir(parents=True)
    metrics = {
        "run_id": "unapproved-001",
        "is_synthetic": False,
        "quote_as_result": "no",
        "val": {"mae_kt": 8.5},
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    with pytest.raises(ValueError, match="quote_as_result"):
        finalize_run(run_dir)


def test_finalize_run_updates_readme_and_renders_model_card(tmp_path: Path) -> None:
    """Verify that a legitimate approved run successfully updates README and outputs MODEL_CARD.md."""
    run_dir = tmp_path / "runs" / "real-convlstm-001"
    run_dir.mkdir(parents=True)

    metrics = {
        "run_id": "real-convlstm-001",
        "is_synthetic": False,
        "quote_as_result": "yes",
        "model": "convlstm",
        "dataset": "incyde",
        "val": {
            "scored": True,
            "mae_kt": 7.42,
            "rmse_kt": 9.15,
            "bias_kt": -0.32,
            "n": 120,
            "by_bin": {
                "D_DD": {"n": 30, "mae_kt": 6.1, "rmse_kt": 7.5, "bias_kt": 0.1},
                "CS": {"n": 50, "mae_kt": 7.0, "rmse_kt": 8.8, "bias_kt": -0.4},
            },
        },
        "calibration": {
            "mace": 0.042,
            "ence": 0.051,
            "spearman_rho": 0.45,
            "coverage_90_raw": 0.78,
            "width_90_raw_kt": 15.2,
            "coverage_90_scaled": 0.898,
            "width_90_scaled_kt": 18.5,
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    # Mock README.md with benchmark tags
    readme_path = tmp_path / "README.md"
    readme_path.write_text(
        "# Cloudie\n\n"
        "<!-- BENCHMARK:START -->\n"
        "Old benchmarks\n"
        "<!-- BENCHMARK:END -->\n\n"
        "## Scientific Honesty\n",
        encoding="utf-8",
    )

    template_path = ROOT / "docs" / "HF_MODEL_CARD_TEMPLATE.md"

    results = finalize_run(run_dir, readme_path=readme_path, template_path=template_path)

    # 1. Check README.md was updated
    updated_readme = readme_path.read_text(encoding="utf-8")
    assert "<!-- BENCHMARK:START -->" in updated_readme
    assert "<!-- BENCHMARK:END -->" in updated_readme
    assert "real-convlstm-001" in updated_readme
    assert "7.42" in updated_readme
    assert "Old benchmarks" not in updated_readme

    # 2. Check MODEL_CARD.md was created with run_id populated
    model_card = run_dir / "MODEL_CARD.md"
    assert model_card.is_file()
    card_text = model_card.read_text(encoding="utf-8")
    assert "real-convlstm-001" in card_text
    assert "<run_id>" not in card_text
    assert "Anlea/cloudie-weights" in card_text
    assert "7.42" in card_text
