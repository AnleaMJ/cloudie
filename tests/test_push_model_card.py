"""Tests for scripts/push_model_card.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.push_model_card import main, push_model_card, render_model_card_content, validate_metrics

ROOT = Path(__file__).resolve().parents[1]


def test_validate_metrics_accepts_reviewed_real_run() -> None:
    valid_metrics = {
        "run_id": "real-storm-001",
        "is_synthetic": False,
        "quote_as_result": "yes",
        "model": "convlstm",
        "dataset": "incyde",
        "test": {
            "scored": True,
            "mae_kt": 5.12,
            "rmse_kt": 7.45,
            "bias_kt": -0.15,
            "n": 120,
        },
    }
    validate_metrics(valid_metrics)


def test_validate_metrics_refuses_synthetic_or_unapproved() -> None:
    synthetic = {
        "run_id": "smoke-run",
        "is_synthetic": True,
        "quote_as_result": "yes",
        "model": "simple_cnn",
        "dataset": "smoke",
        "test": {"mae_kt": 1.0, "rmse_kt": 1.0},
    }
    with pytest.raises(ValueError, match="synthetic/smoke"):
        validate_metrics(synthetic)

    unapproved = {
        "run_id": "unapproved-run",
        "is_synthetic": False,
        "quote_as_result": "no",
        "model": "simple_cnn",
        "dataset": "nasa",
        "test": {"mae_kt": 1.0, "rmse_kt": 1.0},
    }
    with pytest.raises(ValueError, match="quote_as_result"):
        validate_metrics(unapproved)


def test_validate_metrics_requires_mandatory_keys() -> None:
    missing_id = {
        "is_synthetic": False,
        "quote_as_result": "yes",
        "model": "simple_cnn",
        "dataset": "nasa",
        "test": {"mae_kt": 1.0, "rmse_kt": 1.0},
    }
    with pytest.raises(ValueError, match="missing required top-level key: run_id"):
        validate_metrics(missing_id)

    missing_metrics = {
        "run_id": "run-001",
        "is_synthetic": False,
        "quote_as_result": "yes",
        "model": "simple_cnn",
        "dataset": "nasa",
    }
    with pytest.raises(ValueError, match="scored split"):
        validate_metrics(missing_metrics)


def test_render_model_card_dry_run(tmp_path: Path) -> None:
    metrics = {
        "run_id": "cyclone-run-007",
        "is_synthetic": False,
        "quote_as_result": "yes",
        "model": "convlstm",
        "dataset": "incyde",
        "test": {
            "scored": True,
            "mae_kt": 4.88,
            "rmse_kt": 6.92,
            "bias_kt": 0.22,
            "n": 250,
            "by_bin": {
                "D_DD": {"n": 50, "mae_kt": 3.5, "rmse_kt": 4.5, "bias_kt": 0.1},
                "CS": {"n": 200, "mae_kt": 5.2, "rmse_kt": 7.3, "bias_kt": 0.2},
            },
        },
    }
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    out_file = tmp_path / "rendered_README.md"
    rendered_path = push_model_card(
        run_dir=run_dir,
        template_path=ROOT / "docs" / "HF_MODEL_CARD_TEMPLATE.md",
        out_path=out_file,
        dry_run=True,
    )

    assert rendered_path == out_file
    assert out_file.is_file()

    content = out_file.read_text(encoding="utf-8")
    assert content.startswith("---\nlicense: mit\nlibrary_name: pytorch\n")
    assert "conformal-prediction" in content
    assert "tropical-cyclone" in content
    assert "cyclone-run-007" in content
    assert "<run_id>" not in content
    assert "4.88" in content
    assert "6.92" in content


def test_push_model_card_cli_dry_run(tmp_path: Path) -> None:
    metrics = {
        "run_id": "cli-test-run",
        "is_synthetic": False,
        "quote_as_result": "yes",
        "model": "simple_cnn",
        "dataset": "tcir",
        "val": {
            "scored": True,
            "mae_kt": 6.1,
            "rmse_kt": 8.4,
            "bias_kt": -0.4,
            "n": 80,
        },
    }
    run_dir = tmp_path / "cli_run"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    out_file = tmp_path / "cli_README.md"
    code = main([
        "--run-dir", str(run_dir),
        "--template", str(ROOT / "docs" / "HF_MODEL_CARD_TEMPLATE.md"),
        "--out", str(out_file),
        "--dry-run",
    ])
    assert code == 0
    assert out_file.is_file()
    text = out_file.read_text(encoding="utf-8")
    assert "cli-test-run" in text
    assert "6.1" in text
