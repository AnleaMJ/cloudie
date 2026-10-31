"""Gradio Space page. Synthetic frames only. No Hub download and no imagery fetch."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from cyclone_frames import colour_enhanced, spiral_frame

ROOT = Path(__file__).resolve().parents[1]
REFUSAL = "not a tropical cyclone IR image"


@pytest.fixture(autouse=True)
def _clear_space_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLOUDIE_CHECKPOINT", raising=False)
    monkeypatch.delenv("CLOUDIE_HF_MODEL_REPO", raising=False)


def _config() -> dict:
    from src.config import load_config

    return load_config(ROOT / "config" / "config.yaml")


def _isolated(config: dict, directory: Path) -> dict:
    prepared = dict(config)
    prepared["paths"] = dict(config["paths"])
    prepared["paths"]["artifacts"] = str(directory / "artifacts")
    prepared["paths"]["run_records"] = str(directory / "records")
    prepared["paths"]["hf_model_repo"] = None
    return prepared


def _joblib(path: Path, config: dict, knots: float, **overrides) -> Path:
    import joblib
    from sklearn.linear_model import Ridge

    names = [str(name) for name in config["predict"]["sklearn_features"]]
    estimator = Ridge()
    estimator.coef_ = np.zeros(len(names))
    estimator.intercept_ = float(knots)
    estimator.n_features_in_ = len(names)
    payload = {
        "estimator": estimator,
        "feature_names": names,
        "model_name": str(config["model"]["name"]),
        "image_size": None,
        "norm_mean": 0.0,
        "norm_std": 1.0,
        "norm_source_split": "train",
        "max_missing_fraction": 0.30,
        "is_synthetic": True,
        "edges_kt": [float(edge) for edge in config["data"]["imd5_edges_kt"]],
        "class_names": [str(name) for name in config["data"]["imd5_names"]],
        "n_frames": 1,
        "honesty": "not a forecast",
        "wind_averaging": "synthetic",
    }
    payload.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)
    return path


def _held_out_record(config: dict, directory: Path, **overrides) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    body = {
        "run_id": "nio-held-out",
        "is_synthetic": False,
        "data_kind": "real",
        "model_name": str(config["model"]["name"]),
        "image_size": None,
        "synthetic": "no",
        "residual_quantiles": {
            "eval_split": "test",
            "fit_split": "val",
            "test_storms": ["BOB01"],
            "synthetic": "no",
            "levels": {
                "0.90": {
                    "nominal": 0.9,
                    "absolute_residual_quantile_kt": 12.0,
                    "n": 4,
                    "status": "within_tolerance",
                    "eval_split": "test",
                }
            },
        },
    }
    body.update(overrides)
    run_id = str(body["run_id"])
    (directory / f"{run_id}.json").write_text(json.dumps(body), encoding="utf-8")
    config["paths"]["run_records"] = str(directory)
    return run_id


def _rendered(demo) -> str:
    chunks: list[str] = []
    for block in getattr(demo, "blocks", {}).values():
        for attr in ("value", "label", "info"):
            value = getattr(block, attr, None)
            if isinstance(value, str):
                chunks.append(value)
        choices = getattr(block, "choices", None)
        if isinstance(choices, (list, tuple)):
            for choice in choices:
                if isinstance(choice, str):
                    chunks.append(choice)
                elif isinstance(choice, (list, tuple)):
                    chunks.extend(str(item) for item in choice)
    return "\n".join(chunks)


def test_space_sources_use_the_shared_banner_and_src_inference() -> None:
    source = (ROOT / "space" / "app.py").read_text(encoding="utf-8")
    assert "from cloudie" not in source
    assert "import cloudie" not in source
    assert "NO_TRAINED_MODEL_BANNER" in source
    assert "from src.inference import" in source
    assert "RunCheckpoint" in source
    assert "class CycloneImageGate" not in source
    assert "class IntensityEstimate" not in source
    assert "class RunCheckpoint" not in source
    assert "No trained model yet" not in source
    banner = (ROOT / "src" / "inference.py").read_text(encoding="utf-8")
    assert "NO_TRAINED_MODEL_BANNER" in banner
    readme = (ROOT / "space" / "README.md").read_text(encoding="utf-8")
    assert "NO_TRAINED_MODEL_BANNER" in readme
    assert "No trained model yet" not in readme
    assert (ROOT / "cloudie" / "app.py").read_text(encoding="utf-8").count("NO_TRAINED_MODEL_BANNER =") == 0


def test_readme_front_matter_is_a_free_cpu_gradio_space() -> None:
    text = (ROOT / "space" / "README.md").read_text(encoding="utf-8")
    front = yaml.safe_load(text.split("---", 2)[1])
    assert front["sdk"] == "gradio"
    assert front["app_file"] == "app.py"
    assert front["hardware"] == "cpu-basic"
    assert front["suggested_hardware"] == "cpu-basic"
    assert len(str(front["short_description"])) <= 60
    for banned in ("t4-small", "t4-medium", "a10g", "a100", "l4x1", "l40s", "zero-a10g", "rtx-pro"):
        assert banned not in text
    requirements = (ROOT / "space" / "requirements.txt").read_text(encoding="utf-8")
    assert "https://download.pytorch.org/whl/cpu" in requirements
    assert "torch==2.9.1+cpu" in requirements
    assert "torchvision==0.24.1+cpu" in requirements
    assert "gradio" not in requirements.lower()
    assert "cu12" not in requirements and "cuda" not in requirements.lower()


def test_torch_job_runs_space_tests_and_keeps_the_existing_checks() -> None:
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    workflow = yaml.safe_load(text)
    assert set(workflow["jobs"]) == {"test", "torch"}
    torch_text = text.split("  torch:", 1)[1]
    for name in (
        "tests/test_model.py",
        "tests/test_sequences.py",
        "tests/test_train.py",
        "tests/test_predict.py",
        "tests/test_space_app.py",
    ):
        assert name in torch_text
    assert "gradio==6.31.0" in torch_text
    assert "https://download.pytorch.org/whl/cpu" in torch_text
    assert "tests/test_inference.py" in torch_text
    test_job = yaml.dump(workflow["jobs"]["test"])
    assert "download.pytorch.org" not in test_job
    vercel = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "space.app" not in vercel
    assert "import torch" not in vercel


def test_sync_copies_src_config_and_cloudie_without_forking_the_gate(tmp_path: Path) -> None:
    from space.sync_bundle import SpaceBundle

    bundle = SpaceBundle.build(tmp_path / "bundle")
    inference = (bundle / "src" / "inference.py").read_text(encoding="utf-8")
    assert inference == (ROOT / "src" / "inference.py").read_text(encoding="utf-8")
    assert (bundle / "config" / "config.yaml").is_file()
    assert (bundle / "cloudie" / "samples" / "synthetic_ir.npy").is_file()
    assert not (bundle / "cloudie" / "app.py").exists()
    app_text = (bundle / "app.py").read_text(encoding="utf-8")
    assert app_text == (ROOT / "space" / "app.py").read_text(encoding="utf-8")
    assert "class CycloneImageGate" not in app_text
    assert not (bundle / "data").exists()
    with pytest.raises(ValueError):
        SpaceBundle.build(ROOT / "src" / "nested")


def test_vercel_import_does_not_load_the_space_or_torch() -> None:
    probe = (
        "import app, sys; "
        "from fastapi import FastAPI; "
        "assert isinstance(app.app, FastAPI); "
        "assert 'torch' not in sys.modules; "
        "assert 'space' not in sys.modules; "
        "assert 'space.app' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_app_is_importable_and_the_banner_is_on_the_page() -> None:
    pytest.importorskip("gradio")
    import space.app as space_app
    from cloudie.app import NO_TRAINED_MODEL_BANNER

    assert space_app.demo is not None
    assert callable(space_app.demo.launch)
    assert space_app.NO_TRAINED_MODEL_BANNER == NO_TRAINED_MODEL_BANNER
    rendered = _rendered(space_app.demo)
    assert NO_TRAINED_MODEL_BANNER in rendered
    assert "uncalibrated" in rendered
    assert "Wind interval (uncalibrated)" in rendered
    assert "synthetic_ir.npy — bundled synthetic sample" in rendered
    assert "not a landfall forecast" in rendered
    assert "Project grouping" in rendered


def test_importing_the_space_does_not_import_torch() -> None:
    pytest.importorskip("gradio")
    probe = "import space.app, sys; assert 'torch' not in sys.modules; assert space.app.demo is not None"
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_colour_frame_is_refused_with_no_wind() -> None:
    pytest.importorskip("gradio")
    from space.app import SpacePage

    view = SpacePage.view(
        colour_enhanced(48),
        None,
        "(none)",
        None,
        None,
        checkpoint=None,
    )
    assert view.show_wind is False
    assert view.wind == "—"
    assert REFUSAL in view.status
    assert "colour-enhanced" in view.status
    assert "greyscale" in view.status
    assert "uncalibrated" in view.interval_note
    assert view.interval == "—"
    assert "kt" not in view.wind


def test_no_wind_without_a_checkpoint(tmp_path: Path) -> None:
    pytest.importorskip("gradio")
    from cloudie.app import NO_TRAINED_MODEL_BANNER
    from space.app import SpacePage

    config = _isolated(_config(), tmp_path)
    view = SpacePage.view(
        spiral_frame(32),
        None,
        "(none)",
        None,
        None,
        config=config,
        checkpoint=None,
    )
    assert view.permanent_banner == NO_TRAINED_MODEL_BANNER
    assert view.show_wind is False
    assert view.wind == "—"
    assert view.category == "—"
    assert "No checkpoint" in view.status
    assert "Path:" not in view.status
    assert "uncalibrated" in view.confidence
    assert view.interval == "—"
    sample = SpacePage.view(
        None,
        None,
        "synthetic_ir.npy",
        None,
        None,
        config=config,
        root=ROOT,
        checkpoint=None,
    )
    assert sample.show_wind is False
    assert sample.wind == "—"
    assert "No checkpoint" in sample.status


def test_synthetic_checkpoint_is_flagged_and_stays_uncalibrated(tmp_path: Path) -> None:
    pytest.importorskip("gradio")
    from space.app import SpacePage

    config = _isolated(_config(), tmp_path)
    path = _joblib(tmp_path / "software-check.joblib", config, 40.0)
    view = SpacePage.view(
        spiral_frame(32),
        None,
        "(none)",
        19.81,
        85.83,
        config=config,
        checkpoint=path,
    )
    assert view.show_wind is True
    assert view.wind.startswith("40.0 kt")
    assert "software check" in view.status
    assert "uncalibrated" in view.confidence
    assert "uncalibrated" in view.interval_note
    assert view.interval == "—"
    assert "Odisha" in view.coastal
    assert "not a landfall forecast" in view.coastal
    assert "project" in view.grouping.lower()
    assert "not an official IMD class" in view.grouping
    bare = SpacePage.view(
        spiral_frame(32),
        None,
        "(none)",
        None,
        None,
        config=config,
        checkpoint=path,
    )
    assert "Odisha" not in bare.coastal
    assert "not a landfall forecast" in bare.coastal


def test_real_checkpoint_needs_a_held_out_record(tmp_path: Path) -> None:
    pytest.importorskip("gradio")
    from space.app import SpacePage

    config = _isolated(_config(), tmp_path)
    image = spiral_frame(32)
    missing = _joblib(
        tmp_path / "real-missing.joblib",
        config,
        40.0,
        is_synthetic=False,
        wind_averaging="1min",
    )
    refused = SpacePage.view(image, None, "(none)", None, None, config=config, checkpoint=missing)
    assert refused.show_wind is False
    assert refused.wind == "—"
    assert "real-data run record" in refused.status

    _held_out_record(
        config,
        tmp_path / "coverage-only",
        run_id="coverage-only",
        residual_quantiles=None,
        coverage_check={"real_data": True, "n": 12, "nominal": 0.9, "observed": 0.83},
    )
    coverage = _joblib(
        tmp_path / "coverage-only.joblib",
        config,
        40.0,
        is_synthetic=False,
        run_id="coverage-only",
        wind_averaging="1min",
    )
    still_hidden = SpacePage.view(
        image, None, "(none)", None, None, config=config, checkpoint=coverage
    )
    assert still_hidden.show_wind is False
    assert "held-out" in still_hidden.status
    assert "40.0" not in still_hidden.wind

    run_id = _held_out_record(config, tmp_path / "held-out")
    passed = _joblib(
        tmp_path / "held-out.joblib",
        config,
        40.0,
        is_synthetic=False,
        run_id=run_id,
        wind_averaging="1min",
    )
    shown = SpacePage.view(image, None, "(none)", None, None, config=config, checkpoint=passed)
    assert shown.show_wind is True
    assert shown.wind.startswith("40.0 kt")
    assert shown.interval == "—"
    assert "uncalibrated" in shown.interval_note
    assert "software check" not in shown.status


def test_hub_repo_is_fetched_only_from_config_or_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("gradio")
    import space.app as space_app

    config = _isolated(_config(), tmp_path)
    weights = _joblib(tmp_path / "hub.joblib", config, 40.0)
    calls: list[tuple[str, str]] = []

    def fake_download(repo_id: str, filename: str) -> Path:
        calls.append((repo_id, filename))
        return weights

    monkeypatch.setattr(space_app.RunCheckpoint, "download", fake_download)
    quiet = space_app.SpacePage.view(
        spiral_frame(32),
        None,
        "(none)",
        None,
        None,
        config=config,
        resolve_checkpoint=True,
    )
    assert calls == []
    assert quiet.show_wind is False

    config["paths"]["hf_model_repo"] = "Anlea/cloudie-weights"
    from_config = space_app.SpacePage.view(
        spiral_frame(32),
        None,
        "(none)",
        None,
        None,
        config=config,
        resolve_checkpoint=True,
    )
    assert ("Anlea/cloudie-weights", "checkpoint.pt") in calls
    assert from_config.show_wind is True
    assert "software check" in from_config.status

    calls.clear()
    monkeypatch.setenv("CLOUDIE_HF_MODEL_REPO", "Anlea/other-weights")
    space_app.SpacePage.view(
        spiral_frame(32),
        None,
        "(none)",
        None,
        None,
        config=config,
        resolve_checkpoint=True,
    )
    assert ("Anlea/other-weights", "checkpoint.pt") in calls


def test_hub_real_checkpoint_waits_for_the_held_out_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("gradio")
    import space.app as space_app

    config = _isolated(_config(), tmp_path)
    config["paths"]["hf_model_repo"] = "Anlea/cloudie-weights"
    weights = _joblib(
        tmp_path / "real.joblib",
        config,
        40.0,
        is_synthetic=False,
        run_id="nio-held-out",
        wind_averaging="1min",
    )
    record_path = tmp_path / "mirror" / "metrics.json"

    def fake_download(repo_id: str, filename: str) -> Path:
        assert repo_id == "Anlea/cloudie-weights"
        if filename == "checkpoint.pt":
            return weights
        if filename == "metrics.json":
            if not record_path.is_file():
                raise ValueError("missing record")
            return record_path
        raise ValueError(filename)

    monkeypatch.setattr(space_app.RunCheckpoint, "download", fake_download)
    hidden = space_app.SpacePage.view(
        spiral_frame(32),
        None,
        "(none)",
        None,
        None,
        config=config,
        resolve_checkpoint=True,
    )
    assert hidden.show_wind is False
    assert "40.0" not in hidden.wind

    _held_out_record(config, record_path.parent)
    record_path.write_text((record_path.parent / "nio-held-out.json").read_text(encoding="utf-8"), encoding="utf-8")
    config["paths"]["run_records"] = str(tmp_path / "unused-records")
    shown = space_app.SpacePage.view(
        spiral_frame(32),
        None,
        "(none)",
        None,
        None,
        config=config,
        resolve_checkpoint=True,
    )
    assert shown.show_wind is True
    assert shown.wind.startswith("40.0 kt")
    assert shown.interval == "—"
    assert "uncalibrated" in shown.interval_note
