"""Vercel loads the repository-root ASGI app. Tests do not download IMD."""

from __future__ import annotations

import ast
import io
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cloudie.imd_satellite import IMD_BUTTON, SatelliteFrame

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def client() -> TestClient:
    from app import app

    return TestClient(app)


def test_root_module_exposes_an_asgi_app() -> None:
    from app import app

    assert isinstance(app, FastAPI)
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assigned = [
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    ]
    assert "app" in assigned


def test_pyproject_declares_the_vercel_entrypoint() -> None:
    parsed = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert parsed["tool"]["vercel"]["entrypoint"] == "app:app"


def test_home_page_names_the_scale_and_the_imd_button(client: TestClient) -> None:
    from cloudie.app import NO_TRAINED_MODEL_BANNER

    response = client.get("/")
    assert response.status_code == 200
    assert "200 kt" in response.text
    assert IMD_BUTTON in response.text
    assert NO_TRAINED_MODEL_BANNER in response.text
    assert "not an official IMD class" in response.text
    assert 'value="score"' in response.text
    assert response.text.find('value="score"') < response.text.find('value="imd"')
    assert "artifacts/checkpoint.pt" not in response.text
    assert "synthetic_ir.npy" in response.text
    assert "Wind interval" in response.text
    assert "uncalibrated" in response.text


def test_default_page_hides_config_and_weights(client: TestClient) -> None:
    home = client.get("/")
    opened = client.get("/run")
    posted = client.post(
        "/run",
        data={"action": "score", "example": "synthetic_ir.npy", "previous": ""},
    )
    for response in (home, opened, posted):
        assert response.status_code == 200
        assert 'name="config"' not in response.text
        assert 'name="checkpoint"' not in response.text
        assert "Trained weights" not in response.text
    debug = client.get("/?debug=1")
    assert debug.status_code == 200
    assert 'name="config"' in debug.text
    assert 'name="checkpoint"' in debug.text
    assert "Trained weights" in debug.text


def test_get_run_shows_the_banner(client: TestClient) -> None:
    from cloudie.app import NO_TRAINED_MODEL_BANNER

    response = client.get("/run")
    assert response.status_code == 200
    assert NO_TRAINED_MODEL_BANNER in response.text
    assert "Predict intensity" in response.text
    assert "artifacts/checkpoint.pt" not in response.text


def test_missing_action_predicts_and_does_not_fetch_imd(client: TestClient, monkeypatch) -> None:
    def explode() -> None:
        raise AssertionError("Enter must not fetch IMD")

    monkeypatch.setattr("cloudie.web.fetch_current_infrared", explode)
    response = client.post("/run", data={"previous": "", "example": "(none)"})
    assert response.status_code == 200
    assert "Add an infrared frame" in response.text
    assert "No trained model yet" in response.text


def test_imd_without_a_checkpoint_prints_no_wind(client: TestClient, monkeypatch) -> None:
    preview = np.zeros((8, 8, 3), dtype=np.uint8)
    frame = SatelliteFrame(
        image=np.zeros((8, 8), dtype=np.float32),
        preview=preview,
        updated_at=None,
        source_url="https://example.test/ir.jpg",
    )
    monkeypatch.setattr("cloudie.web.fetch_current_infrared", lambda: frame)
    response = client.post(
        "/run",
        data={
            "action": "imd",
            "config": "config/config.yaml",
            "checkpoint": "artifacts/does-not-exist.pt",
            "previous": "",
            "example": "(none)",
        },
    )
    assert response.status_code == 200
    assert "No checkpoint" in response.text
    assert "data:image/jpeg;base64," in response.text
    assert "1644" not in response.text
    assert "not a forecast" in response.text.lower()


def test_predict_without_a_frame_asks_for_one(client: TestClient) -> None:
    response = client.post(
        "/run",
        data={
            "action": "score",
            "config": "config/config.yaml",
            "checkpoint": "artifacts/checkpoint.pt",
            "previous": "",
            "example": "(none)",
        },
    )
    assert response.status_code == 200
    assert "Add an infrared frame" in response.text


def test_checkpoint_path_cannot_leave_the_repository(client: TestClient, monkeypatch) -> None:
    called = {"fetch": False}

    def explode() -> None:
        called["fetch"] = True
        raise AssertionError("fetch should not run")

    monkeypatch.setattr("cloudie.web.fetch_current_infrared", explode)
    response = client.post(
        "/run",
        data={
            "action": "imd",
            "config": "config/config.yaml",
            "checkpoint": "../outside.pt",
            "previous": "",
            "example": "(none)",
        },
    )
    assert called["fetch"] is False
    assert "inside this project" in response.text
    assert "<script>" not in response.text


def test_banner_text_is_escaped(client: TestClient, monkeypatch) -> None:
    from cloudie.predict import Prediction

    def _marked(*_args, **_kwargs) -> Prediction:
        return Prediction(
            wind_kt=None,
            dropout_std_kt=None,
            project_class=None,
            present_as_forecast=False,
            show_number=False,
            banner='<script>alert(1)</script> Path: /secret/weights.pt',
            review=(),
            is_synthetic=False,
            gate_reason="not a tropical cyclone IR image",
        )

    monkeypatch.setattr("cloudie.web.predict_images", _marked)
    sample = ROOT / "cloudie" / "samples" / "synthetic_ir.npy"
    response = client.post(
        "/run",
        data={
            "action": "score",
            "previous": "",
            "example": sample.name,
        },
    )
    assert response.status_code == 200
    assert "<script>" not in response.text
    assert "&lt;script&gt;" in response.text
    assert "/secret/weights.pt" not in response.text
    assert "Path:" not in response.text


def test_importing_the_entrypoint_does_not_import_torch() -> None:
    import subprocess

    probe = (
        "import app, sys; "
        "from fastapi import FastAPI; "
        "assert isinstance(app.app, FastAPI); "
        "assert 'torch' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_missing_model_runtime_prints_no_wind(client: TestClient, monkeypatch) -> None:
    def _no_runtime(*_args, **_kwargs):
        raise ImportError("torch")

    monkeypatch.setattr("cloudie.web.predict_images", _no_runtime)
    image = np.zeros((4, 4), dtype=np.float32)
    payload = io.BytesIO()
    np.save(payload, image)
    response = client.post(
        "/run",
        data={
            "action": "score",
            "config": "config/config.yaml",
            "checkpoint": "artifacts/checkpoint.pt",
            "previous": "",
            "example": "(none)",
        },
        files={"frames": ("frame.npy", payload.getvalue(), "application/octet-stream")},
    )
    assert response.status_code == 200
    assert "cannot load the model runtime" in response.text
    assert "1644" not in response.text


def test_smoke_frame_reports_category_when_weights_exist(client: TestClient) -> None:
    pytest.importorskip("torch")
    weights = ROOT / "artifacts" / "smoke" / "checkpoint.pt"
    image = ROOT / "data" / "smoke" / "images" / "S04_02.npy"
    if not weights.is_file() or not image.is_file():
        pytest.skip("smoke checkpoint is local only")
    response = client.post(
        "/run",
        data={
            "action": "score",
            "config": "config/config_smoke.yaml",
            "checkpoint": "artifacts/smoke/checkpoint.pt",
            "previous": "",
            "example": "(none)",
        },
        files={"frames": ("S04_02.npy", image.read_bytes(), "application/octet-stream")},
    )
    assert response.status_code == 200
    assert "Very Severe Cyclonic Storm" in response.text
    assert "No checkpoint" not in response.text
    assert "1644" not in response.text
    assert "kt" in response.text


def test_sample_without_weights_hides_the_server_path(client: TestClient) -> None:
    from cloudie.app import NO_TRAINED_MODEL_BANNER

    response = client.post(
        "/run",
        data={"action": "score", "example": "synthetic_ir.npy", "previous": ""},
    )
    assert response.status_code == 200
    assert NO_TRAINED_MODEL_BANNER in response.text
    assert "No checkpoint" in response.text
    assert "Path:" not in response.text
    assert "artifacts/" not in response.text
    assert "Wind interval" in response.text
    assert "uncalibrated" in response.text


def test_gate_refusal_and_uncalibrated_label_come_from_inference(client: TestClient) -> None:
    import joblib
    from sklearn.linear_model import Ridge

    from cloudie.config import load_config

    config = load_config(ROOT / "config" / "config.yaml")
    names = [str(name) for name in config["predict"]["sklearn_features"]]
    estimator = Ridge()
    estimator.coef_ = np.zeros(len(names))
    estimator.intercept_ = 40.0
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
    path = ROOT / "artifacts" / "web-demo-gate.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)
    colour = np.zeros((32, 32, 3), dtype=np.float32)
    colour[..., 0] = 1.0
    colour[..., 2] = 0.2
    blob = io.BytesIO()
    np.save(blob, colour)
    try:
        refused = client.post(
            "/run",
            data={
                "action": "score",
                "config": "config/config.yaml",
                "checkpoint": "artifacts/web-demo-gate.joblib",
                "previous": "",
                "example": "(none)",
            },
            files={"frames": ("colour.npy", blob.getvalue(), "application/octet-stream")},
        )
        scored = client.post(
            "/run",
            data={
                "action": "score",
                "config": "config/config.yaml",
                "checkpoint": "artifacts/web-demo-gate.joblib",
                "previous": "",
                "example": "synthetic_ir.npy",
            },
        )
    finally:
        path.unlink(missing_ok=True)
    assert refused.status_code == 200
    assert "not a tropical cyclone IR image" in refused.text
    assert "uncalibrated" in refused.text
    assert "wind speed (kt)" not in refused.text.lower()
    assert scored.status_code == 200
    assert "uncalibrated" in scored.text
    assert "not an official IMD class" in scored.text
    assert "Project grouping" in scored.text
    assert "Wind interval" in scored.text
    assert "40.0 kt" in scored.text or "40.00" in scored.text
