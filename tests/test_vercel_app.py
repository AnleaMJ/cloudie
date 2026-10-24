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
    response = client.get("/")
    assert response.status_code == 200
    assert "200 kt" in response.text
    assert IMD_BUTTON in response.text


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
    frame = SatelliteFrame(
        image=np.zeros((4, 4), dtype=np.float32),
        preview=np.zeros((4, 4, 3), dtype=np.uint8),
        updated_at=None,
        source_url="https://example.test/ir.jpg",
    )
    monkeypatch.setattr("cloudie.web.fetch_current_infrared", lambda: frame)
    marker = '"><script>alert(1)</script>'
    response = client.post(
        "/run",
        data={
            "action": "imd",
            "config": "config/config.yaml",
            "checkpoint": f"artifacts/{marker}.pt",
            "previous": "",
            "example": "(none)",
        },
    )
    assert "<script>" not in response.text
    assert "&lt;script&gt;" in response.text


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
