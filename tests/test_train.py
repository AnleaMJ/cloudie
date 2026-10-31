"""Phase 3 training rules. The smoke fit is a software check, not a score."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import load_config
from src.splits import LeakageError
from cloudie.train import EarlyStopping, assert_model_inputs, class_sample_weights

ROOT = Path(__file__).resolve().parents[1]


def test_early_stopping_uses_patience() -> None:
    stopper = EarlyStopping(2)
    state = {"step": 1}
    assert stopper.update(5.0, state) is False
    assert stopper.update(4.0, state) is False
    assert stopper.best_state == state
    assert stopper.update(4.0, state) is False
    assert stopper.update(4.5, state) is True
    assert stopper.best == pytest.approx(4.0)
    with pytest.raises(ValueError, match="patience"):
        EarlyStopping(0)


def test_class_weights_are_fit_on_the_given_rows_only() -> None:
    sample, weights = class_sample_weights(["D_DD", "D_DD", "CS"], ["D_DD", "CS", "SCS"])
    assert weights[2] == 0.0
    assert weights[0] == pytest.approx(0.75)
    assert weights[1] == pytest.approx(1.5)
    assert sample.tolist() == pytest.approx([0.75, 0.75, 1.5])
    held_out = class_sample_weights(["D_DD"], ["D_DD", "CS"])
    assert held_out[1][1] == 0.0


def test_wind_and_size_are_not_model_inputs() -> None:
    assert_model_inputs(("image",))
    with pytest.raises(LeakageError, match="wind_speed_kt"):
        assert_model_inputs(["image", "wind_speed_kt"])
    with pytest.raises(LeakageError, match="r34_nmi"):
        assert_model_inputs(["r34_nmi"])


def test_smoke_notebook_says_the_images_are_synthetic() -> None:
    notebook = json.loads((ROOT / "notebooks" / "smoke_train.ipynb").read_text(encoding="utf-8"))
    text = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown"
    )
    lowered = text.lower()
    assert "synthetic" in lowered
    assert "not cyclones" in lowered
    assert "do not report" in lowered


def test_manifest_wind_period_must_match_config(tmp_path: Path) -> None:
    from src.make_smoke_data import write_smoke
    from cloudie.train import train

    config = load_config(ROOT / "config" / "config_smoke.yaml")
    config["data"]["wind_averaging"] = "1min"
    manifest = write_smoke(tmp_path / "smoke", size=8)
    config["paths"]["manifest"] = str(manifest)
    with pytest.raises(ValueError, match="wind_averaging"):
        train(config, manifest)


def test_missing_checkpoint_is_not_a_forecast(tmp_path: Path) -> None:
    from cloudie.evaluate import evaluate

    config = load_config(ROOT / "config" / "config_smoke.yaml")
    config["paths"]["manifest"] = str(tmp_path / "manifest.csv")
    with pytest.raises(FileNotFoundError, match="not a forecast"):
        evaluate(config, tmp_path / "missing.pt")


def test_pretrained_download_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    from cloudie.train import build_backbone

    monkeypatch.delenv("CLOUDIE_ALLOW_PRETRAINED", raising=False)
    config = load_config(ROOT / "config" / "config.yaml")
    config["model"]["pretrained"] = True
    config["model"]["name"] = "resnet18"
    with pytest.raises(RuntimeError, match="ImageNet"):
        build_backbone(config)


@pytest.mark.parametrize("image_size", [32, "native"])
def test_smoke_train_and_evaluate_are_not_a_forecast(
    tmp_path: Path, image_size: int | str
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("matplotlib")
    from cloudie.evaluate import evaluate
    from src.make_smoke_data import write_smoke
    from cloudie.train import train

    chip = 16 if image_size == "native" else int(image_size)
    config = load_config(ROOT / "config" / "config_smoke.yaml")
    config["data"]["image_size"] = image_size
    config["train"]["epochs"] = 1
    config["train"]["batch_size"] = 8
    config["paths"]["artifacts"] = str(tmp_path / "artifacts")
    config["paths"]["reports"] = str(tmp_path / "reports")
    manifest = write_smoke(tmp_path / "smoke", size=chip)
    config["paths"]["manifest"] = str(manifest)

    checkpoint = train(config, manifest)
    assert checkpoint.is_file()

    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["is_synthetic"] is True
    assert payload["present_as_forecast"] is False
    assert payload["pretrained"] is False
    assert payload["size_head"] is False
    assert payload["n_frames"] == 1
    assert payload["norm_source_split"] == "train"
    assert payload["wind_averaging"] == "synthetic"
    assert payload["image_size"] == (None if image_size == "native" else chip)

    result = evaluate(config, checkpoint)
    assert result.is_synthetic is True
    assert result.present_as_forecast is False
    assert "do not report" in result.note.lower()
    assert result.model["n"] > 0
    assert result.mean_baseline["n"] == result.model["n"]
    assert result.operational_baselines == {"ADT": None, "SATCON": None, "Dvorak": None}
    assert result.by_class
    assert result.residual_plot.is_file()
    assert result.residual_plot.stat().st_size > 0
    report = json.loads((tmp_path / "reports" / "metrics.json").read_text(encoding="utf-8"))
    assert report["present_as_forecast"] is False
    assert "synthetic" in report["note"].lower()
