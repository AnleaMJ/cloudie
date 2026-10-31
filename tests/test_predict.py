"""Phase 4. A missing or synthetic checkpoint is not a forecast."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.config import load_config
from cloudie.predict import (
    category_confidence,
    format_prediction,
    intensity_label,
    missing_checkpoint,
    predict_images,
    review_flags,
)
from src.preprocessing import leave_unflipped, parse_image_size
from cyclone_frames import spiral_frame

ROOT = Path(__file__).resolve().parents[1]
EDGES = (17.0, 34.0, 48.0, 64.0, 91.0)
NAMES = ("D_DD", "CS", "SCS", "VSCS", "ESCS_SUCS")


def _config() -> dict:
    config = load_config(ROOT / "config" / "config_smoke.yaml")
    config["data"]["image_size"] = 16
    config["model"]["mc_samples"] = 2
    config["model"]["name"] = "simple_cnn"
    return config


def _checkpoint(path: Path, config: dict, **overrides) -> Path:
    torch = pytest.importorskip("torch")
    from cloudie.train import build_net

    model = build_net(config, size_head=bool(overrides.get("size_head", False)))
    payload = {
        "state_dict": model.state_dict(),
        "model_name": str(config["model"]["name"]),
        "dropout": float(config["model"]["dropout"]),
        "pretrained": False,
        "in_channels": 1,
        "n_classes": len(config["data"]["imd5_names"]),
        "class_names": [str(name) for name in config["data"]["imd5_names"]],
        "edges_kt": [float(edge) for edge in config["data"]["imd5_edges_kt"]],
        "norm_mean": 0.0,
        "norm_std": 1.0,
        "norm_source_split": "train",
        "image_size": parse_image_size(config["data"]["image_size"]),
        "max_missing_fraction": 0.30,
        "mean_baseline_kt": 40.0,
        "is_synthetic": True,
        "wind_averaging": "synthetic",
        "n_frames": 1,
        "size_head": False,
        "hidden": None,
        "present_as_forecast": False,
        "honesty": "not a forecast",
    }
    payload.update(overrides)
    if payload["model_name"] == "convlstm" and payload["hidden"] is None:
        payload["hidden"] = int(config["model"]["hidden"])
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def test_missing_checkpoint_has_no_number(tmp_path: Path) -> None:
    prediction = missing_checkpoint(tmp_path / "missing.pt")
    text = format_prediction(prediction)
    assert prediction.show_number is False
    assert prediction.wind_kt is None
    assert prediction.present_as_forecast is False
    assert "no checkpoint" in text.lower()
    assert "not a forecast" in text.lower()
    assert "present_as_forecast: false" in text
    assert "coastal" not in text.lower()
    assert "operational" not in text.lower()
    assert not any(line.lower().startswith("forecast") for line in text.splitlines())
    again = predict_images(_config(), tmp_path / "missing.pt", [np.zeros((8, 8))])
    assert again.wind_kt is None
    assert again.present_as_forecast is False


def test_confidence_is_the_share_of_samples_in_the_predicted_category() -> None:
    winds = [40.0, 42.0, 44.0, 55.0]
    confidence, table, below = category_confidence(winds, "CS", EDGES, NAMES)
    assert below == pytest.approx(0.0)
    assert sum(table.values()) == pytest.approx(1.0)
    assert table["CS"] == pytest.approx(0.75)
    assert table["SCS"] == pytest.approx(0.25)
    assert confidence == pytest.approx(0.75)
    below_winds = [5.0, 8.0, 10.0, 40.0]
    below, below_table, below_share = category_confidence(below_winds, None, EDGES, NAMES)
    assert below == pytest.approx(0.75)
    assert below_share == pytest.approx(0.75)
    assert below == pytest.approx(1.0 - sum(below_table.values()))
    impossible = [40.0, 42.0, 1600.0, 1800.0]
    score, extreme, _ = category_confidence(impossible, "CS", EDGES, NAMES)
    assert score == pytest.approx(0.5)
    assert extreme["ESCS_SUCS"] == pytest.approx(0.0)
    assert extreme["CS"] == pytest.approx(0.5)
    assert intensity_label("VSCS") == "Very Severe Cyclonic Storm"
    assert intensity_label(None) == "Below depression"
    with pytest.raises(ValueError, match="wind samples"):
        category_confidence([np.nan], "CS", EDGES, NAMES)
    with pytest.raises(ValueError, match="wind samples"):
        category_confidence([], "CS", EDGES, NAMES)


def _force_wind(state: dict, knots: float) -> None:
    for key, value in state.items():
        if not str(key).endswith("bias") or value.ndim != 1 or value.numel() != 1:
            continue
        if "class_head" in str(key) or "size_head" in str(key):
            continue
        value.fill_(float(knots))
        weight = str(key)[: -len("bias")] + "weight"
        if weight in state:
            state[weight].zero_()
        return
    raise AssertionError("wind head not found")


def test_review_flags_follow_the_configured_jump_and_withhold_classes() -> None:
    jump = review_flags(
        70.0,
        1.0,
        50.0,
        edges=EDGES,
        names=NAMES,
        wind_averaging="3-min",
        advisory_jump_kt=20.0,
    )
    assert "jump versus the last advisory" in jump
    assert "category change" in jump
    quiet = review_flags(
        55.0,
        1.0,
        50.0,
        edges=EDGES,
        names=NAMES,
        wind_averaging="3-min",
        advisory_jump_kt=20.0,
    )
    assert quiet == ()
    no_period = review_flags(
        80.0,
        1.0,
        40.0,
        edges=EDGES,
        names=NAMES,
        wind_averaging=None,
        advisory_jump_kt=20.0,
    )
    assert "jump versus the last advisory" in no_period
    assert "category change" not in no_period
    spread = review_flags(
        60.0,
        40.0,
        None,
        edges=EDGES,
        names=NAMES,
        wind_averaging="3-min",
        advisory_jump_kt=20.0,
    )
    assert spread == ("dropout spread crosses a project class boundary",)
    gate = review_flags(
        None,
        None,
        40.0,
        edges=EDGES,
        names=NAMES,
        wind_averaging="3-min",
        advisory_jump_kt=20.0,
        failed_gate=True,
    )
    assert gate == ("missing-pixel gate",)


def test_southern_hemisphere_frame_is_not_flipped() -> None:
    image = np.arange(16, dtype=np.float32).reshape(4, 4)
    stored = leave_unflipped(image, "SH")
    assert np.array_equal(stored, image)
    assert not np.array_equal(stored, np.fliplr(image))


def test_smoke_checkpoint_is_not_a_forecast(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    config = _config()
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", config, present_as_forecast=True)
    torch = pytest.importorskip("torch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _force_wind(payload["state_dict"], 40.0)
    torch.save(payload, checkpoint)
    image = spiral_frame(16)
    first = predict_images(config, checkpoint, [image], previous_advisory_kt=30.0)
    second = predict_images(config, checkpoint, [image], previous_advisory_kt=90.0)
    assert first.show_number is True
    assert first.present_as_forecast is False
    assert first.is_synthetic is True
    assert first.wind_kt == pytest.approx(40.0)
    assert first.wind_kt == pytest.approx(second.wind_kt)
    assert first.confidence == pytest.approx(second.confidence)
    assert first.category_label == "Cyclonic Storm"
    assert first.confidence is not None
    assert 0.0 <= first.confidence <= 1.0
    matched = dict(first.class_probabilities)
    share = matched[first.category_label]
    assert share == pytest.approx(1.0)
    assert share > first.confidence
    assert first.confidence_status == config["predict"]["confidence"]["uncalibrated_label"]
    assert first.wind_kmh == pytest.approx(first.wind_kt * float(config["data"]["kt_to_kmh"]))
    assert first.nearest_coastal_states is None
    assert first.landfall_note is None
    assert sum(prob for _label, prob in first.class_probabilities) == pytest.approx(1.0)
    text = format_prediction(first)
    assert "wind speed (kt):" in text
    assert "wind speed (km/h):" in text
    assert "intensity category:" in text
    assert "confidence:" in text
    assert "uncalibrated" in text
    assert "not an official IMD class" in text
    assert config["predict"]["operational_disclaimer"] in text
    assert f"{config['predict']['coastal_heading']}: null" in text
    assert "do not report" in text.lower()
    assert not any(line.lower().startswith("forecast") for line in text.splitlines())


def test_missing_pixels_refuse_a_number(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    config = _config()
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", config)
    image = np.full((16, 16), np.nan, dtype=np.float32)
    image[0, 0] = 1.0
    prediction = predict_images(config, checkpoint, [image], previous_advisory_kt=40.0)
    assert prediction.show_number is False
    assert prediction.wind_kt is None
    assert prediction.present_as_forecast is False
    assert prediction.review == ("missing-pixel gate",)


def test_category_follows_predicted_wind_without_an_averaging_period(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    config = _config()
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", config, wind_averaging=None)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _force_wind(payload["state_dict"], 40.0)
    torch.save(payload, checkpoint)
    image = spiral_frame(16)
    prediction = predict_images(config, checkpoint, [image], previous_advisory_kt=10.0)
    assert prediction.wind_kt == pytest.approx(40.0)
    assert prediction.category_label == "Cyclonic Storm"
    assert prediction.confidence is not None
    assert "category change" not in prediction.review


def test_display_counts_outside_training_have_no_wind(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    config = _config()
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _force_wind(payload["state_dict"], 40.0)
    torch.save(payload, checkpoint)
    image = spiral_frame(16) + 80.0
    prediction = predict_images(config, checkpoint, [image])
    text = format_prediction(prediction)
    assert prediction.show_number is False
    assert prediction.wind_kt is None
    assert prediction.confidence is None
    assert "1600" not in text
    assert "80" not in text.split("Reported")[0]
    assert "0" in text and "200" in text
    assert "super cyclonic" in text.lower()
    assert "trained on" in text.lower()


def test_wind_above_200_kt_has_no_number(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    config = _config()
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _force_wind(payload["state_dict"], 1644.7)
    torch.save(payload, checkpoint)
    image = spiral_frame(16)
    prediction = predict_images(config, checkpoint, [image])
    text = format_prediction(prediction)
    assert prediction.show_number is False
    assert prediction.wind_kt is None
    assert prediction.confidence is None
    assert "1644" not in text
    assert "200" in text
    assert "super cyclonic" in text.lower()


def test_wrong_frame_count_has_no_number(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    config = _config()
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", config, n_frames=4, model_name="simple_cnn")
    # model_name in the payload must still match the config, so a 4-frame
    # simple_cnn checkpoint is only a count check before the forward pass.
    image = np.ones((16, 16), dtype=np.float32)
    prediction = predict_images(config, checkpoint, [image])
    assert prediction.show_number is False
    assert prediction.wind_kt is None
    assert "4" in prediction.banner
    assert "not a forecast" in prediction.banner.lower()


def test_app_module_imports_without_streamlit_side_effects() -> None:
    import cloudie.app as app

    assert "wind speed" in app.PAGE_LEAD.lower()
    assert "intensity category" in app.PAGE_LEAD.lower()


@pytest.mark.parametrize("image_size", [128, 201, 366])
def test_native_checkpoint_accepts_chip_sizes(tmp_path: Path, image_size: int) -> None:
    pytest.importorskip("torch")
    config = _config()
    config["data"]["image_size"] = "native"
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", config)
    image = np.zeros((image_size, image_size), dtype=np.float32)
    prediction = predict_images(config, checkpoint, [image])
    assert prediction.present_as_forecast is False
    assert prediction.is_synthetic is True


def test_native_config_refuses_a_resized_checkpoint(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    config = _config()
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", config)
    config["data"]["image_size"] = "native"
    image = np.zeros((16, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="image_size"):
        predict_images(config, checkpoint, [image])


def test_streamlit_page_refuses_a_missing_checkpoint() -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    page = AppTest.from_file(str(ROOT / "cloudie" / "app.py"), default_timeout=30)
    page.run()
    visible = " ".join(getattr(item, "value", "") for item in page.error)
    assert "no checkpoint" in visible.lower()
    assert "not a forecast" in visible.lower()
    labels = [getattr(item, "label", "") for item in page.metric]
    assert "Forecast" not in labels
