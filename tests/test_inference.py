"""Honest intensity output. Synthetic frames only. No dataset download."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.config import load_config
from cloudie.inference import CycloneImageGate as CloudGate
from cloudie.predict import format_prediction, predict_images
from cyclone_frames import (
    clear_sky,
    cold_count,
    colour_enhanced,
    rgb_photo,
    spiral_frame,
    stratiform,
    uniform_noise,
)
from src.inference import (
    ConfidenceScore,
    CycloneImageGate,
    FrameGeometry,
    GateReport,
    IntensityEstimate,
    IntensityGrouping,
    RunRecord,
)
from src.model_access import ModelAccess

ROOT = Path(__file__).resolve().parents[1]
EDGES = (17.0, 34.0, 48.0, 64.0, 91.0)
NAMES = ("D_DD", "CS", "SCS", "VSCS", "ESCS_SUCS")
REFUSAL = "not a tropical cyclone IR image"


def _config() -> dict:
    config = load_config(ROOT / "config" / "config_smoke.yaml")
    config["data"]["image_size"] = 16
    config["model"]["mc_samples"] = 2
    config["model"]["name"] = "simple_cnn"
    return config


def _joblib(path: Path, config: dict, knots: float, **overrides) -> Path:
    import joblib
    from sklearn.linear_model import Ridge

    estimator = Ridge()
    names = [str(name) for name in config["predict"]["sklearn_features"]]
    estimator.coef_ = np.zeros(len(names))
    estimator.intercept_ = float(knots)
    estimator.n_features_in_ = len(names)
    payload = {
        "estimator": estimator,
        "feature_names": names,
        "model_name": str(config["model"]["name"]),
        "image_size": int(config["data"]["image_size"]),
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


def _record(config: dict, directory: Path, **overrides) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    body = {
        "run_id": "nio-check",
        "is_synthetic": False,
        "data_kind": "real",
        "model_name": str(config["model"]["name"]),
        "image_size": int(config["data"]["image_size"]),
        "coverage_check": {"real_data": True, "n": 12, "nominal": 0.9, "observed": 0.83},
    }
    body.update(overrides)
    run_id = str(body["run_id"])
    (directory / f"{run_id}.json").write_text(json.dumps(body), encoding="utf-8")
    config["paths"]["run_records"] = str(directory)
    return run_id


def test_gate_classes_are_the_same_object_in_both_trees() -> None:
    assert CloudGate is CycloneImageGate


def test_gate_rejects_non_cyclones_and_accepts_a_spiral(tmp_path: Path) -> None:
    config = _config()
    path = _joblib(tmp_path / "checkpoint.joblib", config, 40.0)
    cases = {
        "noise": uniform_noise(48),
        "clear sky": clear_sky(48),
        "rgb": rgb_photo(48),
        "stratiform": stratiform(48),
        "zeros": np.zeros((16, 16), dtype=np.float32),
    }
    for name, image in cases.items():
        prediction = predict_images(config, path, [image])
        text = format_prediction(prediction)
        assert prediction.wind_kt is None, name
        assert prediction.show_number is False, name
        assert REFUSAL in prediction.banner, name
        assert "wind speed (kt):" not in text, name
        assert prediction.gate_reason
    assert "colour-enhanced" in predict_images(config, path, [rgb_photo(48)]).banner

    accepted = predict_images(config, path, [spiral_frame(32)])
    assert accepted.show_number is True
    assert accepted.wind_kt == pytest.approx(40.0)
    bright = predict_images(config, path, [-spiral_frame(32)])
    assert bright.show_number is True
    gray = np.stack([spiral_frame(32)] * 3, axis=-1)
    stacked = predict_images(config, path, [gray])
    assert stacked.show_number is True
    assert stacked.wind_kt == pytest.approx(40.0)
    rgba = np.concatenate([gray, np.ones((*gray.shape[:2], 1), dtype=np.float32)], axis=-1)
    with_alpha = predict_images(config, path, [rgba])
    assert with_alpha.show_number is True
    assert with_alpha.wind_kt == pytest.approx(40.0)


def test_colour_enhanced_frames_have_no_wind(tmp_path: Path) -> None:
    """A painted spiral is still not greyscale IR, including an RGBA file."""
    config = _config()
    path = _joblib(tmp_path / "checkpoint.joblib", config, 40.0)
    painted = colour_enhanced(64)
    painted_alpha = colour_enhanced(64, alpha=True)
    channel_first = np.moveaxis(painted, -1, 0)
    for name, image in {
        "rgb": painted,
        "rgba": painted_alpha,
        "channel-first": channel_first,
    }.items():
        prediction = predict_images(config, path, [image])
        text = format_prediction(prediction)
        assert prediction.wind_kt is None, name
        assert prediction.show_number is False, name
        assert REFUSAL in prediction.banner, name
        assert "colour-enhanced" in prediction.banner, name
        assert "greyscale" in prediction.banner, name
        assert "wind speed (kt):" not in text, name
        assert prediction.present_as_forecast is False


def test_category_bins_follow_the_project_edges(tmp_path: Path) -> None:
    config = _config()
    expected = {
        16.9: "Below depression",
        17.0: "Depression / Deep Depression",
        34.0: "Cyclonic Storm",
        48.0: "Severe Cyclonic Storm",
        64.0: "Very Severe Cyclonic Storm",
        90.0: "Very Severe Cyclonic Storm",
        91.0: "Extremely Severe / Super Cyclonic Storm",
    }
    image = spiral_frame(32)
    for knots, label in expected.items():
        path = _joblib(tmp_path / f"{knots}.joblib", config, knots)
        prediction = predict_images(config, path, [image])
        assert prediction.category_label == label
        assert prediction.wind_kt == pytest.approx(knots)
        assert IntensityGrouping.code(knots, EDGES, NAMES) == (
            None if knots < 17 else prediction.project_class
        )
    assert "not an official IMD class" in IntensityGrouping.note(config)


def test_landfall_field_is_null_without_coordinates(tmp_path: Path) -> None:
    config = _config()
    path = _joblib(tmp_path / "checkpoint.joblib", config, 40.0)
    image = spiral_frame(32)
    bare = predict_images(config, path, [image])
    text = format_prediction(bare)
    assert bare.nearest_coastal_states is None
    assert bare.landfall_note is None
    assert f"{config['predict']['coastal_heading']}: null" in text
    assert config["predict"]["landfall_note"] not in text

    located = predict_images(config, path, [image], latitude=19.81, longitude=85.9)
    located_text = format_prediction(located)
    assert located.nearest_coastal_states is not None
    assert located.nearest_coastal_states[0] == "Odisha"
    assert located.landfall_note == "not a landfall forecast"
    assert "not a landfall forecast" in located_text
    assert located.present_as_forecast is False
    assert "Odisha" in located_text

    from_meta = predict_images(
        config, path, [image], metadata={"lat": 19.81, "lon": 85.9}
    )
    assert from_meta.nearest_coastal_states[0] == "Odisha"
    override = predict_images(
        config,
        path,
        [image],
        latitude=19.81,
        longitude=85.9,
        metadata={"lat": 9.93, "lon": 76.26},
    )
    assert override.nearest_coastal_states[0] == "Odisha"
    with pytest.raises(ValueError, match="together"):
        predict_images(config, path, [image], latitude=19.81)
    with pytest.raises(ValueError, match="geographic"):
        predict_images(config, path, [image], latitude=120.0, longitude=85.9)


def test_joblib_confidence_is_uncalibrated_and_a_real_record_can_label_coverage(
    tmp_path: Path,
) -> None:
    config = _config()
    image = spiral_frame(32)
    path = _joblib(tmp_path / "synthetic.joblib", config, 40.0)
    synthetic = predict_images(config, path, [image])
    text = format_prediction(synthetic)
    assert synthetic.wind_kmh == pytest.approx(40.0 * 1.852)
    assert synthetic.confidence_status == "uncalibrated"
    assert "uncalibrated" in text
    assert "wind speed (km/h):" in text
    assert config["predict"]["operational_disclaimer"] in text
    assert "no matching real-data run record" in synthetic.review
    assert synthetic.present_as_forecast is False
    share = dict(synthetic.class_probabilities)[synthetic.category_label]
    assert share == pytest.approx(1.0)
    assert share > synthetic.confidence

    claimed = _joblib(
        tmp_path / "claimed.joblib",
        config,
        40.0,
        is_synthetic=False,
        residual_quantiles={"low_kt": 35.0, "high_kt": 45.0},
    )
    refused = predict_images(config, claimed, [image])
    assert refused.wind_kt is None
    assert "no matching real-data run record" in refused.banner.lower()
    assert REFUSAL not in refused.banner

    photo = predict_images(config, claimed, [rgb_photo(48)])
    assert photo.wind_kt is None
    assert REFUSAL in photo.banner

    run_id = _record(config, tmp_path / "records")
    recorded = _joblib(
        tmp_path / "recorded.joblib",
        config,
        40.0,
        is_synthetic=False,
        run_id=run_id,
        residual_quantiles={"low_kt": 35.0, "high_kt": 45.0},
    )
    covered = predict_images(config, recorded, [image])
    assert covered.show_number is True
    assert covered.confidence_status == "real-data coverage recorded"
    assert "no matching real-data run record" not in covered.review

    still_synthetic = _joblib(
        tmp_path / "synthetic-with-record.joblib",
        config,
        40.0,
        is_synthetic=True,
        run_id=run_id,
    )
    flagged = predict_images(config, still_synthetic, [image])
    assert flagged.show_number is True
    assert flagged.confidence_status == "uncalibrated"
    assert "no matching real-data run record" in flagged.review


def test_confidence_mixes_only_the_terms_that_exist() -> None:
    config = _config()
    gate_only = ConfidenceScore.combine(config, 0.5, None, None)
    assert gate_only == pytest.approx(0.5)
    full = ConfidenceScore.combine(
        config, 1.0, 0.0, {"low_kt": 10.0, "high_kt": 10.0}
    )
    assert full == pytest.approx(1.0)
    wide = ConfidenceScore.combine(
        config, 1.0, None, {"low_kt": 0.0, "high_kt": 60.0}
    )
    # Width 60 kt on a 30 kt scale clips the interval term to 0.
    # Weights 0.4 and 0.2 remain, so the score is 0.4 / 0.6.
    assert wide == pytest.approx(0.4 / 0.6)
    assert ConfidenceScore.status(config, None) == "uncalibrated"
    assert (
        ConfidenceScore.status(
            config,
            {"coverage_check": {"real_data": True, "n": 0, "nominal": 0.9, "observed": 0.9}},
        )
        == "uncalibrated"
    )


def test_evaluate_residual_quantiles_are_half_widths_and_stay_uncalibrated() -> None:
    from src.evaluate import residual_quantile_coverage

    config = _config()
    val_truth = np.zeros(4)
    val_pred = np.array([0.0, 2.0, 4.0, 10.0])
    levels = residual_quantile_coverage(
        val_truth, val_pred, val_truth, val_pred, levels=(0.8, 0.9), tolerance=0.05
    )
    half = float(levels["0.90"]["absolute_residual_quantile_kt"])
    assert half >= float(levels["0.80"]["absolute_residual_quantile_kt"])
    assert ConfidenceScore.interval_width(levels) == pytest.approx(2.0 * half)
    nested = {
        "uncertainty": {
            "residual_quantile": {
                "eval_split": "test",
                "test_storms": [],
                "levels": levels,
            }
        }
    }
    assert ConfidenceScore.interval_width(nested) == pytest.approx(2.0 * half)
    assert ConfidenceScore.status(config, nested) == "uncalibrated"
    width = 2.0 * half
    scale = float(config["predict"]["confidence"]["interval_scale_kt"])
    interval = float(np.clip(1.0 - width / scale, 0.0, 1.0))
    expected = (0.4 * 1.0 + 0.2 * interval) / 0.6
    assert ConfidenceScore.combine(config, 1.0, None, levels) == pytest.approx(expected)

    quoted = {
        "nominal": 0.9,
        "absolute_residual_quantile_kt": 8.0,
        "n": 5,
        "status": "within_tolerance",
        "eval_split": "test",
    }
    held_out = {
        "residual_quantiles": {
            "eval_split": "test",
            "fit_split": "val",
            "test_storms": ["BOB01"],
            "synthetic": "no",
            "levels": {
                "0.80": {
                    "nominal": 0.8,
                    "absolute_residual_quantile_kt": 4.0,
                    "n": 5,
                    "status": "uncalibrated",
                    "eval_split": "test",
                },
                "0.90": quoted,
            },
        }
    }
    assert ConfidenceScore.interval_width(held_out) == pytest.approx(16.0)
    assert ConfidenceScore.status(config, held_out) == "real-data coverage recorded"
    missed = {
        "residual_quantiles": {
            **held_out["residual_quantiles"],
            "levels": {
                "0.90": {**quoted, "status": "uncalibrated"},
            },
        }
    }
    assert ConfidenceScore.status(config, missed) == "uncalibrated"
    assert ConfidenceScore.interval_width(missed) == pytest.approx(16.0)
    smoke = {
        "uncertainty": {
            "residual_quantile": {
                **held_out["residual_quantiles"],
                "synthetic": "yes",
                "test_storms": ["SMOKE"],
            }
        }
    }
    assert ConfidenceScore.status(config, smoke) == "uncalibrated"
    assert "calibrated" not in ConfidenceScore.status(config, held_out)


def test_predict_reuses_a_residual_quantile_half_width(tmp_path: Path) -> None:
    config = _config()
    image = spiral_frame(32)
    levels = {
        "0.90": {
            "nominal": 0.9,
            "absolute_residual_quantile_kt": 15.0,
            "n": 4,
            "status": "within_tolerance",
            "eval_split": "test",
        }
    }
    plain = _joblib(tmp_path / "plain.joblib", config, 40.0)
    wide = _joblib(
        tmp_path / "wide.joblib",
        config,
        40.0,
        residual_quantiles={"levels": levels},
    )
    bare = predict_images(config, plain, [image])
    scored = predict_images(config, wide, [image])
    assert bare.show_number is True
    assert scored.show_number is True
    assert scored.confidence_status == "uncalibrated"
    assert scored.confidence < bare.confidence
    text = format_prediction(scored)
    assert "(uncalibrated)" in text
    assert "real-data coverage recorded" not in text

    run_id = _record(
        config,
        tmp_path / "records",
        coverage_check=None,
        residual_quantiles={
            "eval_split": "test",
            "fit_split": "val",
            "test_storms": ["BOB01"],
            "synthetic": "no",
            "levels": levels,
        },
    )
    recorded_path = _joblib(
        tmp_path / "recorded.joblib",
        config,
        40.0,
        is_synthetic=False,
        run_id=run_id,
        residual_quantiles={"levels": levels},
    )
    recorded = predict_images(config, recorded_path, [image])
    assert recorded.show_number is True
    assert recorded.confidence_status == "real-data coverage recorded"
    assert recorded.confidence == pytest.approx(scored.confidence)


def test_native_image_size_matches_without_an_integer_cast(tmp_path: Path) -> None:
    config = _config()
    config["paths"]["run_records"] = str(tmp_path)
    body = {
        "run_id": "native-check",
        "is_synthetic": False,
        "data_kind": "real",
        "model_name": "simple_cnn",
        "image_size": "native",
    }
    (tmp_path / "native-check.json").write_text(json.dumps(body), encoding="utf-8")
    matched, record = RunRecord.load_match(
        config,
        {"run_id": "native-check", "model_name": "simple_cnn", "image_size": None},
    )
    assert matched is True
    assert record["image_size"] == "native"
    assert RunRecord.load_match(
        config,
        {"run_id": "native-check", "model_name": "simple_cnn", "image_size": 16},
    ) == (False, None)
    payload = {
        "model_name": "simple_cnn",
        "image_size": None,
        "norm_source_split": "train",
    }
    IntensityEstimate._check_identity(
        {"model": {"name": "simple_cnn"}, "data": {"image_size": "native"}},
        payload,
    )
    with pytest.raises(ValueError, match="image_size"):
        IntensityEstimate._check_identity(
            {"model": {"name": "simple_cnn"}, "data": {"image_size": 16}},
            payload,
        )


def test_run_record_rejects_a_path_and_a_synthetic_file(tmp_path: Path) -> None:
    config = _config()
    payload = {"model_name": "simple_cnn", "image_size": 16, "run_id": "../secret"}
    assert RunRecord.load_match(config, payload) == (False, None)
    config["paths"]["run_records"] = str(tmp_path)
    (tmp_path / "broken.json").write_text("{", encoding="utf-8")
    assert RunRecord.load_match(config, {**payload, "run_id": "broken"}) == (False, None)
    synthetic = {
        "run_id": "smoke-only",
        "is_synthetic": True,
        "data_kind": "real",
        "model_name": "simple_cnn",
        "image_size": 16,
    }
    (tmp_path / "smoke-only.json").write_text(json.dumps(synthetic), encoding="utf-8")
    assert RunRecord.load_match(
        config, {"run_id": "smoke-only", "model_name": "simple_cnn", "image_size": 16}
    ) == (False, None)


def test_checkpoint_identity_and_suffix(tmp_path: Path) -> None:
    config = _config()
    image = spiral_frame(32)
    wrong_name = _joblib(tmp_path / "name.joblib", config, 40.0, model_name="convlstm")
    with pytest.raises(ValueError, match="model.name"):
        predict_images(config, wrong_name, [image])
    wrong_size = _joblib(tmp_path / "size.joblib", config, 40.0, image_size=8)
    with pytest.raises(ValueError, match="image_size"):
        predict_images(config, wrong_size, [image])
    wrong_norm = _joblib(tmp_path / "norm.joblib", config, 40.0, norm_source_split="val")
    with pytest.raises(ValueError, match="train"):
        predict_images(config, wrong_norm, [image])
    other = tmp_path / "weights.pkl"
    other.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="joblib"):
        predict_images(config, other, [image])


def test_letterbox_does_not_stretch_a_square_or_a_wide_frame() -> None:
    square = spiral_frame(16)
    assert np.array_equal(FrameGeometry.pad_square(square, "median"), square)
    wide = np.arange(8 * 16, dtype=np.float32).reshape(8, 16)
    padded = FrameGeometry.pad_square(wide, "median")
    assert padded.shape == (16, 16)
    assert np.array_equal(padded[4:12, :], wide)
    assert padded[0, 0] == pytest.approx(float(np.median(wide)))
    assert padded[0, 0] != pytest.approx(float(wide[0, 0]))

    config = _config()
    report = GateReport(True, "kept", 1.0, wide)
    payload = {
        "norm_mean": 0.0,
        "norm_std": 1.0,
        "image_size": 16,
        "max_missing_fraction": 0.30,
    }
    boxed = IntensityEstimate._prepare(config, payload, [report])
    stretched_config = dict(config)
    stretched_config["data"] = dict(config["data"], resize="stretch")
    stretched = IntensityEstimate._prepare(stretched_config, payload, [report])
    assert boxed[0].shape == stretched[0].shape
    assert not np.allclose(boxed[0], stretched[0])
    with pytest.raises(ValueError, match="letterbox or stretch"):
        FrameGeometry.mode({"data": {"resize": "crop"}})


def test_uncertainty_above_the_limit_returns_no_wind(tmp_path: Path, monkeypatch) -> None:
    config = _config()
    path = _joblib(tmp_path / "checkpoint.joblib", config, 40.0)

    def _wide(*_args, **_kwargs):
        return 40.0, np.asarray([10.0, 90.0], dtype=np.float64), 80.0

    monkeypatch.setattr(IntensityEstimate, "_wind", staticmethod(_wide))
    prediction = predict_images(config, path, [spiral_frame(32)])
    text = format_prediction(prediction)
    assert prediction.wind_kt is None
    assert REFUSAL in prediction.banner
    assert "80.0 kt" in prediction.gate_reason
    assert "wind speed (kt):" not in text


def test_scale_invariance_across_nasa_frame_sizes(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from PIL import Image

    from cloudie.train import build_net

    import torch

    config = _config()
    # An untrained net can wander outside 0–200 kt. The seed keeps this a
    # scale check rather than a draw of the initial weights.
    torch.manual_seed(0)
    model = build_net(config, size_head=False)
    payload = {
        "state_dict": model.state_dict(),
        "model_name": "simple_cnn",
        "edges_kt": list(EDGES),
        "class_names": list(NAMES),
        "norm_mean": 0.0,
        "norm_std": 1.0,
        "norm_source_split": "train",
        "image_size": 16,
        "max_missing_fraction": 0.30,
        "is_synthetic": True,
        "wind_averaging": "synthetic",
        "n_frames": 1,
        "size_head": False,
        "hidden": None,
        "honesty": "not a forecast",
    }
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(payload, checkpoint)
    # 366 and 1093 are squares in the NASA infrared set. 128 and 201 sit
    # with them so the same storm is checked across that range.
    sizes = (128, 201, 366, 1093)
    frames = {size: spiral_frame(size) for size in sizes}
    counts = {size: cold_count(frame) for size, frame in frames.items()}
    assert len(set(counts.values())) == len(counts)
    winds = {}
    for size, frame in frames.items():
        prediction = predict_images(config, checkpoint, [frame])
        assert prediction.show_number is True, size
        assert prediction.wind_kt is not None, size
        winds[size] = prediction.wind_kt
    tolerance = float(config["predict"]["scale_tolerance_kt"])
    assert max(winds.values()) - min(winds.values()) < tolerance

    picture = Image.fromarray(frames[1093], mode="F")
    picture = picture.resize((366, 366), Image.BILINEAR)
    resampled = np.asarray(picture, dtype=np.float32)
    assert not np.allclose(frames[366], resampled)
    assert cold_count(frames[1093]) != cold_count(resampled)
    resampled_wind = predict_images(config, checkpoint, [resampled]).wind_kt
    assert resampled_wind is not None
    assert abs(winds[366] - resampled_wind) < tolerance


def test_model_access_points_at_the_merged_dropout_class() -> None:
    pytest.importorskip("torch")
    from cloudie.model import MCDropout as shim_dropout
    from src.model import MCDropout

    assert ModelAccess.mc_dropout() is MCDropout
    assert shim_dropout is MCDropout


def test_real_torch_checkpoint_without_a_record_has_no_wind(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from cloudie.train import build_net

    import torch

    config = _config()
    model = build_net(config, size_head=False)
    payload = {
        "state_dict": model.state_dict(),
        "model_name": "simple_cnn",
        "edges_kt": list(EDGES),
        "class_names": list(NAMES),
        "norm_mean": 0.0,
        "norm_std": 1.0,
        "norm_source_split": "train",
        "image_size": 16,
        "max_missing_fraction": 0.30,
        "is_synthetic": False,
        "n_frames": 1,
        "size_head": False,
        "hidden": None,
        "honesty": "not a forecast",
    }
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(payload, checkpoint)
    prediction = predict_images(config, checkpoint, [spiral_frame(32)])
    assert prediction.wind_kt is None
    assert prediction.show_number is False
    assert "real-data run record" in prediction.banner


def test_run_checkpoint_reads_artifacts_runs_and_withholds_wind_until_held_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real model is the directory ``paths.runs/<run_id>/``, not a loose file."""
    from src.inference import NO_TRAINED_MODEL_BANNER, RunCheckpoint

    monkeypatch.delenv("CLOUDIE_CHECKPOINT", raising=False)
    monkeypatch.delenv("CLOUDIE_HF_MODEL_REPO", raising=False)
    monkeypatch.delenv("CLOUDIE_RUN_ID", raising=False)
    config = _config()
    run_id = "nasa-check"
    config["paths"]["runs"] = str(tmp_path / "runs")
    config["paths"]["run_id"] = run_id
    config["paths"]["checkpoint_name"] = "checkpoint.joblib"
    config["paths"]["run_records"] = str(tmp_path / "unused")
    directory = tmp_path / "runs" / run_id
    weights = _joblib(
        directory / "checkpoint.joblib",
        config,
        40.0,
        is_synthetic=False,
        run_id=run_id,
        wind_averaging="1min",
    )
    metrics = {
        "run_id": run_id,
        "is_synthetic": False,
        "result": True,
        "dataset": "nasa",
        "model": "simple_cnn",
        "quote_as_result": "no",
        "test": {"n": 4, "mae_kt": 1.0, "rmse_kt": 1.0, "bias_kt": 0.0},
    }
    (directory / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    image = spiral_frame(32)
    withheld = RunCheckpoint.estimate(config, [image], root=tmp_path)
    assert withheld.wind_kt is None
    assert "held-out" in withheld.banner

    metrics["residual_quantiles"] = {
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
    }
    (directory / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    shown = RunCheckpoint.estimate(config, [image], root=tmp_path)
    assert shown.outcome == "ok"
    assert shown.wind_kt == pytest.approx(40.0)
    assert shown.is_synthetic is False
    assert weights.is_file()

    colour = RunCheckpoint.estimate(config, [colour_enhanced(32)], path=weights, resolve=False)
    assert colour.wind_kt is None
    assert REFUSAL in colour.banner
    from cloudie.app import NO_TRAINED_MODEL_BANNER as exported

    assert exported is NO_TRAINED_MODEL_BANNER
