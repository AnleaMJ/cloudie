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


def _write_run_config(tmp_path: Path, config: dict) -> Path:
    import yaml

    path = tmp_path / "run_config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _assert_run_record(run_dir: Path, checkpoint_name: str) -> dict:
    names = (
        checkpoint_name,
        "resolved_config.yaml",
        "split_manifest.csv",
        "loss_curve.csv",
        "loss_curve.png",
        "metrics.json",
    )
    for name in names:
        file_path = run_dir / name
        assert file_path.is_file(), name
        assert file_path.stat().st_size > 0, name
    return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))


def test_nasa_phase1_keeps_366_and_does_not_use_source_split(tmp_path: Path, capsys) -> None:
    import numpy as np
    import pandas as pd
    from PIL import Image

    from src.splits import storm_ids_by_split
    from src.train import assign_project_split, load_nasa_frame, refuse_mixed_resolution

    root = tmp_path / "nasa"
    rows = {
        "train": [
            ("aaa_000", "aaa", 0, 1, 30),
            ("aaa_001", "aaa", 1800, 1, 35),
            ("bbb_000", "bbb", 0, 2, 90),
            ("npf_000", "npf", 0, 2, 50),
            ("ddd_000", "ddd", 0, 1, 40),
        ],
        "test": [
            ("aaa_002", "aaa", 3600, 1, 40),
            ("ccc_000", "ccc", 0, 2, 120),
        ],
    }
    for split, items in rows.items():
        stem = "training_set" if split == "train" else "test_set"
        (root / split).mkdir(parents=True)
        pd.DataFrame(
            [item[:4] for item in items],
            columns=["Image ID", "Storm ID", "Relative Time", "Ocean"],
        ).to_csv(root / f"{stem}_features.csv", index=False)
        pd.DataFrame(
            [(item[0], item[4]) for item in items],
            columns=["Image ID", "Wind Speed"],
        ).to_csv(root / f"{stem}_labels.csv", index=False)
        for item in items:
            side = 1093 if item[0] == "ddd_000" else 366
            Image.fromarray(np.full((side, side), 128, dtype=np.uint8)).save(
                root / split / f"{item[0]}.jpg"
            )
    config = load_config(ROOT / "config" / "config.yaml")
    config["paths"]["nasa_root"] = str(root)
    frame, excluded = load_nasa_frame(config)
    assert excluded == 2
    assert "nasa excluded 2 frames" in capsys.readouterr().out
    assert set(frame["orig_size"].astype(int)) == {366}
    assert "split" not in frame.columns
    assert "source_split" in frame.columns
    assert set(frame["source_split"]) == {"train", "test"}
    assert "ddd_000" not in set(frame["sample_id"])
    assert "npf_000" not in set(frame["sample_id"])
    assigned = assign_project_split(frame, config)
    aaa = assigned.loc[assigned["storm_id"] == "aaa"]
    assert set(aaa["source_split"]) == {"train", "test"}
    assert aaa["split"].nunique() == 1
    assert (aaa["split"].astype(str) != aaa["source_split"].astype(str)).any()
    groups = storm_ids_by_split(assigned)
    assert groups["train"].isdisjoint(groups["val"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["val"].isdisjoint(groups["test"])
    mixed = pd.DataFrame({"orig_size": [366, 201], "storm_id": ["a", "b"]})
    with pytest.raises(ValueError, match="mixes resolutions"):
        refuse_mixed_resolution(mixed, "nasa")


def test_smoke_run_writes_the_record_and_loads_in_inference(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("matplotlib")
    import numpy as np
    import pandas as pd
    import torch

    from cloudie.predict import predict_images
    from src.inference import CheckpointStore, IntensityEstimate
    from src.make_smoke_data import write_smoke
    from src.model_access import ModelAccess
    from src.splits import storm_ids_by_split
    from src.train import main

    manifest = write_smoke(tmp_path / "smoke", n_storms=6, frames_per_storm=2, size=16)
    config = load_config(ROOT / "config" / "config_smoke.yaml")
    config["data"]["image_size"] = 16
    config["train"]["epochs"] = 1
    config["train"]["batch_size"] = 4
    config["paths"]["manifest"] = str(manifest)
    config["paths"]["runs"] = str(tmp_path / "runs")
    config["paths"]["artifacts"] = str(tmp_path / "artifacts")
    config["paths"]["reports"] = str(tmp_path / "reports")
    config_path = _write_run_config(tmp_path, config)

    run_dir = main(
        [
            "--config",
            str(config_path),
            "--dataset",
            "smoke",
            "--model",
            "simplecnn",
            "--max-steps",
            "2",
            "--device",
            "cpu",
        ]
    )
    metrics = _assert_run_record(run_dir, config["paths"]["checkpoint_name"])
    assert metrics["result"] is False
    assert metrics["is_synthetic"] is True
    assert metrics["quote_as_result"] == "no"
    assert metrics["dataset"] == "smoke"
    assert metrics["scored_by"] == "src.evaluate.score_winds"
    for split_name in ("val", "test"):
        for key in ("mae_kt", "rmse_kt", "bias_kt", "n"):
            assert key in metrics[split_name]
        assert metrics[split_name]["n"] > 0
    curve = pd.read_csv(run_dir / "loss_curve.csv")
    assert list(curve.columns) == ["epoch", "step", "train_loss", "val_mae"]
    assert len(curve) >= 1

    manifest_out = pd.read_csv(run_dir / "split_manifest.csv")
    groups = storm_ids_by_split(manifest_out)
    assert groups["train"] and groups["val"] and groups["test"]
    assert groups["train"].isdisjoint(groups["val"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["val"].isdisjoint(groups["test"])

    checkpoint = run_dir / config["paths"]["checkpoint_name"]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["is_synthetic"] is True
    assert payload["result"] is False
    assert payload["dataset"] == "smoke"
    assert payload["model_name"] == "simple_cnn"
    assert payload["seed"] == config["project"]["seed"]
    assert payload["wind_averaging"] == "synthetic"
    assert payload["config_hash"]
    assert payload["norm_source_split"] == "train"
    assert set(payload["split_storm_ids"]) == {"train", "val", "test"}
    saved = {name: set(payload["split_storm_ids"][name]) for name in ("train", "val", "test")}
    assert saved["train"].isdisjoint(saved["val"])
    assert saved["train"].isdisjoint(saved["test"])
    assert saved["val"].isdisjoint(saved["test"])

    resolved = load_config(run_dir / "resolved_config.yaml")
    loaded = CheckpointStore.load(checkpoint)
    model = ModelAccess.build_net(resolved, size_head=bool(loaded.get("size_head", False)))
    model.load_state_dict(loaded["state_dict"])
    assert any(layer.__class__.__name__ == "Dropout" for layer in model.modules())
    output = model(torch.zeros(1, 1, 16, 16))
    wind = output[0] if isinstance(output, tuple) else output
    assert tuple(wind.shape) == (1, 1)

    image = np.load(tmp_path / "smoke" / "images" / "S00_00.npy")
    prediction = predict_images(resolved, checkpoint, [image])
    assert prediction.present_as_forecast is False
    assert prediction.is_synthetic is True
    assert "No checkpoint was found" not in prediction.banner
    estimate = IntensityEstimate.predict(
        resolved,
        checkpoint,
        [image],
        hemisphere=None,
        latitude=None,
        longitude=None,
        metadata=None,
    )
    assert estimate.is_synthetic is True
    assert estimate.outcome != "frame_count"


def test_early_stopping_on_val_mae_ends_the_run(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("matplotlib")
    import pandas as pd

    from src.make_smoke_data import write_smoke
    from src.train import run_training

    manifest = write_smoke(tmp_path / "smoke", n_storms=6, frames_per_storm=2, size=16)
    config = load_config(ROOT / "config" / "config_smoke.yaml")
    config["data"]["image_size"] = 16
    config["train"]["epochs"] = 5
    config["train"]["patience"] = 1
    config["train"]["lr"] = 0.0
    config["train"]["weight_decay"] = 0.0
    config["train"]["batch_size"] = 4
    config["train"]["max_steps"] = None
    config["paths"]["manifest"] = str(manifest)
    config["paths"]["runs"] = str(tmp_path / "runs")
    run_dir = run_training(config, dataset="smoke", model="simplecnn", device="cpu")
    metrics = _assert_run_record(run_dir, config["paths"]["checkpoint_name"])
    assert metrics["stopped_early"] is True
    assert metrics["epochs_ran"] == 2
    assert metrics["epochs_ran"] < config["train"]["epochs"]
    assert metrics["result"] is False
    curve = pd.read_csv(run_dir / "loss_curve.csv")
    assert len(curve) == 2


def test_cuda_is_refused_without_approval(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    from src.train import main, resolve_device

    monkeypatch.delenv("CLOUDIE_ALLOW_GPU", raising=False)
    with pytest.raises(RuntimeError, match="CLOUDIE_ALLOW_GPU"):
        resolve_device("cuda")
    with pytest.raises(RuntimeError, match="CLOUDIE_ALLOW_GPU"):
        main(
            [
                "--config",
                str(ROOT / "config" / "config_smoke.yaml"),
                "--dataset",
                "smoke",
                "--device",
                "cuda",
            ]
        )
    monkeypatch.setenv("CLOUDIE_ALLOW_GPU", "1")
    import torch

    if torch.cuda.is_available():
        assert resolve_device("cuda").type == "cuda"
    else:
        with pytest.raises(RuntimeError, match="not available"):
            resolve_device("cuda")


def test_tcir_path_trains_on_a_local_fixture(tmp_path: Path) -> None:
    """CPU smoke of the real TCIR loader. The HDF5 is a local fixture, not a download."""
    pytest.importorskip("torch")
    pytest.importorskip("h5py")
    pytest.importorskip("matplotlib")
    import pandas as pd

    from src.splits import storm_ids_by_split
    from src.train import run_training
    from tests.test_tcir import write_fixture

    h5_path = tmp_path / "TCIR-TINY.h5"
    write_fixture(h5_path)
    config = load_config(ROOT / "config" / "config.yaml")
    config["data"]["image_size"] = 32
    config["train"]["epochs"] = 1
    config["train"]["batch_size"] = 4
    config["train"]["max_steps"] = 1
    config["paths"]["runs"] = str(tmp_path / "runs")
    config["paths"]["tcir_files"] = [str(h5_path)]
    config["paths"]["tcir_manifest_dir"] = str(tmp_path / "tcir")
    run_dir = run_training(config, dataset="tcir", model="simplecnn", max_steps=1, device="cpu")
    metrics = _assert_run_record(run_dir, config["paths"]["checkpoint_name"])
    assert metrics["dataset"] == "tcir"
    assert metrics["quote_as_result"] == "no"
    assert metrics["n_excluded"] > 0
    manifest = pd.read_csv(run_dir / "split_manifest.csv")
    assert "source_split" not in manifest.columns
    assert set(manifest["orig_size"].astype(int)) == {201}
    assert set(manifest["channel"]) == {"IR1"}
    groups = storm_ids_by_split(manifest)
    assert groups["train"].isdisjoint(groups["val"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["val"].isdisjoint(groups["test"])
    for split_name in ("val", "test"):
        assert metrics[split_name]["n"] > 0
