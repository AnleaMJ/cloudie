"""Device, cache, normalization, resume, and fine-tune checks. CPU only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.preprocessing import NormStats, norm_by_dataset, norm_stats_from_checkpoint
from src.schema import split_wind_averaging, wind_averaging_seconds

ROOT = Path(__file__).resolve().parents[1]


def test_wind_averaging_seconds_follow_the_label() -> None:
    assert wind_averaging_seconds("1min") == 60
    assert wind_averaging_seconds("3-min") == 180
    assert wind_averaging_seconds("10min") == 600
    assert wind_averaging_seconds(" 10-minute ") == 600
    assert wind_averaging_seconds("synthetic") is None
    frame = pd.DataFrame(
        {
            "split": ["train", "val", "test"],
            "wind_averaging": ["1min", "1min", "1min"],
        }
    )
    report = split_wind_averaging(frame)
    assert report["train"]["wind_averaging"] == ["1min"]
    assert report["train"]["wind_averaging_s"] == [60]
    assert report["val"]["wind_averaging_s"] == [60]
    assert report["test"]["wind_averaging_s"] == [60]


def _write_constant(root: Path, sample_id: str, value: float, *, nan: bool = False) -> str:
    folder = root / "images"
    folder.mkdir(parents=True, exist_ok=True)
    relative = f"images/{sample_id}.npy"
    array = np.full((8, 8), value, dtype=np.float32)
    if nan:
        array[0, 0] = np.nan
    np.save(root / relative, array)
    return relative


def test_norm_uses_train_finite_pixels_and_stays_per_dataset(tmp_path: Path) -> None:
    from src.inference import GateReport, IntensityEstimate
    from src.train import fit_split_normalization

    rows = []
    plan = [
        ("tr0", "T0", "train", 2.0, True),
        ("tr1", "T1", "train", 2.0, False),
        ("va0", "V0", "val", 100.0, False),
        ("te0", "X0", "test", -40.0, False),
    ]
    for sample_id, storm_id, split, pixel, nan in plan:
        rows.append(
            {
                "sample_id": sample_id,
                "image_path": _write_constant(tmp_path, sample_id, pixel, nan=nan),
                "storm_id": storm_id,
                "split": split,
                "wind_speed_kt": 35.0,
                "_class": "CS",
                "wind_averaging": "1min",
                "is_synthetic": True,
            }
        )
    frame = pd.DataFrame(rows)
    manifest = tmp_path / "manifest.csv"
    frame.to_csv(manifest, index=False)
    config = {
        "data": {"image_size": 8, "max_missing_fraction": 0.30, "dataset": "nasa"},
        "model": {"name": "simple_cnn", "n_frames": 4},
    }
    from src.preprocessing import fit_norm_stats

    stats = fit_split_normalization(frame, manifest, config)
    assert stats.source_split == "train"
    assert stats.mean == pytest.approx(2.0)
    assert stats.std == pytest.approx(1.0)
    dirty = np.full((1, 8, 8), 2.0, dtype=np.float32)
    dirty[0, 0, 0] = np.nan
    finite_only = fit_norm_stats([dirty], "train")
    assert finite_only.mean == pytest.approx(2.0)
    np.save(tmp_path / "images" / "va0.npy", np.full((8, 8), 250.0, dtype=np.float32))
    np.save(tmp_path / "images" / "te0.npy", np.full((8, 8), -80.0, dtype=np.float32))
    again = fit_split_normalization(frame, manifest, config)
    assert again.mean == pytest.approx(stats.mean)
    assert again.std == pytest.approx(stats.std)

    nasa = norm_by_dataset("nasa", stats)
    tcir = norm_by_dataset("tcir", NormStats(mean=220.0, std=12.0, source_split="train"))
    assert set(nasa) == {"nasa"}
    assert nasa["nasa"]["pixel_kind"] == "uint8"
    assert nasa["nasa"]["source_split"] == "train"
    assert tcir["tcir"]["pixel_kind"] == "brightness_temperature_k"
    assert tcir["tcir"]["mean"] == pytest.approx(220.0)
    mixed = {
        "dataset": "nasa",
        "norm_mean": 2.0,
        "norm_std": 1.0,
        "norm_source_split": "train",
        "norm_by_dataset": {**nasa, **tcir},
    }
    with pytest.raises(ValueError, match="mix"):
        norm_stats_from_checkpoint(mixed)
    single = {
        "dataset": "nasa",
        "norm_source_split": "train",
        "norm_by_dataset": nasa,
    }
    loaded = norm_stats_from_checkpoint(single)
    assert loaded.mean == pytest.approx(2.0)
    assert loaded.std == pytest.approx(1.0)

    smoke_config = load_config(ROOT / "config" / "config_smoke.yaml")
    smoke_config["data"]["image_size"] = 8
    smoke_config["data"]["resize"] = "stretch"
    field = np.full((8, 8), 4.0, dtype=np.float32)
    report = GateReport(accepted=True, reason="", score=1.0, field=field)
    payload = {
        "dataset": "nasa",
        "image_size": 8,
        "max_missing_fraction": 0.30,
        "norm_mean": 0.0,
        "norm_std": 1.0,
        "norm_source_split": "train",
        "norm_by_dataset": {
            "nasa": {
                "mean": 4.0,
                "std": 2.0,
                "source_split": "train",
                "pixel_kind": "uint8",
            }
        },
    }
    prepared = IntensityEstimate._prepare(smoke_config, payload, [report])
    assert float(np.mean(prepared[0])) == pytest.approx(0.0)
    payload["norm_by_dataset"]["tcir"] = {
        "mean": 200.0,
        "std": 1.0,
        "source_split": "train",
        "pixel_kind": "brightness_temperature_k",
    }
    with pytest.raises(ValueError, match="mix"):
        IntensityEstimate._prepare(smoke_config, payload, [report])


def test_frame_cache_writes_fp16_and_training_can_see_it(tmp_path: Path) -> None:
    from src.cache_frames import cache_dataset
    from src.data_loader import apply_frame_cache, load_manifest
    from src.make_smoke_data import write_smoke

    manifest = write_smoke(tmp_path / "smoke", n_storms=4, frames_per_storm=1, size=8)
    config = load_config(ROOT / "config" / "config_smoke.yaml")
    config["paths"]["manifest"] = str(manifest)
    config["paths"]["frame_cache"] = str(tmp_path / "cache")
    index = cache_dataset("smoke", config)
    table = pd.read_csv(index)
    assert len(table) == 4
    assert set(table["dtype"]) == {"float16"}
    assert set(table["dataset"]) == {"smoke"}
    cached_array = np.load(table["cache_path"].iloc[0])
    assert cached_array.dtype == np.float16
    frame = load_manifest(manifest)
    remapped = apply_frame_cache(frame, "smoke", config["paths"]["frame_cache"])
    assert all(str(path).endswith(".npy") for path in remapped["image_path"])
    assert Path(remapped["image_path"].iloc[0]).is_file()
    untouched = apply_frame_cache(frame, "smoke", tmp_path / "missing-cache")
    assert untouched["image_path"].tolist() == frame["image_path"].tolist()


def test_parity_tolerance_scales_with_the_cpu_magnitude(capsys) -> None:
    from src.device import parity_verdict, warn_if_parity_failed

    tight = parity_verdict(0.2, 0.0, 0.05, 1.0)
    assert tight["ok"] is False
    assert tight["max_abs_diff"] == pytest.approx(0.2)
    assert tight["limit"] == pytest.approx(0.05)
    wide = parity_verdict(0.2, 0.01, 0.05, 10.0)
    assert wide["ok"] is True
    assert wide["limit"] == pytest.approx(0.5)
    exact = parity_verdict(0.0, 0.0, 0.05, 0.1)
    assert exact["ok"] is True
    assert exact["reference_scale"] == pytest.approx(1.0)
    warn_if_parity_failed(tight)
    assert "WARNING: device parity check failed" in capsys.readouterr().out
    with pytest.raises(ValueError, match="device_parity_tol"):
        parity_verdict(0.0, 0.0, -1.0, 1.0)


def test_amp_policy_is_bf16_on_xpu_fp16_scaler_on_cuda_and_fp32_on_cpu() -> None:
    from src.device import amp_policy_for, xpu_native_bf16

    class _Dev:
        def __init__(self, kind: str) -> None:
            self.type = kind

    class _NativeXpu:
        def __init__(self, native: bool) -> None:
            self.native = native
            self.flag = None

        def is_bf16_supported(self, including_emulation: bool = True) -> bool:
            self.flag = including_emulation
            return bool(self.native) and including_emulation is False

    class _OldXpu:
        def __init__(self) -> None:
            self.called = False

        def is_bf16_supported(self) -> bool:
            self.called = True
            return True

    class _Torch:
        float16 = "f16"
        bfloat16 = "bf16"

        def __init__(self, xpu) -> None:
            self.xpu = xpu

    native = _NativeXpu(True)
    xpu = amp_policy_for(_Dev("xpu"), _Torch(native))
    assert native.flag is False
    assert xpu.precision == "bf16"
    assert xpu.autocast_device == "xpu"
    assert xpu.dtype == "bf16"
    assert xpu.use_grad_scaler is False
    emulated = _NativeXpu(False)
    fallen = amp_policy_for(_Dev("xpu"), _Torch(emulated))
    assert fallen.precision == "fp32"
    assert fallen.autocast_device is None
    assert fallen.use_grad_scaler is False
    old = _OldXpu()
    legacy = amp_policy_for(_Dev("xpu"), _Torch(old))
    assert old.called is True
    assert legacy.precision == "bf16"
    assert xpu_native_bf16(_Torch(None)) is False
    cuda = amp_policy_for(_Dev("cuda"), _Torch(None))
    assert cuda.precision == "fp16"
    assert cuda.autocast_device == "cuda"
    assert cuda.dtype == "f16"
    assert cuda.use_grad_scaler is True
    cpu = amp_policy_for(_Dev("cpu"), _Torch(None))
    assert cpu.precision == "fp32"
    assert cpu.use_grad_scaler is False


def test_device_gate_and_auto_order(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    pytest.importorskip("torch")
    from src.device import resolve_device
    import src.device as device_mod

    monkeypatch.delenv("CLOUDIE_ALLOW_GPU", raising=False)
    with pytest.raises(RuntimeError, match="CLOUDIE_ALLOW_GPU"):
        resolve_device("xpu")
    with pytest.raises(RuntimeError, match="CLOUDIE_ALLOW_GPU"):
        resolve_device("cuda")
    monkeypatch.setattr(
        device_mod, "accelerator_available", lambda _torch, name: name in {"xpu", "cuda", "cpu"}
    )
    fallen = resolve_device("auto")
    assert fallen.type == "cpu"
    text = capsys.readouterr().out
    assert "Falling back to cpu" in text
    assert "xpu" in text
    monkeypatch.setenv("CLOUDIE_ALLOW_GPU", "1")
    assert resolve_device("auto").type == "xpu"
    monkeypatch.setattr(
        device_mod, "accelerator_available", lambda _torch, name: name in {"cuda", "cpu"}
    )
    assert resolve_device("auto").type == "cuda"
    monkeypatch.setattr(device_mod, "accelerator_available", lambda _torch, name: name == "cpu")
    assert resolve_device("auto").type == "cpu"
    with pytest.raises(RuntimeError, match="not available"):
        resolve_device("xpu")
    with pytest.raises(RuntimeError, match="not available"):
        resolve_device("cuda")


def test_grad_scaler_is_built_only_for_cuda_fp16() -> None:
    pytest.importorskip("torch")
    import warnings

    import torch

    from src.device import AmpPolicy, amp_policy_for, make_grad_scaler

    cpu = amp_policy_for(torch.device("cpu"))
    assert cpu.precision == "fp32"
    assert make_grad_scaler(cpu) is None
    xpu = AmpPolicy("bf16", "xpu", torch.bfloat16, False)
    assert make_grad_scaler(xpu) is None
    with pytest.raises(RuntimeError, match="cuda fp16"):
        make_grad_scaler(AmpPolicy("bf16", "xpu", torch.bfloat16, True))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        scaler = make_grad_scaler(AmpPolicy("fp16", "cuda", torch.float16, True))
    assert scaler is not None
    assert type(scaler).__name__ == "GradScaler"


def test_cpu_forward_parity_is_exact() -> None:
    pytest.importorskip("torch")
    import torch

    from src.device import compare_model_parity, seeded_parity_batch
    from src.train import build_net

    config = load_config(ROOT / "config" / "config_smoke.yaml")
    config["data"]["image_size"] = 8
    config["model"]["dropout"] = 0.0
    model = build_net(config, size_head=False)
    images, winds = seeded_parity_batch(2, 1, 8, seed=0)
    result = compare_model_parity(model, images, winds, torch.device("cpu"), 0.05)
    assert result["ok"] is True
    assert result["max_abs_diff"] == pytest.approx(0.0)


def test_loader_workers_persist_only_when_used() -> None:
    pytest.importorskip("torch")
    from src.train import ArrayFrames, build_loader

    images = np.zeros((2, 1, 8, 8), dtype=np.float32)
    dataset = ArrayFrames(images, np.zeros(2), [None, None], ["D_DD"], np.ones(2))
    loader = build_loader(dataset, batch_size=1, shuffle=False, num_workers=2)
    idle = build_loader(dataset, batch_size=1, shuffle=False, num_workers=0)
    assert loader.num_workers == 2
    assert loader.persistent_workers is True
    assert idle.num_workers == 0
    assert idle.persistent_workers is False


def test_grad_accumulation_steps_once_per_group(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    import torch

    from src.train import ArrayFrames, _fit_run, build_loader, build_net

    calls = {"n": 0}
    real_step = torch.optim.AdamW.step

    def _counted(self, *args, **kwargs):
        calls["n"] += 1
        return real_step(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", _counted)
    config = load_config(ROOT / "config" / "config_smoke.yaml")
    config["model"]["dropout"] = 0.0
    config["train"]["epochs"] = 1
    config["train"]["patience"] = 5
    config["train"]["lr"] = 0.0
    config["train"]["weight_decay"] = 0.0
    config["train"]["grad_accum_steps"] = 2
    images = np.zeros((4, 1, 8, 8), dtype=np.float32)
    names = [str(name) for name in config["data"]["imd5_names"]]
    dataset = ArrayFrames(images, np.ones(4, dtype=np.float32), [None] * 4, names, np.ones(4))
    train_loader = build_loader(dataset, batch_size=1, shuffle=False, num_workers=0)
    val_loader = build_loader(dataset, batch_size=2, shuffle=False, num_workers=0)
    model = build_net(config, size_head=False)
    _fit_run(model, train_loader, val_loader, config["train"], torch.device("cpu"), None)
    assert calls["n"] == 2
    calls["n"] = 0
    config["train"]["grad_accum_steps"] = 1
    _fit_run(model, train_loader, val_loader, config["train"], torch.device("cpu"), None)
    assert calls["n"] == 4


def _runtime_config(tmp_path: Path, manifest: Path) -> dict:
    config = load_config(ROOT / "config" / "config_smoke.yaml")
    config["data"]["image_size"] = 16
    config["train"]["epochs"] = 1
    config["train"]["batch_size"] = 4
    config["train"]["patience"] = 10
    config["train"]["num_workers"] = 0
    config["train"]["lr"] = 0.0
    config["train"]["weight_decay"] = 0.0
    config["train"]["max_steps"] = None
    config["paths"]["manifest"] = str(manifest)
    config["paths"]["runs"] = str(tmp_path / "runs")
    config["paths"]["frame_cache"] = str(tmp_path / "cache")
    return config


def _save_parent(path: Path, config: dict, wind: str) -> None:
    import torch

    from src.train import build_net

    model = build_net(config, size_head=False)
    with torch.no_grad():
        for tensor in model.parameters():
            tensor.fill_(0.125)
    seconds = 60 if wind == "1min" else None
    payload = {
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "model_name": "simple_cnn",
        "run_id": "parent-nasa",
        "dataset": "nasa",
        "wind_averaging": wind,
        "wind_averaging_s": seconds,
        "wind_averaging_by_split": {
            name: {"wind_averaging": [wind], "wind_averaging_s": [seconds]}
            for name in ("train", "val", "test")
        },
        "norm_mean": 999.0,
        "norm_std": 1.0,
        "norm_source_split": "train",
        "norm_by_dataset": {
            "nasa": {
                "mean": 999.0,
                "std": 1.0,
                "source_split": "train",
                "pixel_kind": "uint8",
            }
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def test_init_from_loads_weights_refits_norm_and_warns(tmp_path: Path, capsys) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("matplotlib")
    import torch

    from src.data_loader import load_manifest, resolve_image
    from src.train import assign_project_split, run_training
    from src.make_smoke_data import write_smoke

    manifest = write_smoke(tmp_path / "smoke", n_storms=6, frames_per_storm=2, size=16)
    config = _runtime_config(tmp_path, manifest)
    parent = tmp_path / "parent.pt"
    _save_parent(parent, config, "1min")
    frame = load_manifest(manifest)
    if "split" in frame.columns:
        frame = frame.drop(columns=["split"])
    assigned = assign_project_split(frame, config)
    for row in assigned.itertuples(index=False):
        value = 4.0 if row.split == "train" else 80.0
        image_path = resolve_image(manifest, row.image_path)
        image = np.load(image_path)
        np.save(image_path, np.full(image.shape, value, dtype=np.float32))
    run_dir = run_training(
        config,
        dataset="smoke",
        model="simplecnn",
        max_steps=1,
        device="cpu",
        init_from=parent,
    )
    text = capsys.readouterr().out
    assert "WARNING: wind_averaging mismatch" in text
    assert "No conversion factor" in text
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["wind_averaging_mismatch"] is True
    assert metrics["parent_run_id"] == "parent-nasa"
    assert metrics["parent_checkpoint_sha256"]
    assert set(metrics["norm_by_dataset"]) == {"smoke"}
    assert metrics["norm_by_dataset"]["smoke"]["mean"] == pytest.approx(4.0)
    assert metrics["norm_by_dataset"]["smoke"]["source_split"] == "train"
    assert metrics["norm_by_dataset"]["smoke"]["pixel_kind"] == "synthetic"
    assert metrics["wind_averaging_by_split"]["train"]["wind_averaging"] == ["synthetic"]
    assert metrics["wind_averaging_by_split"]["val"]["wind_averaging_s"] == [None]
    assert metrics["wind_averaging_s"] is None
    assert metrics["precision"] == "fp32"
    assert metrics["use_grad_scaler"] is False
    assert metrics["device_parity_ok"] is None
    payload = torch.load(run_dir / config["paths"]["checkpoint_name"], map_location="cpu", weights_only=False)
    parent_payload = torch.load(parent, map_location="cpu", weights_only=False)
    for key, tensor in parent_payload["state_dict"].items():
        assert torch.allclose(payload["state_dict"][key], tensor)
    assert "optimizer" not in payload


def test_init_from_rejects_an_architecture_mismatch(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("matplotlib")
    from src.make_smoke_data import write_smoke
    from src.train import run_training

    manifest = write_smoke(tmp_path / "smoke", n_storms=6, frames_per_storm=2, size=16)
    config = _runtime_config(tmp_path, manifest)
    parent = tmp_path / "parent.pt"
    _save_parent(parent, config, "synthetic")
    with pytest.raises(RuntimeError, match="architecture does not match"):
        run_training(
            config,
            dataset="smoke",
            model="resnet18",
            max_steps=1,
            device="cpu",
            init_from=parent,
        )


def test_resume_continues_from_the_last_epoch(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("matplotlib")
    from src.make_smoke_data import write_smoke
    from src.train import run_training

    manifest = write_smoke(tmp_path / "smoke", n_storms=6, frames_per_storm=2, size=16)
    config = _runtime_config(tmp_path, manifest)
    first = run_training(config, dataset="smoke", model="simplecnn", device="cpu")
    assert (first / "last.pt").is_file()
    curve = pd.read_csv(first / "loss_curve.csv")
    assert len(curve) == 1
    assert int(curve.iloc[0]["epoch"]) == 0
    config["train"]["epochs"] = 2
    second = run_training(config, dataset="smoke", model="simplecnn", device="cpu", resume=first)
    assert second == first
    continued = pd.read_csv(first / "loss_curve.csv")
    assert len(continued) == 2
    assert int(continued.iloc[1]["epoch"]) == 1
    assert continued.iloc[0]["train_loss"] == pytest.approx(curve.iloc[0]["train_loss"])
    metrics = json.loads((first / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["epochs_ran"] == 2
    assert metrics["device"] == "cpu"
    assert metrics["precision"] == "fp32"
    assert metrics["peak_memory_bytes"] is None
    assert metrics["git_commit"]
    assert metrics["seed"] == config["project"]["seed"]
    assert (first / "resolved_config.yaml").is_file()


def test_bench_device_runs_two_cpu_steps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from scripts.bench_device import run_bench

    monkeypatch.delenv("CLOUDIE_ALLOW_GPU", raising=False)
    with pytest.raises(RuntimeError, match="CLOUDIE_ALLOW_GPU"):
        run_bench(device="xpu", steps=2, batch_size=1, out_dir=tmp_path)
    payload = run_bench(device="cpu", steps=2, batch_size=1, out_dir=tmp_path)
    assert payload["is_synthetic"] is True
    assert payload["result"] is False
    assert payload["device"] == "cpu"
    assert payload["precision"] == "fp32"
    assert payload["use_grad_scaler"] is False
    assert len(payload["runs"]) == 4
    sizes = {row["dataset"]: row["image_size"] for row in payload["runs"]}
    assert sizes["nasa"] == 366
    assert sizes["tcir"] == 201
    for row in payload["runs"]:
        assert row["is_synthetic"] is True
        assert row["result"] is False
        assert row["steps"] == 2
        assert row["warmup_s"] >= 0
        assert row["images_per_sec"] > 0
        assert row["device_parity_ok"] is True
        assert row["precision"] == "fp32"
    written = json.loads(Path(payload["path"]).read_text(encoding="utf-8"))
    assert written["result"] is False
    assert written["is_synthetic"] is True
    assert Path(payload["path"]).parent == tmp_path
