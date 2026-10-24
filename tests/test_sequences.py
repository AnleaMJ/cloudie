"""Phase 5. Four-frame windows, and a size head only when size labels exist."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cloudie.config import load_config
from cloudie.make_smoke_data import write_smoke
from cloudie.preprocessing import prepare_image
from cloudie.sequences import N_FRAMES, frame_windows, size_targets_present, stack_windows
from cloudie.splits import null_cross_split_links

ROOT = Path(__file__).resolve().parents[1]


def test_size_head_uses_the_train_split_only() -> None:
    from cloudie.train import train_size_head

    frame = pd.DataFrame(
        {
            "split": ["train", "train", "test"],
            "r34_nmi": [np.nan, np.nan, 40.0],
            "rmw_nmi": [np.nan, np.nan, 15.0],
        }
    )
    assert train_size_head(frame) is False
    frame.loc[0, "r34_nmi"] = 30.0
    frame.loc[0, "rmw_nmi"] = 12.0
    assert train_size_head(frame) is True


def test_training_step_updates_the_size_head_from_image_features() -> None:
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader

    from cloudie.train import _dataset, _run_epoch, build_net

    config = load_config(ROOT / "config" / "config_smoke.yaml")
    config["data"]["image_size"] = 16
    net = build_net(config, size_head=True)
    before = net.size_head.weight.detach().clone()
    images = np.random.default_rng(0).random((4, 1, 16, 16), dtype=np.float32)
    winds = np.array([30, 40, 55, 70], dtype=np.float32)
    labels = ["D_DD", "CS", "SCS", "VSCS"]
    weights = np.ones(4, dtype=np.float64)
    size = np.array([[20, 10], [22, 11], [24, 12], [26, 13]], dtype=np.float32)
    known = [str(name) for name in config["data"]["imd5_names"]]
    loader = DataLoader(
        _dataset(images, winds, labels, known, weights, size),
        batch_size=4,
    )
    optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
    _run_epoch(
        net,
        loader,
        optimizer,
        torch.ones(len(known)),
        0.0,
        True,
        size_loss_weight=1.0,
    )
    assert not torch.equal(before, net.size_head.weight.detach())
    assert build_net(config, size_head=False).size_head is None


def test_size_head_stays_off_until_both_labels_exist() -> None:
    empty = pd.DataFrame({"wind_speed_kt": [40.0]})
    assert size_targets_present(empty) is False
    one = empty.copy()
    one["r34_nmi"] = [30.0]
    assert size_targets_present(one) is False
    both = one.copy()
    both["rmw_nmi"] = [np.nan]
    assert size_targets_present(both) is False
    both.loc[0, "rmw_nmi"] = 12.0
    assert size_targets_present(both) is True


def test_windows_do_not_cross_a_split() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": ["a0", "a1", "a2", "a3", "a4"],
            "storm_id": ["A"] * 5,
            "split": ["test", "train", "train", "train", "train"],
            "timestamp": ["1", "2", "3", "4", "5"],
            "prev_sample_id": [pd.NA, "a0", "a1", "a2", "a3"],
        }
    )
    blocked = frame_windows(frame, "train")
    assert blocked == []
    assert all("a0" not in window for window in frame_windows(frame, "test"))
    cleaned, dropped = null_cross_split_links(frame)
    assert dropped == 1
    assert frame_windows(cleaned, "train") == [["a1", "a2", "a3", "a4"]]
    assert frame_windows(cleaned, "test") == []
    sliding = pd.DataFrame(
        {
            "sample_id": ["b0", "b1", "b2", "b3", "b4"],
            "storm_id": ["B"] * 5,
            "split": ["train"] * 5,
            "timestamp": ["1", "2", "3", "4", "5"],
            "prev_sample_id": [pd.NA, "b0", "b1", "b2", "b3"],
        }
    )
    assert frame_windows(sliding, "train") == [
        ["b0", "b1", "b2", "b3"],
        ["b1", "b2", "b3", "b4"],
    ]


def test_short_storm_has_no_window_and_length_is_four() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": ["s0", "s1", "s2"],
            "storm_id": ["S", "S", "S"],
            "split": ["train", "train", "train"],
            "timestamp": ["1", "2", "3"],
        }
    )
    assert frame_windows(frame, "train") == []
    with pytest.raises(ValueError, match="4"):
        frame_windows(frame, "train", n_frames=3)
    assert N_FRAMES == 4


def test_targets_are_not_image_channels(tmp_path: Path) -> None:
    manifest = write_smoke(tmp_path, n_storms=1, frames_per_storm=4, size=16)
    frame = pd.read_csv(manifest)
    frame["split"] = "train"
    frame["hemisphere"] = "SH"
    frame["r34_nmi"] = [20.0, 21.0, 22.0, 23.0]
    frame["rmw_nmi"] = [8.0, 8.0, 9.0, 9.0]
    images, winds, _labels, size = stack_windows(frame, manifest, "train", 16, None, 0.30)
    assert images.shape[1:] == (4, 1, 16, 16)
    shifted = frame.copy()
    shifted["wind_speed_kt"] = shifted["wind_speed_kt"] + 25
    shifted["r34_nmi"] = shifted["r34_nmi"] + 10
    shifted["rmw_nmi"] = shifted["rmw_nmi"] + 3
    images_again, winds_again, _labels_again, size_again = stack_windows(
        shifted, manifest, "train", 16, None, 0.30
    )
    assert np.allclose(images, images_again)
    assert not np.allclose(winds, winds_again)
    assert not np.allclose(size, size_again)
    original = np.load(tmp_path / frame.iloc[0]["image_path"])
    prepared = prepare_image(original, 16, None)
    flipped = prepare_image(np.fliplr(original), 16, None)
    assert np.allclose(images[0, 0], prepared)
    assert not np.allclose(images[0, 0], flipped)


def test_convlstm_shape_and_size_head_ignore_wind() -> None:
    torch = pytest.importorskip("torch")
    from cloudie.model import ConvLSTMRegressor
    from cloudie.train import _size_loss, build_net

    model = ConvLSTMRegressor(hidden=8)
    output = model(torch.zeros(2, 4, 1, 16, 16))
    assert tuple(output.shape) == (2, 1)
    with pytest.raises(ValueError, match="4"):
        model(torch.zeros(2, 3, 1, 16, 16))
    with pytest.raises(ValueError, match="4"):
        ConvLSTMRegressor(n_frames=2)

    config = load_config(ROOT / "config" / "config_smoke.yaml")
    config["model"]["name"] = "convlstm"
    config["model"]["hidden"] = 8
    config["data"]["image_size"] = 16
    net = build_net(config, size_head=True)
    net.eval()
    frames = torch.rand(2, 4, 1, 16, 16)
    wind, _logits, size = net(frames)
    assert tuple(size.shape) == (2, 2)
    assert net.size_head.in_features == net.backbone.feature_dim
    again = net(frames)[2]
    assert torch.equal(size, again)
    assert tuple(wind.shape) == (2, 1)
    bare = build_net(config, size_head=False)
    assert bare.size_head is None
    assert len(bare(frames)) == 2

    net.train()
    _wind, _logits, size = net(frames)
    loss = _size_loss(size, torch.tensor([[30.0, 10.0], [float("nan"), 12.0]]))
    loss.backward()
    assert net.size_head.weight.grad is not None
    empty = _size_loss(size.detach(), torch.full((2, 2), float("nan")))
    assert float(empty) == pytest.approx(0.0)


def test_convlstm_checkpoint_needs_four_frames(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from cloudie.predict import predict_images

    config = load_config(ROOT / "config" / "config_smoke.yaml")
    config["data"]["image_size"] = 16
    config["model"]["mc_samples"] = 2
    config["model"]["name"] = "convlstm"
    config["model"]["hidden"] = 4
    from cloudie.train import build_net

    import torch

    model = build_net(config, size_head=False)
    payload = {
        "state_dict": model.state_dict(),
        "model_name": "convlstm",
        "dropout": float(config["model"]["dropout"]),
        "pretrained": False,
        "in_channels": 1,
        "n_classes": len(config["data"]["imd5_names"]),
        "class_names": [str(name) for name in config["data"]["imd5_names"]],
        "edges_kt": [float(edge) for edge in config["data"]["imd5_edges_kt"]],
        "norm_mean": 0.0,
        "norm_std": 1.0,
        "norm_source_split": "train",
        "image_size": 16,
        "max_missing_fraction": 0.30,
        "mean_baseline_kt": 40.0,
        "is_synthetic": True,
        "wind_averaging": "synthetic",
        "n_frames": 4,
        "size_head": False,
        "hidden": 4,
        "present_as_forecast": True,
        "honesty": "not a forecast",
    }
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(payload, checkpoint)
    image = np.ones((16, 16), dtype=np.float32)
    short = predict_images(config, checkpoint, [image, image, image])
    assert short.show_number is False
    assert short.wind_kt is None
    prediction = predict_images(
        config, checkpoint, [image, image, image, image], hemisphere="SH"
    )
    assert prediction.show_number is True
    assert prediction.present_as_forecast is False
    assert prediction.is_synthetic is True


def test_smoke_convlstm_train_has_no_size_term_and_is_not_a_forecast(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("matplotlib")
    from cloudie.evaluate import evaluate
    from cloudie.train import train

    config = load_config(ROOT / "config" / "config_smoke.yaml")
    config["model"]["name"] = "convlstm"
    config["model"]["hidden"] = 4
    config["data"]["image_size"] = 16
    config["train"]["epochs"] = 1
    config["train"]["batch_size"] = 4
    config["paths"]["artifacts"] = str(tmp_path / "artifacts")
    config["paths"]["reports"] = str(tmp_path / "reports")
    manifest = write_smoke(tmp_path / "smoke", size=16)
    config["paths"]["manifest"] = str(manifest)
    checkpoint = train(config, manifest)

    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["size_head"] is False
    assert payload["n_frames"] == 4
    assert payload["is_synthetic"] is True
    assert payload["present_as_forecast"] is False
    assert "size_head.weight" not in payload["state_dict"]
    result = evaluate(config, checkpoint)
    assert result.present_as_forecast is False
    assert result.is_synthetic is True
    assert result.model["n"] > 0
    assert result.operational_baselines == {"ADT": None, "SATCON": None, "Dvorak": None}


def test_size_labels_turn_the_head_on_for_the_train_split_only(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from cloudie.train import train

    config = load_config(ROOT / "config" / "config_smoke.yaml")
    config["data"]["image_size"] = 16
    config["train"]["epochs"] = 1
    config["train"]["batch_size"] = 8
    config["paths"]["artifacts"] = str(tmp_path / "artifacts")
    config["paths"]["reports"] = str(tmp_path / "reports")
    manifest = write_smoke(tmp_path / "smoke", size=16)
    frame = pd.read_csv(manifest)
    frame["r34_nmi"] = 20.0 + frame["wind_speed_kt"] * 0.0 + np.arange(len(frame))
    frame["rmw_nmi"] = 10.0
    frame.to_csv(manifest, index=False)
    config["paths"]["manifest"] = str(manifest)
    checkpoint = train(config, manifest)

    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["size_head"] is True
    assert "size_head.weight" in payload["state_dict"]
    assert tuple(payload["state_dict"]["size_head.weight"].shape) == (2, 64)
    assert payload["present_as_forecast"] is False
    assert payload["n_frames"] == 1
