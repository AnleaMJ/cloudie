"""Forward pass and MC-dropout for the deep regressors.

ImageNet weights are not downloaded. ``pretrained=False`` uses ``weights=None``.
``pretrained=True`` is refused unless a test mocks the constructor.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cloudie.config import load_config, parse_image_size

ROOT = Path(__file__).resolve().parents[1]
# 128 is a square resize, 201 is a TCIR chip, 366 is a NASA TorchGeo chip
# (Gurung et al. 2023). The model does not store these sizes.
CHIP_SIZES = [128, 201, 366]


def _config() -> dict:
    return load_config(ROOT / "config" / "config.yaml")


def _forbid_download(*_args, **_kwargs):
    raise AssertionError("ImageNet weight download is not allowed in tests")


@pytest.fixture
def config() -> dict:
    loaded = _config()
    assert loaded["model"]["pretrained"] is False
    assert float(loaded["model"]["dropout"]) > 0
    assert int(loaded["model"]["mc_samples"]) >= 2
    assert int(loaded["data"]["channels"]) == 1
    assert parse_image_size(loaded["data"]["image_size"]) is None
    return loaded


@pytest.fixture
def forbid_imagenet_download(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("torchvision")
    monkeypatch.setattr("torchvision.models._api.load_state_dict_from_url", _forbid_download)


def _batch(torch, channels: int, size: int, seed: int):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.rand(2, channels, size, size, generator=generator)


def _bn_snapshot(model):
    from torch.nn.modules.batchnorm import _BatchNorm

    rows = []
    for layer in model.modules():
        if isinstance(layer, _BatchNorm):
            rows.append(
                (
                    layer.running_mean.detach().clone(),
                    layer.running_var.detach().clone(),
                    layer.num_batches_tracked.detach().clone(),
                )
            )
    return rows


def _same_snapshot(left, right) -> bool:
    torch = pytest.importorskip("torch")
    if len(left) != len(right):
        return False
    for before, after in zip(left, right, strict=True):
        for old, new in zip(before, after, strict=True):
            if not torch.equal(old, new):
                return False
    return True


def _watch_modes(model):
    from torch import nn
    from torch.nn.modules.batchnorm import _BatchNorm

    flags = {"batchnorm": [], "dropout": []}

    def remember(sink):
        def hook(module, _inputs):
            sink.append(bool(module.training))

        return hook

    for layer in model.modules():
        if isinstance(layer, _BatchNorm):
            layer.register_forward_pre_hook(remember(flags["batchnorm"]))
        if isinstance(layer, nn.Dropout):
            layer.register_forward_pre_hook(remember(flags["dropout"]))
    return flags


def test_model_source_has_no_fixed_image_size() -> None:
    source = (ROOT / "cloudie" / "model.py").read_text(encoding="utf-8")
    assert "128" not in source
    assert "201" not in source
    assert "224" not in source
    assert "366" not in source
    assert "AdaptiveAvgPool2d" in source


@pytest.mark.parametrize("image_size", CHIP_SIZES)
def test_forward_and_mc_dropout_at_each_native_size(config: dict, forbid_imagenet_download, image_size: int) -> None:
    torch = pytest.importorskip("torch")
    from torch import nn
    from torch.nn.modules.batchnorm import _BatchNorm

    from cloudie.model import MCDropout, ResNetRegressor, SimpleCNN, mc_dropout_predict

    channels = int(config["data"]["channels"])
    dropout = float(config["model"]["dropout"])
    n_samples = int(config["model"]["mc_samples"])
    pretrained = bool(config["model"]["pretrained"])
    seed = int(config["project"]["seed"])
    builders = (
        lambda: SimpleCNN(in_channels=channels, dropout=dropout),
        lambda: ResNetRegressor(
            in_channels=channels, pretrained=pretrained, dropout=dropout
        ),
    )

    inputs = _batch(torch, channels, image_size, seed)
    for build in builders:
        torch.manual_seed(seed)
        model = build()
        pools = [layer for layer in model.modules() if isinstance(layer, nn.AdaptiveAvgPool2d)]
        assert pools
        assert all(layer.output_size in (1, (1, 1)) for layer in pools)
        assert model.class_head is None

        output = model(inputs)
        assert torch.is_tensor(output)
        assert tuple(output.shape) == (2, 1)
        assert torch.isfinite(output).all()

        if isinstance(model, ResNetRegressor):
            conv1 = model.backbone.conv1
            assert int(conv1.in_channels) == 3
            assert int(conv1.weight.shape[1]) == 3
            seen: dict[str, int] = {}

            def _channels(_module, args, box=seen):
                box["channels"] = int(args[0].shape[1])

            handle = conv1.register_forward_pre_hook(_channels)
            model(inputs)
            handle.remove()
            assert seen["channels"] == 3
            held = conv1.weight.detach().clone()
            batchnorm = [layer for layer in model.modules() if isinstance(layer, _BatchNorm)]
            assert batchnorm
            initial = _bn_snapshot(model)
            model.train()
            with torch.no_grad():
                model(inputs)
            warmed = _bn_snapshot(model)
            assert not _same_snapshot(initial, warmed)
            flags = _watch_modes(model)
            mean, std = MCDropout.predict(model, inputs, n_samples)
            assert torch.equal(conv1.weight, held)
            assert _same_snapshot(warmed, _bn_snapshot(model))
            assert flags["batchnorm"]
            assert all(flag is False for flag in flags["batchnorm"])
            assert flags["dropout"]
            assert all(flag is True for flag in flags["dropout"])
            assert model.training is False
            assert all(layer.training is False for layer in batchnorm)
            assert all(
                layer.training is True
                for layer in model.modules()
                if isinstance(layer, nn.Dropout)
            )
        else:
            assert not any(isinstance(layer, _BatchNorm) for layer in model.modules())
            flags = _watch_modes(model)
            mean, std = mc_dropout_predict(model, inputs, n_samples)
            assert flags["dropout"]
            assert all(flag is True for flag in flags["dropout"])
            assert flags["batchnorm"] == []

        assert tuple(mean.shape) == (2, 1)
        assert tuple(std.shape) == (2, 1)
        assert torch.isfinite(mean).all()
        assert torch.all(std > 0)


@pytest.mark.parametrize("image_size", CHIP_SIZES)
def test_optional_class_head_shapes(config: dict, forbid_imagenet_download, image_size: int) -> None:
    torch = pytest.importorskip("torch")

    from cloudie.model import MCDropout, OptionalClassHead, ResNetRegressor, SimpleCNN

    n_classes = len(config["data"]["imd5_names"])
    assert n_classes >= 2
    channels = int(config["data"]["channels"])
    dropout = float(config["model"]["dropout"])
    n_samples = int(config["model"]["mc_samples"])
    seed = int(config["project"]["seed"])
    with pytest.raises(ValueError, match="n_classes"):
        OptionalClassHead.build(8, 0)

    inputs = _batch(torch, channels, image_size, seed)
    for model in (
        SimpleCNN(in_channels=channels, dropout=dropout, n_classes=n_classes),
        ResNetRegressor(
            in_channels=channels,
            pretrained=False,
            dropout=dropout,
            n_classes=n_classes,
        ),
    ):
        wind, logits = model(inputs)
        assert tuple(wind.shape) == (2, 1)
        assert tuple(logits.shape) == (2, n_classes)
        assert int(model.class_head.in_features) == int(model.feature_dim)
        torch.manual_seed(seed)
        mean, std = MCDropout.predict(model, inputs, n_samples)
        assert tuple(mean.shape) == (2, 1)
        assert torch.all(std > 0)


@pytest.mark.parametrize("image_size", CHIP_SIZES)
def test_resnet_conv1_accepts_three_channels_without_repeating(
    config: dict, forbid_imagenet_download, image_size: int
) -> None:
    torch = pytest.importorskip("torch")

    from cloudie.model import ResNetRegressor

    model = ResNetRegressor(
        in_channels=3, pretrained=False, dropout=float(config["model"]["dropout"])
    )
    seen: dict[str, int] = {}

    def _channels(_module, args, box=seen):
        box["channels"] = int(args[0].shape[1])

    model.backbone.conv1.register_forward_pre_hook(_channels)
    output = model(_batch(torch, 3, image_size, int(config["project"]["seed"])))
    assert seen["channels"] == 3
    assert tuple(output.shape) == (2, 1)
    with pytest.raises(ValueError, match="conv1"):
        ResNetRegressor(in_channels=4, pretrained=False)


def test_weights_none_does_not_download(config: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from torchvision.models import resnet18 as real_resnet18

    from cloudie.model import ResNetRegressor

    monkeypatch.setattr("torchvision.models._api.load_state_dict_from_url", _forbid_download)
    calls: dict[str, object] = {}

    def fake_resnet18(weights=None, **kwargs):
        calls["weights"] = weights
        return real_resnet18(weights=None)

    monkeypatch.setattr("torchvision.models.resnet18", fake_resnet18)
    model = ResNetRegressor(
        in_channels=int(config["data"]["channels"]),
        pretrained=bool(config["model"]["pretrained"]),
        dropout=float(config["model"]["dropout"]),
    )
    assert calls["weights"] is None
    assert int(model.backbone.conv1.in_channels) == 3
    assert int(model.backbone.conv1.weight.shape[1]) == 3


def test_pretrained_true_is_refused_without_download(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")

    from cloudie.model import ResNetRegressor

    monkeypatch.delenv("CLOUDIE_ALLOW_PRETRAINED", raising=False)
    monkeypatch.setattr("torchvision.models._api.load_state_dict_from_url", _forbid_download)

    def fail_if_built(*_args, **_kwargs):
        raise AssertionError("resnet18 was constructed for a refused pretrained load")

    monkeypatch.setattr("torchvision.models.resnet18", fail_if_built)
    with pytest.raises(RuntimeError, match="ImageNet"):
        ResNetRegressor(pretrained=True)


def test_pretrained_true_is_mockable(config: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from torchvision.models import ResNet18_Weights
    from torchvision.models import resnet18 as real_resnet18

    from cloudie.model import ResNetRegressor

    monkeypatch.setenv("CLOUDIE_ALLOW_PRETRAINED", "1")
    monkeypatch.setattr("torchvision.models._api.load_state_dict_from_url", _forbid_download)
    calls: dict[str, object] = {}

    def fake_resnet18(weights=None, **kwargs):
        calls["weights"] = weights
        return real_resnet18(weights=None)

    monkeypatch.setattr("torchvision.models.resnet18", fake_resnet18)
    model = ResNetRegressor(
        in_channels=int(config["data"]["channels"]),
        pretrained=True,
        dropout=float(config["model"]["dropout"]),
    )
    assert calls["weights"] is ResNet18_Weights.DEFAULT
    assert int(model.backbone.conv1.in_channels) == 3
    assert int(model.backbone.conv1.weight.shape[1]) == 3
    assert isinstance(model.backbone.avgpool, torch.nn.AdaptiveAvgPool2d)


def test_mc_dropout_rejects_a_single_pass() -> None:
    torch = pytest.importorskip("torch")

    from cloudie.model import MCDropout, SimpleCNN

    model = SimpleCNN(dropout=float(_config()["model"]["dropout"]))
    inputs = torch.zeros(1, int(_config()["data"]["channels"]), 8, 8)
    with pytest.raises(ValueError, match="n_samples"):
        MCDropout.predict(model, inputs, 1)


@pytest.mark.parametrize("image_size", CHIP_SIZES)
def test_convlstm_accepts_chip_sizes(config: dict, image_size: int) -> None:
    torch = pytest.importorskip("torch")

    from cloudie.model import ConvLSTMRegressor

    model = ConvLSTMRegressor(
        in_channels=int(config["data"]["channels"]),
        hidden=int(config["model"]["hidden"]),
        n_frames=int(config["model"]["n_frames"]),
        dropout=float(config["model"]["dropout"]),
    )
    inputs = torch.zeros(
        1,
        int(config["model"]["n_frames"]),
        int(config["data"]["channels"]),
        image_size,
        image_size,
    )
    output = model(inputs)
    assert tuple(output.shape) == (1, 1)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("image_size", CHIP_SIZES)
def test_src_simple_cnn_accepts_chip_sizes(image_size: int) -> None:
    torch = pytest.importorskip("torch")

    from src.model import SimpleCNN

    output = SimpleCNN()(torch.zeros(1, 1, image_size, image_size))
    assert tuple(output.shape) == (1, 1)
    assert torch.isfinite(output).all()
