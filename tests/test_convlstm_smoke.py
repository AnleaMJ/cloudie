"""ConvLSTM training smoke test.

The batches are seeded noise, not satellite images. A falling loss only
shows that the convolutional gates learn this toy batch. It is not a
wind-speed score and it is not a forecast.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from src.config import load_config

ROOT = Path(__file__).resolve().parents[1]
SEED = 7
CHIP = 8
BATCH = 2
FRAMES = 4
CHANNELS = 1
HIDDEN = 4
HANDFUL_OF_STEPS = 5
OVERFIT_STEPS = 20
LEARNING_RATE = 0.05
# Half the starting loss. This seed falls much further on CPU; the looser
# bar leaves room for fp32 rounding on xpu or cuda.
OVERFIT_RATIO = 0.5
PARITY_ATOL = 1e-4
PARITY_RTOL = 1e-3


def _installed_accelerators() -> list[str]:
    """``xpu`` and ``cuda``, and only when this process can run them.

    Missing accelerators are left out of the parameter list. ``pytest.skip``
    would fail the torch CI job, which treats a skip as a failure.
    """
    try:
        import torch
    except ImportError:
        return []
    found: list[str] = []
    for name in ("xpu", "cuda"):
        backend = getattr(torch, name, None)
        if backend is None:
            continue
        try:
            ready = bool(backend.is_available())
        except (AttributeError, RuntimeError):
            ready = False
        if ready:
            found.append(name)
    return found


def _training_devices() -> list[str]:
    """CPU, plus an accelerator only when torch reports it available."""
    return ["cpu", *_installed_accelerators()]


def _new_model(torch, device: str):
    from torch import nn

    from src.model import ConvLSTMRegressor

    model = ConvLSTMRegressor(
        in_channels=CHANNELS,
        hidden=HIDDEN,
        n_frames=FRAMES,
        dropout=0.0,
    ).to(device=device, dtype=torch.float32)
    assert isinstance(model.cell.gates, nn.Conv2d)
    assert not any(isinstance(layer, nn.LSTM) for layer in model.modules())
    return model


def _seeded_problem(torch, device: str):
    """One fixed ``(batch, 4, C, H, W)`` batch. The values are not imagery."""
    torch.manual_seed(SEED)
    model = _new_model(torch, device)
    inputs = torch.rand(BATCH, FRAMES, CHANNELS, CHIP, CHIP, dtype=torch.float32).to(device)
    target = torch.tensor([[0.25], [0.75]], dtype=torch.float32, device=device)
    return model, inputs, target


def _assert_cell_grads(cell_params: list) -> None:
    torch = pytest.importorskip("torch")
    assert cell_params
    for param in cell_params:
        assert param.grad is not None
        assert tuple(param.grad.shape) == tuple(param.shape)
        assert torch.isfinite(param.grad).all()


@pytest.mark.parametrize("device", _training_devices())
def test_convlstm_steps_have_finite_loss_and_cell_grads(device: str) -> None:
    torch = pytest.importorskip("torch")
    model, inputs, target = _seeded_problem(torch, device)
    cell_params = list(model.cell.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    model.train()
    for _ in range(HANDFUL_OF_STEPS):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(inputs)
        assert tuple(prediction.shape) == (BATCH, 1)
        loss = torch.nn.functional.mse_loss(prediction, target)
        assert torch.isfinite(loss).all()
        loss.backward()
        _assert_cell_grads(cell_params)
        optimizer.step()


@pytest.mark.parametrize("device", _training_devices())
def test_convlstm_overfits_a_fixed_batch(device: str) -> None:
    torch = pytest.importorskip("torch")
    model, inputs, target = _seeded_problem(torch, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    model.train()
    losses: list[float] = []
    for _ in range(OVERFIT_STEPS):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(inputs), target)
        assert torch.isfinite(loss).all()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert math.isfinite(losses[0]) and losses[0] > 0.0
    assert math.isfinite(losses[-1])
    assert losses[-1] < losses[0] * OVERFIT_RATIO


@pytest.mark.parametrize("device", _training_devices())
def test_convlstm_state_dict_round_trip(tmp_path: Path, device: str) -> None:
    torch = pytest.importorskip("torch")
    model, inputs, target = _seeded_problem(torch, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.nn.functional.mse_loss(model(inputs), target).backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        expected = model(inputs)
    snapshot = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    path = tmp_path / "convlstm_state.pt"
    torch.save(snapshot, path)

    fresh = _new_model(torch, device)
    fresh.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    fresh.eval()
    with torch.no_grad():
        actual = fresh(inputs)
    for key, value in snapshot.items():
        assert torch.equal(fresh.state_dict()[key].detach().cpu(), value)
    assert torch.equal(actual, expected)


def _assert_fp32_matches_cpu(torch, device: str) -> None:
    """Same weights and the same seeded batch, CPU versus one accelerator.

    Lunar Lake Arc 140V has returned wrong answers without an error on old
    XPU drivers (pytorch/pytorch#200713). Eval-mode fp32 outputs, and the
    grads from one backward, have to agree with CPU.
    """
    cpu_model, inputs, target = _seeded_problem(torch, "cpu")
    other = _new_model(torch, device)
    other.load_state_dict(cpu_model.state_dict())
    for left, right in zip(cpu_model.parameters(), other.parameters(), strict=True):
        assert left.dtype == torch.float32
        assert right.dtype == torch.float32
        assert right.device.type == device
        assert torch.equal(left, right.detach().cpu())

    cpu_model.eval()
    other.eval()
    assert cpu_model.training is False
    assert other.training is False
    other_inputs = inputs.to(device)
    other_target = target.to(device)
    cpu_out = cpu_model(inputs)
    other_out = other(other_inputs)
    assert cpu_out.dtype == torch.float32
    assert other_out.dtype == torch.float32
    assert torch.isfinite(cpu_out).all()
    assert torch.isfinite(other_out).all()
    assert torch.allclose(
        other_out.detach().cpu(),
        cpu_out.detach(),
        atol=PARITY_ATOL,
        rtol=PARITY_RTOL,
    )

    torch.nn.functional.mse_loss(cpu_out, target).backward()
    torch.nn.functional.mse_loss(other_out, other_target).backward()
    cpu_named = dict(cpu_model.named_parameters())
    other_named = dict(other.named_parameters())
    assert cpu_named.keys() == other_named.keys()
    assert any(name.startswith("cell.") for name in cpu_named)
    for name, left in cpu_named.items():
        right = other_named[name]
        assert left.grad is not None, name
        assert right.grad is not None, name
        assert torch.isfinite(left.grad).all()
        assert torch.isfinite(right.grad).all()
        assert torch.allclose(
            right.grad.detach().cpu(),
            left.grad.detach(),
            atol=PARITY_ATOL,
            rtol=PARITY_RTOL,
        )


# Collected only for an accelerator this interpreter can run. An empty
# parameter set would be recorded as a skip, and CPU CI fails on skips.
_PARITY_DEVICES = _installed_accelerators()
if _PARITY_DEVICES:

    @pytest.mark.parametrize("device", _PARITY_DEVICES)
    def test_convlstm_fp32_matches_cpu(device: str) -> None:
        torch = pytest.importorskip("torch")
        _assert_fp32_matches_cpu(torch, device)


def test_train_runs_one_epoch_of_convlstm_on_synthetic_smoke(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``train()`` has no step cap. One epoch on a tiny smoke manifest is the cheap path.

    ``run_training`` does not accept convlstm. This stays on ``train()`` and
    on synthetic rows from ``write_smoke``.
    """
    torch = pytest.importorskip("torch")
    from src.make_smoke_data import write_smoke
    from src.train import build_net, train

    config = load_config(ROOT / "config" / "config_smoke.yaml")
    config["model"]["name"] = "convlstm"
    config["model"]["hidden"] = HIDDEN
    config["data"]["image_size"] = CHIP
    config["train"]["epochs"] = 1
    config["train"]["batch_size"] = 2
    config["train"]["patience"] = 1
    config["paths"]["artifacts"] = str(tmp_path / "artifacts")
    config["paths"]["reports"] = str(tmp_path / "reports")
    manifest = write_smoke(
        tmp_path / "smoke",
        n_storms=3,
        frames_per_storm=FRAMES,
        size=CHIP,
        seed=SEED,
    )
    config["paths"]["manifest"] = str(manifest)

    checkpoint = train(config, manifest)
    printed = capsys.readouterr().out.lower()
    assert "synthetic" in printed
    assert "do not report" in printed
    assert checkpoint.is_file()

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["is_synthetic"] is True
    assert payload["present_as_forecast"] is False
    assert payload["model_name"] == "convlstm"
    assert int(payload["n_frames"]) == FRAMES
    assert int(payload["hidden"]) == HIDDEN
    assert payload["size_head"] is False
    assert "backbone.cell.gates.weight" in payload["state_dict"]
    assert "backbone.cell.gates.bias" in payload["state_dict"]
    assert math.isfinite(float(payload["best_val_loss"]))
    assert "synthetic" in str(payload["honesty"]).lower()

    net = build_net(config, size_head=bool(payload["size_head"]))
    net.load_state_dict(payload["state_dict"])
    net.eval()
    with torch.no_grad():
        wind, logits = net(torch.zeros(1, FRAMES, CHANNELS, CHIP, CHIP))
    assert tuple(wind.shape) == (1, 1)
    assert torch.isfinite(wind).all()
    assert torch.isfinite(logits).all()
