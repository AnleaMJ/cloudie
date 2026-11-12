"""Device selection and mixed precision for native PyTorch.

The order is XPU, then CUDA, then CPU. XPU means ``torch.xpu`` in an
official PyTorch build (2.5 or newer; 2.15 is the pinned Windows wheel).
This module does not import or call the archived IPEX package.

``CLOUDIE_ALLOW_GPU=1`` is required for XPU and CUDA alike. An explicit
accelerator without that gate is refused. Auto falls back to CPU and
logs the reason. CPU stays fp32. CUDA fp16 is the only path that uses
GradScaler. XPU uses bf16 autocast when native bf16 is supported, and
fp32 otherwise. ``cudnn.benchmark`` is not enabled.
"""

from __future__ import annotations

import inspect
import logging
import math
import os
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass

LOGGER = logging.getLogger("src.device")

DEVICE_CHOICES = ("auto", "xpu", "cuda", "cpu")
GPU_DEVICES = ("xpu", "cuda")
GPU_GATE = "CLOUDIE_ALLOW_GPU"


def _torch():
    import torch

    return torch


def accelerator_available(torch_module, name: str) -> bool:
    """True when this torch build can see ``name``. Missing XPU attributes are false."""
    if name == "cpu":
        return True
    if name == "xpu":
        xpu = getattr(torch_module, "xpu", None)
        checker = getattr(xpu, "is_available", None) if xpu is not None else None
        if not callable(checker):
            return False
        try:
            return bool(checker())
        except Exception:
            return False
    if name == "cuda":
        try:
            return bool(torch_module.cuda.is_available())
        except Exception:
            return False
    return False


def xpu_native_bf16(torch_module) -> bool:
    """True when native XPU bf16 is supported.

    Uses ``torch.xpu.is_bf16_supported(including_emulation=False)`` when
    that parameter exists, so emulated bf16 does not select the autocast
    path. Older builds without the function, or whose call fails, stay
    on fp32. A build that has the function but not the parameter is
    called with no arguments.
    """
    xpu = getattr(torch_module, "xpu", None)
    checker = getattr(xpu, "is_bf16_supported", None) if xpu is not None else None
    if not callable(checker):
        return False
    try:
        parameters = inspect.signature(checker).parameters
        has_kwarg = "including_emulation" in parameters
    except (TypeError, ValueError):
        has_kwarg = False
    try:
        if has_kwarg:
            return bool(checker(including_emulation=False))
        return bool(checker())
    except TypeError:
        try:
            return bool(checker())
        except Exception:
            return False
    except Exception:
        return False


@dataclass(frozen=True)
class AmpPolicy:
    """How one device runs the forward and the backward."""

    precision: str
    autocast_device: str | None
    dtype: object | None
    use_grad_scaler: bool


def amp_policy_for(device, torch_module=None) -> AmpPolicy:
    """fp16+GradScaler on CUDA, native bf16 on XPU, fp32 everywhere else.

    GradScaler is never constructed for XPU or CPU. A missing native bf16
    on XPU falls back to fp32 with no autocast.
    """
    torch_module = torch_module or _torch()
    kind = device.type if hasattr(device, "type") else str(device)
    if kind == "cuda":
        return AmpPolicy("fp16", "cuda", torch_module.float16, True)
    if kind == "xpu":
        if xpu_native_bf16(torch_module):
            return AmpPolicy("bf16", "xpu", torch_module.bfloat16, False)
        return AmpPolicy("fp32", None, None, False)
    return AmpPolicy("fp32", None, None, False)


@contextmanager
def autocast_context(policy: AmpPolicy) -> Iterator[None]:
    """Enter ``torch.autocast`` for the policy, or do nothing for fp32."""
    if policy.autocast_device is None or policy.dtype is None:
        with nullcontext():
            yield
        return
    torch_module = _torch()
    with torch_module.autocast(policy.autocast_device, dtype=policy.dtype):
        yield


def make_grad_scaler(policy: AmpPolicy):
    """A CUDA fp16 GradScaler, or None. XPU and CPU never get one."""
    if not policy.use_grad_scaler:
        return None
    if policy.precision != "fp16" or policy.autocast_device != "cuda":
        raise RuntimeError("GradScaler is used only for cuda fp16")
    torch_module = _torch()
    scaler_cls = getattr(getattr(torch_module, "amp", None), "GradScaler", None)
    if scaler_cls is not None:
        try:
            return scaler_cls("cuda")
        except TypeError:
            pass
    return torch_module.cuda.amp.GradScaler()


def resolve_device(name: str):
    """Pick a ``torch.device``. Auto order is xpu, cuda, cpu.

    Explicit xpu or cuda without ``CLOUDIE_ALLOW_GPU=1`` raises. Auto
    that would have picked an accelerator falls back to cpu and logs why.
    """
    choice = str(name).strip().lower()
    if choice not in DEVICE_CHOICES:
        raise ValueError("device must be auto, xpu, cuda, or cpu")
    torch_module = _torch()
    if choice == "auto":
        if accelerator_available(torch_module, "xpu"):
            picked = "xpu"
        elif accelerator_available(torch_module, "cuda"):
            picked = "cuda"
        else:
            picked = "cpu"
        if picked != "cpu" and os.environ.get(GPU_GATE) != "1":
            message = (
                f"Refusing auto-picked device {picked} because {GPU_GATE} is not 1. "
                "Falling back to cpu. An XPU or GPU run needs misty wind's approval. "
                f"Set {GPU_GATE}=1 only after that approval."
            )
            LOGGER.warning(message)
            print(message)
            return torch_module.device("cpu")
        choice = picked
    if choice in GPU_DEVICES and os.environ.get(GPU_GATE) != "1":
        raise RuntimeError(
            f"Refusing device {choice}. An XPU or GPU run needs misty wind's approval. "
            f"Set {GPU_GATE}=1 only after that approval."
        )
    if choice == "xpu" and not accelerator_available(torch_module, "xpu"):
        raise RuntimeError("device xpu was approved but XPU is not available")
    if choice == "cuda" and not accelerator_available(torch_module, "cuda"):
        raise RuntimeError("device cuda was approved but CUDA is not available")
    try:
        return torch_module.device(choice)
    except RuntimeError as exc:
        raise RuntimeError(f"device {choice} is not supported by this torch build") from exc


def reset_peak_memory(device) -> None:
    """Clear the allocator peak for the device that tracks it. CPU is a no-op."""
    torch_module = _torch()
    if getattr(device, "type", None) == "cuda":
        reset = getattr(torch_module.cuda, "reset_peak_memory_stats", None)
        if callable(reset):
            try:
                reset()
            except Exception:
                return
    if getattr(device, "type", None) == "xpu":
        xpu = getattr(torch_module, "xpu", None)
        reset = getattr(xpu, "reset_peak_memory_stats", None) if xpu is not None else None
        if callable(reset):
            try:
                reset()
            except Exception:
                return


def peak_memory_bytes(device) -> int | None:
    """Peak bytes from XPU or CUDA, or null when the device does not report it.

    XPU peak memory is trustworthy on the torch 2.15 line. Older builds
    may return a number that is not the true peak; the value is still
    whatever ``max_memory_allocated`` reports.
    """
    torch_module = _torch()
    kind = getattr(device, "type", None)
    if kind == "xpu":
        xpu = getattr(torch_module, "xpu", None)
        reader = getattr(xpu, "max_memory_allocated", None) if xpu is not None else None
        if not callable(reader):
            return None
        try:
            return int(reader())
        except Exception:
            return None
    if kind == "cuda":
        reader = getattr(torch_module.cuda, "max_memory_allocated", None)
        if not callable(reader):
            return None
        try:
            return int(reader())
        except Exception:
            return None
    return None


def _jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def xpu_device_properties(device) -> dict | None:
    """``torch.xpu.get_device_properties`` as a JSON object, or null off XPU."""
    if getattr(device, "type", None) != "xpu":
        return None
    torch_module = _torch()
    xpu = getattr(torch_module, "xpu", None)
    getter = getattr(xpu, "get_device_properties", None) if xpu is not None else None
    if not callable(getter):
        return None
    index = 0
    current = getattr(xpu, "current_device", None)
    if callable(current):
        try:
            index = int(current())
        except Exception:
            index = 0
    try:
        props = getter(index)
    except Exception:
        return None
    if hasattr(props, "_asdict"):
        raw = dict(props._asdict())
    else:
        raw = {}
        for key in dir(props):
            if key.startswith("_"):
                continue
            try:
                value = getattr(props, key)
            except Exception:
                continue
            if callable(value):
                continue
            raw[key] = value
    return {str(key): _jsonable(value) for key, value in raw.items()}


def parity_verdict(
    output_max_abs: float,
    loss_max_abs: float,
    tolerance: float,
    reference_scale: float,
) -> dict:
    """Decide whether a device-versus-CPU gap is inside the tolerance.

    ``tolerance`` is a fraction. The allowed absolute gap is
    ``tolerance * max(1, reference_scale)``. The floor of 1 keeps a small
    bf16-versus-fp32 drift from failing when the CPU output is near zero.
    A larger wind head may move by that fraction of its magnitude.
    This does not convert physical units. It only compares two forwards.
    """
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise ValueError("device_parity_tol must be a finite number >= 0")
    if not math.isfinite(float(tolerance)) or float(tolerance) < 0:
        raise ValueError("device_parity_tol must be a finite number >= 0")
    output = float(output_max_abs)
    loss = float(loss_max_abs)
    if not math.isfinite(output) or not math.isfinite(loss):
        worst = float("inf")
    else:
        worst = max(output, loss)
    scale = float(reference_scale)
    if not math.isfinite(scale) or scale < 1.0:
        scale = 1.0
    limit = float(tolerance) * scale
    return {
        "output_max_abs": output,
        "loss_max_abs": loss,
        "max_abs_diff": worst,
        "reference_scale": scale,
        "tolerance": float(tolerance),
        "limit": limit,
        "ok": bool(worst <= limit),
    }


def warn_if_parity_failed(result: dict) -> None:
    """Loud warning when the seeded forward disagrees. Training still continues."""
    if result.get("ok"):
        return
    message = (
        "WARNING: device parity check failed. "
        f"max abs diff {result['max_abs_diff']:.6g} exceeds limit {result['limit']:.6g} "
        f"(tolerance {result['tolerance']:.6g}, scale {result['reference_scale']:.6g}). "
        "device_parity_ok is false. Training continues. This is not a forecast."
    )
    LOGGER.warning(message)
    print(message)


def seeded_parity_batch(batch_size: int, channels: int, side: int, seed: int = 0):
    """One fixed CPU batch. The same tensors are copied onto the device."""
    torch_module = _torch()
    generator = torch_module.Generator(device="cpu")
    generator.manual_seed(int(seed))
    images = torch_module.randn(
        int(batch_size), int(channels), int(side), int(side), generator=generator
    )
    winds = torch_module.rand(int(batch_size), 1, generator=generator)
    return images, winds


def _forward_wind_loss(model, images, winds):
    import torch.nn.functional as F

    torch_module = _torch()
    was_training = model.training
    model.eval()
    try:
        with torch_module.no_grad():
            output = model(images)
            wind = output[0] if isinstance(output, tuple) else output
            loss = F.smooth_l1_loss(wind.float().view(-1), winds.float().view(-1))
        return wind.detach().float().cpu(), loss.detach().float().cpu()
    finally:
        model.train(was_training)


def compare_model_parity(model, images, winds, device, tolerance: float, *, policy=None) -> dict:
    """Run one forward on ``device`` and on CPU. Record the max absolute gap.

    The training model is copied. Its weights are not updated. The device
    copy uses the same autocast policy as training when one is passed.
    """
    import copy

    cpu_model = copy.deepcopy(model).to("cpu").eval()
    device_model = copy.deepcopy(model).to(device).eval()
    cpu_wind, cpu_loss = _forward_wind_loss(cpu_model, images, winds)
    if policy is None:
        policy = amp_policy_for(device)
    with autocast_context(policy):
        dev_wind, dev_loss = _forward_wind_loss(
            device_model, images.to(device), winds.to(device)
        )
    output_diff = float((dev_wind - cpu_wind).abs().max())
    loss_diff = abs(float(dev_loss) - float(cpu_loss))
    scale = max(1.0, float(cpu_wind.abs().max()), abs(float(cpu_loss)))
    result = parity_verdict(output_diff, loss_diff, float(tolerance), scale)
    warn_if_parity_failed(result)
    return result
