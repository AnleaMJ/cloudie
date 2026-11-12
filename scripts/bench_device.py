"""Synthetic speed check for simplecnn and resnet18.

Runs a fixed number of steps on one device at the native NASA (366) and
TCIR (201) sizes. Prints JSON and writes it under ``artifacts/bench/``.
The file is marked ``is_synthetic`` and ``result: false``. This is not a
model score and it does not download imagery.

The ``CLOUDIE_ALLOW_GPU`` gate is the same one training uses. This script
does not set ``PYTORCH_ENABLE_XPU_FALLBACK``. Set
``PYTORCH_DEBUG_XPU_FALLBACK=1`` in the environment when you want a log of
ops that fell back to CPU, and run this once that way on the Arc machine.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

NATIVE_SIZES = (("nasa", 366), ("tcir", 201))
MODELS = ("simplecnn", "resnet18")


def _bench_config(model_name: str) -> dict:
    canonical = "simple_cnn" if model_name == "simplecnn" else "resnet18"
    return {
        "model": {
            "name": canonical,
            "pretrained": False,
            "dropout": 0.0,
            "n_frames": 4,
            "hidden": 16,
        },
        "data": {
            "channels": 1,
            "imd5_names": ["D_DD", "CS", "SCS", "VSCS", "ESCS_SUCS"],
        },
    }


def _one_model(
    model_name: str,
    dataset: str,
    side: int,
    *,
    device,
    policy,
    steps: int,
    batch_size: int,
    tolerance: float,
) -> dict:
    import torch
    import torch.nn.functional as F

    from src.device import (
        autocast_context,
        compare_model_parity,
        make_grad_scaler,
        peak_memory_bytes,
        reset_peak_memory,
        seeded_parity_batch,
    )
    from src.train import build_net

    if steps < 1:
        raise ValueError("steps must be at least 1")
    if batch_size < 1:
        raise ValueError("batch size must be at least 1")
    config = _bench_config(model_name)
    model = build_net(config, size_head=False).to(device)
    images, winds = seeded_parity_batch(batch_size, 1, side, seed=0)
    parity = compare_model_parity(model, images, winds, device, tolerance, policy=policy)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    scaler = make_grad_scaler(policy)
    reset_peak_memory(device)
    model.train()
    images = images.to(device)
    winds = winds.to(device)
    warmup_s = 0.0
    measured = 0.0
    measured_steps = 0
    for step in range(int(steps)):
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(policy):
            output = model(images)
            wind = output[0] if isinstance(output, tuple) else output
            loss = F.smooth_l1_loss(wind.float().view(-1), winds.float().view(-1))
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "xpu":
            xpu = getattr(torch, "xpu", None)
            sync = getattr(xpu, "synchronize", None) if xpu is not None else None
            if callable(sync):
                sync()
        elapsed = time.perf_counter() - start
        if step == 0:
            warmup_s = elapsed
        else:
            measured += elapsed
            measured_steps += 1
    if measured_steps == 0:
        measured = warmup_s
        measured_steps = 1
    images_per_sec = (measured_steps * int(batch_size)) / measured if measured > 0 else None
    return {
        "model": model_name,
        "dataset": dataset,
        "image_size": int(side),
        "batch_size": int(batch_size),
        "steps": int(steps),
        "precision": policy.precision,
        "images_per_sec": None if images_per_sec is None else float(images_per_sec),
        "peak_memory_bytes": peak_memory_bytes(device),
        "warmup_s": float(warmup_s),
        "device_parity_ok": bool(parity["ok"]),
        "device_parity_max_abs_diff": parity["max_abs_diff"],
        "device_parity_output_max_abs": parity["output_max_abs"],
        "device_parity_loss_max_abs": parity["loss_max_abs"],
        "is_synthetic": True,
        "result": False,
    }


def run_bench(
    *,
    device: str = "cpu",
    steps: int = 200,
    batch_size: int = 2,
    tolerance: float = 0.05,
    out_dir: str | Path | None = None,
) -> dict:
    """Time each model at each native size. Writes JSON and returns it."""
    from src.device import amp_policy_for, resolve_device

    torch_device = resolve_device(device)
    policy = amp_policy_for(torch_device)
    runs = [
        _one_model(
            model_name,
            dataset,
            side,
            device=torch_device,
            policy=policy,
            steps=int(steps),
            batch_size=int(batch_size),
            tolerance=float(tolerance),
        )
        for model_name in MODELS
        for dataset, side in NATIVE_SIZES
    ]
    payload = {
        "is_synthetic": True,
        "result": False,
        "device": torch_device.type,
        "device_requested": str(device),
        "precision": policy.precision,
        "steps": int(steps),
        "batch_size": int(batch_size),
        "use_grad_scaler": bool(policy.use_grad_scaler),
        "runs": runs,
        "note": (
            "Synthetic tensors only. Not a forecast and not a model result. "
            "PYTORCH_ENABLE_XPU_FALLBACK is not set by this script."
        ),
    }
    text = json.dumps(payload, indent=2)
    print(text)
    destination = Path(out_dir) if out_dir is not None else Path("artifacts/bench")
    destination.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = destination / f"bench-{torch_device.type}-{stamp}-{time.time_ns()}.json"
    path.write_text(text + "\n", encoding="utf-8")
    payload["path"] = str(path)
    return payload


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(
        description="Synthetic device benchmark. Not a model result. Does not download data."
    )
    parser.add_argument("--device", choices=("auto", "xpu", "cuda", "cpu"), default="cpu")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    return run_bench(
        device=args.device,
        steps=args.steps,
        batch_size=args.batch_size,
        tolerance=args.tolerance,
        out_dir=args.out,
    )


if __name__ == "__main__":
    main()
