"""Model output for one image, or four frames.

A missing checkpoint produces no number. A smoke checkpoint is a software
check. Neither is a forecast. Dropout standard deviation is not a calibrated
prediction interval. Project class names are applied only when the checkpoint
recorded one wind-averaging period.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cloudie.preprocessing import NormStats, leave_unflipped, prepare_image
from cloudie.schema import wind_to_imd5
from cloudie.train import NOT_A_FORECAST, SMOKE_BANNER, build_net

MISSING_CHECKPOINT_BANNER = (
    "No checkpoint was found. There is no model output. This is not a forecast."
)
MISSING_PIXEL_BANNER = (
    "This crop failed the missing-pixel gate. There is no model output. "
    "A person reviews a failed crop. This is not a forecast."
)
NOT_CALIBRATED = "Dropout standard deviation is not a calibrated prediction interval."


@dataclass(frozen=True)
class Prediction:
    wind_kt: float | None
    dropout_std_kt: float | None
    project_class: str | None
    present_as_forecast: bool
    show_number: bool
    banner: str
    review: tuple[str, ...]
    is_synthetic: bool


def missing_checkpoint(path: str | Path) -> Prediction:
    return Prediction(
        wind_kt=None,
        dropout_std_kt=None,
        project_class=None,
        present_as_forecast=False,
        show_number=False,
        banner=f"{MISSING_CHECKPOINT_BANNER} Path: {path}",
        review=(),
        is_synthetic=False,
    )


def review_flags(
    wind_kt: float | None,
    dropout_std_kt: float | None,
    previous_advisory_kt: float | None,
    *,
    edges: tuple[float, ...],
    names: tuple[str, ...],
    wind_averaging: str | None,
    advisory_jump_kt: float,
    failed_gate: bool = False,
) -> tuple[str, ...]:
    """Review notes for a person. These are not a warning product."""
    flags: list[str] = []
    if failed_gate:
        flags.append("missing-pixel gate")
        return tuple(flags)
    if wind_kt is None:
        return tuple(flags)
    if previous_advisory_kt is not None:
        if abs(float(wind_kt) - float(previous_advisory_kt)) >= float(advisory_jump_kt):
            flags.append("jump versus the last advisory")
        if wind_averaging:
            predicted = wind_to_imd5(float(wind_kt), edges, names)
            previous = wind_to_imd5(float(previous_advisory_kt), edges, names)
            if predicted != previous:
                flags.append("category change")
    if (
        dropout_std_kt is not None
        and wind_averaging
        and wind_to_imd5(float(wind_kt) - float(dropout_std_kt), edges, names)
        != wind_to_imd5(float(wind_kt) + float(dropout_std_kt), edges, names)
    ):
        flags.append("dropout spread crosses a project class boundary")
    return tuple(flags)


def _refusal(
    banner: str,
    *,
    review: tuple[str, ...] = (),
    is_synthetic: bool = False,
) -> Prediction:
    return Prediction(
        wind_kt=None,
        dropout_std_kt=None,
        project_class=None,
        present_as_forecast=False,
        show_number=False,
        banner=banner,
        review=review,
        is_synthetic=is_synthetic,
    )


def _load_checkpoint(path: Path) -> dict:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _banner(payload: dict) -> str:
    if bool(payload.get("is_synthetic")):
        return (
            SMOKE_BANNER
            + " The number is a software check. It is not a forecast."
        )
    return str(payload.get("honesty") or NOT_A_FORECAST)


def _advisory_jump_kt(config: dict) -> float:
    predict_cfg = config.get("predict")
    if not isinstance(predict_cfg, dict) or "advisory_jump_kt" not in predict_cfg:
        raise ValueError("config predict.advisory_jump_kt is required")
    return float(predict_cfg["advisory_jump_kt"])


def _edges(payload: dict, config: dict) -> tuple[float, ...]:
    raw = payload.get("edges_kt", config["data"]["imd5_edges_kt"])
    return tuple(float(edge) for edge in raw)


def _names(payload: dict, config: dict) -> tuple[str, ...]:
    raw = payload.get("class_names", config["data"]["imd5_names"])
    return tuple(str(name) for name in raw)


def _dropout_std(model, tensor, n_samples: int) -> float:
    import torch

    from cloudie.model import enable_mc_dropout

    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    enable_mc_dropout(model)
    draws = []
    with torch.no_grad():
        for _ in range(n_samples):
            output = model(tensor)
            wind = output[0] if isinstance(output, tuple) else output
            draws.append(wind.reshape(-1))
    stacked = torch.stack(draws, dim=0)
    return float(stacked.std(dim=0, unbiased=True).reshape(-1)[0])


def predict_images(
    config: dict,
    checkpoint_path: str | Path,
    images: list[np.ndarray],
    *,
    previous_advisory_kt: float | None = None,
    hemisphere: str | None = None,
    n_samples: int | None = None,
) -> Prediction:
    """Run the checkpoint. ``previous_advisory_kt`` is a review input only."""
    path = Path(checkpoint_path)
    if not path.is_file():
        return missing_checkpoint(path)
    jump = _advisory_jump_kt(config)
    payload = _load_checkpoint(path)
    if str(config["model"].get("name", "simple_cnn")) != str(payload["model_name"]):
        raise ValueError("config model.name does not match the checkpoint")
    if int(config["data"]["image_size"]) != int(payload["image_size"]):
        raise ValueError("config image_size does not match the checkpoint")
    if payload.get("hidden") is not None and int(config["model"].get("hidden", 16)) != int(
        payload["hidden"]
    ):
        raise ValueError("config model.hidden does not match the checkpoint")
    if str(payload.get("norm_source_split")) != "train":
        raise ValueError("checkpoint normalization was not fit on train")
    expected = int(payload.get("n_frames", 1))
    synthetic = bool(payload.get("is_synthetic"))
    if len(images) != expected:
        return _refusal(
            (
                f"This checkpoint expects {expected} infrared frames and received "
                f"{len(images)}. There is no model output. This is not a forecast."
            ),
            is_synthetic=synthetic,
        )
    stats = NormStats(
        mean=float(payload["norm_mean"]),
        std=float(payload["norm_std"]),
        source_split="train",
    )
    image_size = int(payload["image_size"])
    max_missing = float(payload["max_missing_fraction"])
    prepared: list[np.ndarray] = []
    try:
        for image in images:
            prepared.append(
                prepare_image(
                    leave_unflipped(image, hemisphere),
                    image_size,
                    stats,
                    max_missing,
                )
            )
    except ValueError as exc:
        if "missing fraction" not in str(exc):
            raise
        edges = _edges(payload, config)
        names = _names(payload, config)
        return _refusal(
            MISSING_PIXEL_BANNER,
            review=review_flags(
                None,
                None,
                previous_advisory_kt,
                edges=edges,
                names=names,
                wind_averaging=payload.get("wind_averaging"),
                advisory_jump_kt=jump,
                failed_gate=True,
            ),
            is_synthetic=synthetic,
        )
    import torch

    stack = np.stack(prepared, axis=0)
    tensor = torch.from_numpy(np.ascontiguousarray(stack))
    if expected == 1:
        model_input = tensor
    else:
        model_input = tensor.unsqueeze(0)
    size_head = bool(payload.get("size_head", False))
    model = build_net(config, size_head=size_head)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    with torch.no_grad():
        output = model(model_input)
        wind_hat = output[0] if isinstance(output, tuple) else output
        wind_kt = float(wind_hat.detach().cpu().reshape(-1)[0])
    samples = int(
        config["model"].get("mc_samples", 20) if n_samples is None else n_samples
    )
    dropout_std = _dropout_std(model, model_input, samples)
    edges = _edges(payload, config)
    names = _names(payload, config)
    averaging = payload.get("wind_averaging")
    averaging = None if averaging in (None, "") else str(averaging)
    project_class = wind_to_imd5(wind_kt, edges, names) if averaging else None
    return Prediction(
        wind_kt=wind_kt,
        dropout_std_kt=dropout_std,
        project_class=project_class,
        present_as_forecast=False,
        show_number=True,
        banner=_banner(payload),
        review=review_flags(
            wind_kt,
            dropout_std,
            previous_advisory_kt,
            edges=edges,
            names=names,
            wind_averaging=averaging,
            advisory_jump_kt=jump,
        ),
        is_synthetic=synthetic,
    )


def format_prediction(prediction: Prediction) -> str:
    """Text for the CLI. The heading is never a forecast label."""
    lines = [
        prediction.banner,
        "present_as_forecast: false",
        NOT_CALIBRATED,
    ]
    if prediction.show_number and prediction.wind_kt is not None:
        lines.append(f"model output (kt): {prediction.wind_kt:.2f}")
        lines.append("This number is not a forecast.")
        if prediction.dropout_std_kt is not None:
            lines.append(f"dropout std (kt): {prediction.dropout_std_kt:.2f}")
        if prediction.project_class is not None:
            lines.append(
                f"project class: {prediction.project_class}. Not an IMD warning."
            )
    if prediction.review:
        lines.append("A person needs to review this output.")
        lines.extend(f"review: {item}" for item in prediction.review)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from cloudie.config import load_config
    from cloudie.data_loader import load_image

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, nargs="+", required=True)
    parser.add_argument("--previous-advisory-kt", type=float, default=None)
    parser.add_argument("--hemisphere", type=str, default=None)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    prediction = predict_images(
        config,
        args.checkpoint,
        [load_image(path) for path in args.image],
        previous_advisory_kt=args.previous_advisory_kt,
        hemisphere=args.hemisphere,
    )
    print(format_prediction(prediction))
    return 0 if prediction.show_number else 2


if __name__ == "__main__":
    raise SystemExit(main())
