"""Model output for one image, or four frames.

The reported number is wind speed in knots. The intensity category is the
project bin of that wind. Confidence is the share of stochastic forward
passes that fall in the same bin. A missing checkpoint produces no number.
A smoke checkpoint is a software check. Neither is a forecast. Dropout
standard deviation is not the displayed score and is not a calibrated
prediction interval.
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

# Display names for the project bins. The code names stay on the checkpoint.
CATEGORY_LABELS = {
    "D_DD": "Depression / Deep Depression",
    "CS": "Cyclonic Storm",
    "SCS": "Severe Cyclonic Storm",
    "VSCS": "Very Severe Cyclonic Storm",
    "ESCS_SUCS": "Extremely Severe / Super Cyclonic Storm",
}
BELOW_CATEGORY = "Below depression"


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
    confidence: float | None = None
    category_label: str | None = None
    class_probabilities: tuple[tuple[str, float], ...] = ()


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


def intensity_label(project_class: str | None) -> str:
    """Human-readable intensity category for a project class code."""
    if project_class is None:
        return BELOW_CATEGORY
    return CATEGORY_LABELS.get(project_class, project_class)


def category_span(
    project_class: str | None, edges: tuple[float, ...], names: tuple[str, ...]
) -> str:
    """Knot range for the category the predicted wind fell in."""
    if project_class is None:
        return f"under {edges[0]:.0f} kt"
    index = names.index(project_class)
    lower = edges[index]
    if index + 1 < len(edges):
        return f"{lower:.0f}–{edges[index + 1]:.0f} kt"
    return f"{lower:.0f} kt and above"


# A reported wind stays inside this span unless config sets another one.
# Samples outside it do not count as confidence in a category.
DEFAULT_PLAUSIBLE_KT = (0.0, 200.0)
# Image mean must sit within this many training standard deviations.
TRAINING_MEAN_WIDTH = 3.0


def category_confidence(
    winds,
    category: str | None,
    edges: tuple[float, ...],
    names: tuple[str, ...],
    *,
    low: float = DEFAULT_PLAUSIBLE_KT[0],
    high: float = DEFAULT_PLAUSIBLE_KT[1],
) -> tuple[float, dict[str, float], float]:
    """Share of samples in ``category`` and inside ``low``–``high`` knots.

    A sample outside that span is not evidence for any category, including
    the top bin. ``below`` is the share that is inside the span and under
    the first category edge.
    """
    values = np.asarray(winds, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("wind samples did not produce a probability")
    if high <= low:
        raise ValueError("plausible wind span is empty")
    table = {name: 0.0 for name in names}
    below = 0.0
    support = 0.0
    for value in values:
        if float(value) < low or float(value) > high:
            continue
        label = wind_to_imd5(float(value), edges, names)
        if label is None:
            below += 1.0
        elif label in table:
            table[label] += 1.0
        if label == category or (category is None and label is None):
            support += 1.0
    total = float(values.size)
    table = {name: count / total for name, count in table.items()}
    return support / total, table, below / total


def _banner(payload: dict) -> str:
    if bool(payload.get("is_synthetic")):
        return (
            SMOKE_BANNER
            + " The number is a software check. It is not a forecast."
        )
    return str(payload.get("honesty") or NOT_A_FORECAST)


def _mc_samples(config: dict) -> int:
    count = int(config["model"].get("mc_samples", 20))
    if count < 1:
        raise ValueError("model.mc_samples must be at least 1")
    return count


def _wind_value(output) -> float:
    wind = output[0] if isinstance(output, tuple) else output
    return float(wind.detach().cpu().reshape(-1)[0])


def _wind_samples(model, model_input, n_samples: int, seed: int) -> tuple[float, np.ndarray]:
    """Point wind with dropout off, then seeded samples with dropout on.

    The point wind is the reported speed. The samples are only the
    confidence estimate. The global RNG is restored before returning.
    """
    import torch

    from cloudie.model import enable_mc_dropout

    state = torch.get_rng_state()
    try:
        model.eval()
        with torch.no_grad():
            point = _wind_value(model(model_input))
            if n_samples == 1:
                return point, np.asarray([point], dtype=np.float64)
            enable_mc_dropout(model)
            torch.manual_seed(int(seed))
            draws = [_wind_value(model(model_input)) for _ in range(n_samples)]
        return point, np.asarray(draws, dtype=np.float64)
    finally:
        model.eval()
        torch.set_rng_state(state)


def _plausible_kt(config: dict) -> tuple[float, float]:
    raw = config.get("data", {}).get("plausible_wind_kt", DEFAULT_PLAUSIBLE_KT)
    low, high = float(raw[0]), float(raw[1])
    if high <= low:
        raise ValueError("data.plausible_wind_kt must be a rising span")
    return low, high


def _field_mean(images: list[np.ndarray]) -> float:
    finite = [
        np.asarray(image, dtype=np.float64)[np.isfinite(image)] for image in images
    ]
    finite = [values for values in finite if values.size]
    if not finite:
        raise ValueError("image has no finite pixels")
    return float(np.concatenate(finite).mean())


def _outside_training_values(field_mean: float, norm_mean: float, norm_std: float) -> bool:
    scale = max(float(norm_std), 1e-6)
    return abs(float(field_mean) - float(norm_mean)) > TRAINING_MEAN_WIDTH * scale


def _scale_banner(low: float, high: float, *, outside_training: bool) -> str:
    span = (
        f"Reported winds stay between {low:.0f} and {high:.0f} kt: "
        "below depression, depression, cyclonic storm, severe, very severe, "
        "and extremely severe or super cyclonic storm."
    )
    if outside_training:
        reason = (
            "This infrared frame is outside the values the checkpoint was trained on. "
            "No wind speed or confidence is reported."
        )
    else:
        reason = (
            "The model wind is outside that range. "
            "No wind speed or confidence is reported."
        )
    return f"{reason} {span}"


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


def predict_images(
    config: dict,
    checkpoint_path: str | Path,
    images: list[np.ndarray],
    *,
    previous_advisory_kt: float | None = None,
    hemisphere: str | None = None,
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
    seed = int(config.get("project", {}).get("seed", 0))
    low_kt, high_kt = _plausible_kt(config)
    wind_kt, samples = _wind_samples(model, model_input, _mc_samples(config), seed)
    edges = _edges(payload, config)
    names = _names(payload, config)
    averaging = payload.get("wind_averaging")
    averaging = None if averaging in (None, "") else str(averaging)
    outside_training = _outside_training_values(
        _field_mean(images), float(payload["norm_mean"]), float(payload["norm_std"])
    )
    if outside_training or not (low_kt <= wind_kt <= high_kt):
        return _refusal(
            _scale_banner(low_kt, high_kt, outside_training=outside_training),
            is_synthetic=synthetic,
        )
    # The category is the project bin of the predicted wind, so the knots
    # and the intensity label describe the same number.
    project_class = wind_to_imd5(wind_kt, edges, names)
    confidence, table, below = category_confidence(
        samples, project_class, edges, names, low=low_kt, high=high_kt
    )
    probability_rows = tuple((intensity_label(name), table[name]) for name in names)
    if project_class is None or below > 1e-8:
        probability_rows = ((BELOW_CATEGORY, below), *probability_rows)
    return Prediction(
        wind_kt=wind_kt,
        dropout_std_kt=None,
        project_class=project_class,
        present_as_forecast=False,
        show_number=True,
        banner=_banner(payload),
        review=review_flags(
            wind_kt,
            None,
            previous_advisory_kt,
            edges=edges,
            names=names,
            wind_averaging=averaging,
            advisory_jump_kt=jump,
        ),
        is_synthetic=synthetic,
        confidence=confidence,
        category_label=intensity_label(project_class),
        class_probabilities=probability_rows,
    )


def format_prediction(prediction: Prediction) -> str:
    """Text for the CLI. Wind, intensity category, and category confidence."""
    if not prediction.show_number or prediction.wind_kt is None:
        return "\n".join(
            [
                prediction.banner,
                "present_as_forecast: false",
            ]
        )
    lines = [
        f"wind speed (kt): {prediction.wind_kt:.2f}",
        f"intensity category: {prediction.category_label}",
    ]
    if prediction.confidence is not None:
        lines.append(f"confidence: {prediction.confidence * 100:.0f}%")
    if prediction.is_synthetic:
        lines.append(prediction.banner)
    if prediction.review:
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
