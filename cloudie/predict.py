"""Model output for one image, or four frames.

The reported number is wind speed in knots. The intensity category is this
project's grouping of that wind, not an official IMD class. Confidence mixes
the image gate with dropout spread, and with residual quantiles when a
checkpoint has them. Evaluate residual quantiles are half-widths; the score
uses the full width, twice that half-width. The label stays uncalibrated
until a matching real-data run record has a coverage check or a held-out
storm check. A missing checkpoint produces no number. A smoke checkpoint
is a software check. This is not an operational forecast. A single frame
cannot forecast landfall. Dropout standard deviation is not the displayed
score and is not a calibrated prediction interval.

The gate, the geometry, and the checkpoint load live in ``src.inference``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from cloudie.inference import IntensityEstimate
from cloudie.schema import wind_to_imd5
from cloudie.train import NOT_A_FORECAST, SMOKE_BANNER

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
    wind_kmh: float | None = None
    confidence_status: str | None = None
    grouping_note: str | None = None
    operational_disclaimer: str | None = None
    coastal_heading: str | None = None
    nearest_coastal_states: tuple[str, ...] | None = None
    landfall_note: str | None = None
    coast_reference_note: str | None = None
    gate_reason: str | None = None


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


def _plausible_kt(config: dict) -> tuple[float, float]:
    raw = config.get("data", {}).get("plausible_wind_kt", DEFAULT_PLAUSIBLE_KT)
    low, high = float(raw[0]), float(raw[1])
    if high <= low:
        raise ValueError("data.plausible_wind_kt must be a rising span")
    return low, high


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


def _from_estimate(
    config: dict,
    estimate,
    previous_advisory_kt: float | None,
) -> Prediction:
    """Map an estimate onto ``Prediction``. The missing-checkpoint banner is not built here."""
    jump = _advisory_jump_kt(config)
    context = dict(
        operational_disclaimer=estimate.operational_disclaimer,
        coastal_heading=estimate.coastal_heading,
        nearest_coastal_states=estimate.nearest_coastal_states,
        landfall_note=estimate.landfall_note,
        coast_reference_note=estimate.coast_reference_note,
        gate_reason=estimate.gate_reason,
    )
    if estimate.outcome == "missing_pixels":
        edges = estimate.edges or tuple(float(edge) for edge in config["data"]["imd5_edges_kt"])
        names = estimate.names or tuple(str(name) for name in config["data"]["imd5_names"])
        return replace(
            _refusal(
                MISSING_PIXEL_BANNER,
                review=review_flags(
                    None,
                    None,
                    previous_advisory_kt,
                    edges=edges,
                    names=names,
                    wind_averaging=estimate.wind_averaging,
                    advisory_jump_kt=jump,
                    failed_gate=True,
                ),
                is_synthetic=estimate.is_synthetic,
            ),
            **context,
        )
    if estimate.outcome == "frame_count":
        return replace(_refusal(estimate.banner, is_synthetic=estimate.is_synthetic), **context)
    if estimate.outcome in {"not_cyclone", "no_run_record"}:
        return replace(_refusal(estimate.banner, is_synthetic=estimate.is_synthetic), **context)
    if estimate.outcome == "outside_range":
        low_kt, high_kt = _plausible_kt(config)
        return replace(
            _refusal(
                _scale_banner(low_kt, high_kt, outside_training=estimate.outside_training),
                is_synthetic=estimate.is_synthetic,
            ),
            **context,
        )
    # The category is the project bin of the predicted wind, so the knots
    # and the intensity label describe the same number.
    low_kt, high_kt = _plausible_kt(config)
    project_class = wind_to_imd5(estimate.wind_kt, estimate.edges, estimate.names)
    _share, table, below = category_confidence(
        estimate.samples, project_class, estimate.edges, estimate.names, low=low_kt, high=high_kt
    )
    probability_rows = tuple((intensity_label(name), table[name]) for name in estimate.names)
    if project_class is None or below > 1e-8:
        probability_rows = ((BELOW_CATEGORY, below), *probability_rows)
    review = review_flags(
        estimate.wind_kt,
        estimate.dropout_std_kt,
        previous_advisory_kt,
        edges=estimate.edges,
        names=estimate.names,
        wind_averaging=estimate.wind_averaging,
        advisory_jump_kt=jump,
    )
    if not estimate.matched_run:
        note = str(config["predict"]["no_run_record_review"]).strip()
        if note not in review:
            review = review + (note,)
    banner = _banner({"is_synthetic": estimate.is_synthetic, "honesty": estimate.honesty})
    return Prediction(
        wind_kt=estimate.wind_kt,
        dropout_std_kt=estimate.dropout_std_kt,
        project_class=project_class,
        present_as_forecast=False,
        show_number=True,
        banner=banner,
        review=review,
        is_synthetic=estimate.is_synthetic,
        confidence=estimate.confidence,
        category_label=intensity_label(project_class),
        class_probabilities=probability_rows,
        wind_kmh=estimate.wind_kmh,
        confidence_status=estimate.confidence_status,
        grouping_note=estimate.grouping_note,
        **context,
    )


def predict_images(
    config: dict,
    checkpoint_path: str | Path,
    images: list[np.ndarray],
    *,
    previous_advisory_kt: float | None = None,
    hemisphere: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    metadata: dict | None = None,
) -> Prediction:
    """Run the checkpoint. ``previous_advisory_kt`` is a review input only.

    This is not an operational forecast. A frame that fails the infrared
    gate returns no wind speed.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        return missing_checkpoint(path)
    estimate = IntensityEstimate.predict(
        config,
        path,
        images,
        hemisphere=hemisphere,
        latitude=latitude,
        longitude=longitude,
        metadata=metadata,
    )
    return _from_estimate(config, estimate, previous_advisory_kt)


def _report_lines(prediction: Prediction) -> list[str]:
    """Disclaimer and coastal context. Empty for a bare missing-checkpoint result."""
    lines: list[str] = []
    if prediction.operational_disclaimer:
        lines.append(prediction.operational_disclaimer)
    if prediction.coastal_heading:
        if prediction.nearest_coastal_states:
            joined = ", ".join(prediction.nearest_coastal_states)
            lines.append(f"{prediction.coastal_heading}: {joined}")
            if prediction.landfall_note:
                lines.append(prediction.landfall_note)
            if prediction.coast_reference_note:
                lines.append(prediction.coast_reference_note)
        else:
            lines.append(f"{prediction.coastal_heading}: null")
    return lines


def format_prediction(prediction: Prediction) -> str:
    """Text for the CLI. Wind, project category, and uncalibrated confidence.

    This is not an operational forecast.
    """
    if not prediction.show_number or prediction.wind_kt is None:
        return "\n".join(
            [
                prediction.banner,
                "present_as_forecast: false",
                *_report_lines(prediction),
            ]
        )
    lines = [f"wind speed (kt): {prediction.wind_kt:.2f}"]
    if prediction.wind_kmh is not None:
        lines.append(f"wind speed (km/h): {prediction.wind_kmh:.2f}")
    lines.append(f"intensity category: {prediction.category_label}")
    if prediction.grouping_note:
        lines.append(prediction.grouping_note)
    if prediction.confidence is not None:
        status = (
            f" ({prediction.confidence_status})" if prediction.confidence_status else ""
        )
        lines.append(f"confidence: {prediction.confidence * 100:.0f}%{status}")
    lines.extend(_report_lines(prediction))
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
    parser.add_argument("--latitude", type=float, default=None)
    parser.add_argument("--longitude", type=float, default=None)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    prediction = predict_images(
        config,
        args.checkpoint,
        [load_image(path) for path in args.image],
        previous_advisory_kt=args.previous_advisory_kt,
        hemisphere=args.hemisphere,
        latitude=args.latitude,
        longitude=args.longitude,
    )
    print(format_prediction(prediction))
    return 0 if prediction.show_number else 2


if __name__ == "__main__":
    raise SystemExit(main())
