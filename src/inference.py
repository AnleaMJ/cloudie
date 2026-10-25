"""Image gate, scale-relative geometry, and an honest intensity estimate.

This is not an operational forecast. A single infrared frame cannot
forecast landfall. Confidence is uncalibrated unless a matching
real-data run record includes a coverage check. The intensity category
is this project's grouping on the configured knot bins, not an official
IMD class.

The network is reached only through :class:`src.model_access.ModelAccess`.
Letterboxing happens before ``cloudie.preprocessing.prepare_image``, which
still resizes a square to the checkpoint size. ``src.preprocessing`` has
no ``leave_unflipped``, so that call stays on the cloudie copy until the
trees are consolidated.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.model_access import ModelAccess
from src.schema import wind_to_imd5


def _section(config: dict, *keys: str):
    cursor = config
    walked: list[str] = []
    for key in keys:
        walked.append(key)
        if not isinstance(cursor, dict) or key not in cursor:
            raise ValueError(f"config {'.'.join(walked)} is required")
        cursor = cursor[key]
    return cursor


def _finite(raw, path: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"config {path} must be a number") from exc
    if not math.isfinite(value):
        raise ValueError(f"config {path} must be finite")
    return value


@dataclass(frozen=True)
class GateSettings:
    """Thresholds for the infrared-cyclone gate. All of them come from config."""

    refusal: str
    grayscale_tolerance: float
    min_std: float
    min_symmetry: float
    min_cold_concentration: float
    cold_radius_fraction: float
    cold_tail_fraction: float
    eye_radius_fraction: float
    eyewall_outer_fraction: float
    outer_inner_fraction: float
    outer_outer_fraction: float
    min_wall_contrast: float
    radial_bins: int
    cold_value: str
    max_dropout_std_kt: float
    concentration_reference: float

    @staticmethod
    def from_config(config: dict) -> GateSettings:
        raw = _section(config, "predict", "gate")
        settings = GateSettings(
            refusal=str(raw["refusal"]).strip(),
            grayscale_tolerance=_finite(raw["grayscale_tolerance"], "predict.gate.grayscale_tolerance"),
            min_std=_finite(raw["min_std"], "predict.gate.min_std"),
            min_symmetry=_finite(raw["min_symmetry"], "predict.gate.min_symmetry"),
            min_cold_concentration=_finite(
                raw["min_cold_concentration"], "predict.gate.min_cold_concentration"
            ),
            cold_radius_fraction=_finite(
                raw["cold_radius_fraction"], "predict.gate.cold_radius_fraction"
            ),
            cold_tail_fraction=_finite(raw["cold_tail_fraction"], "predict.gate.cold_tail_fraction"),
            eye_radius_fraction=_finite(raw["eye_radius_fraction"], "predict.gate.eye_radius_fraction"),
            eyewall_outer_fraction=_finite(
                raw["eyewall_outer_fraction"], "predict.gate.eyewall_outer_fraction"
            ),
            outer_inner_fraction=_finite(raw["outer_inner_fraction"], "predict.gate.outer_inner_fraction"),
            outer_outer_fraction=_finite(raw["outer_outer_fraction"], "predict.gate.outer_outer_fraction"),
            min_wall_contrast=_finite(raw["min_wall_contrast"], "predict.gate.min_wall_contrast"),
            radial_bins=int(raw["radial_bins"]),
            cold_value=str(raw["cold_value"]).strip(),
            max_dropout_std_kt=_finite(raw["max_dropout_std_kt"], "predict.gate.max_dropout_std_kt"),
            concentration_reference=_finite(
                raw["concentration_reference"], "predict.gate.concentration_reference"
            ),
        )
        if not settings.refusal:
            raise ValueError("config predict.gate.refusal is empty")
        if settings.cold_value not in {"auto", "low", "high"}:
            raise ValueError("config predict.gate.cold_value must be auto, low, or high")
        if settings.radial_bins < 2:
            raise ValueError("config predict.gate.radial_bins must be at least 2")
        if settings.concentration_reference <= 0:
            raise ValueError("config predict.gate.concentration_reference must be positive")
        if settings.max_dropout_std_kt <= 0:
            raise ValueError("config predict.gate.max_dropout_std_kt must be positive")
        radii = (
            settings.eye_radius_fraction,
            settings.eyewall_outer_fraction,
            settings.outer_inner_fraction,
            settings.outer_outer_fraction,
        )
        if not (0 < radii[0] < radii[1] < radii[2] < radii[3] <= 1):
            raise ValueError("config predict.gate radii must increase and stay within (0, 1]")
        if not 0 < settings.cold_radius_fraction <= 1:
            raise ValueError("config predict.gate.cold_radius_fraction must be in (0, 1]")
        if not 0 < settings.cold_tail_fraction < 0.5:
            raise ValueError("config predict.gate.cold_tail_fraction must be in (0, 0.5)")
        if settings.min_std < 0:
            raise ValueError("config predict.gate.min_std must be >= 0")
        return settings


@dataclass(frozen=True)
class GateReport:
    """One frame's gate. ``score`` is 0 when the frame is refused."""

    accepted: bool
    reason: str
    score: float
    field: np.ndarray | None
    symmetry: float | None = None
    cold_concentration: float | None = None
    wall_contrast: float | None = None
    eye_contrast: float | None = None
    polarity: str | None = None

    def feature_map(self) -> dict[str, float]:
        """Scale-relative summaries. None of these are pixel counts."""
        return {
            "symmetry": 0.0 if self.symmetry is None else float(self.symmetry),
            "cold_concentration": (
                0.0 if self.cold_concentration is None else float(self.cold_concentration)
            ),
            "wall_contrast": 0.0 if self.wall_contrast is None else float(self.wall_contrast),
            "eye_contrast": 0.0 if self.eye_contrast is None else float(self.eye_contrast),
        }


class FrameGeometry:
    """Pad a frame to a square without stretching it.

    ``prepare_image`` then resizes that square to the checkpoint size.
    A square input is returned unchanged, so the resize is the only resample.
    """

    @staticmethod
    def mode(config: dict) -> str:
        mode = str(_section(config, "data", "resize")).strip()
        if mode not in {"letterbox", "stretch"}:
            raise ValueError("config data.resize must be letterbox or stretch")
        return mode

    @staticmethod
    def pad_square(image: np.ndarray, pad: str) -> np.ndarray:
        """Return a square view of one channel. ``pad`` is ``median``."""
        if pad != "median":
            raise ValueError("config data.pad must be median")
        array = np.asarray(image, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError("letterbox expects one infrared channel")
        height, width = array.shape
        if height == width:
            return array
        side = max(height, width)
        finite = array[np.isfinite(array)]
        fill = float(np.median(finite)) if finite.size else 0.0
        canvas = np.full((side, side), fill, dtype=np.float32)
        top = (side - height) // 2
        left = (side - width) // 2
        canvas[top : top + height, left : left + width] = array
        return canvas


class CycloneImageGate:
    """Decide whether a frame is a plausible tropical-cyclone infrared image.

    The checks are a single greyscale channel, a cold core relative to the
    frame, and radial organization. A colour-enhanced RGB or RGBA frame is
    refused before those checks. Radii are fractions of the half-width.
    Model uncertainty is applied later by :meth:`uncertainty_reason`. A
    refusal is not a wind speed.
    """

    @staticmethod
    def uncertainty_reason(std_kt: float | None, settings: GateSettings) -> str | None:
        """Refuse when dropout spread exceeds the configured limit."""
        if std_kt is None:
            return None
        if float(std_kt) > settings.max_dropout_std_kt:
            return (
                f"model uncertainty {float(std_kt):.1f} kt exceeds the configured "
                f"limit of {settings.max_dropout_std_kt:.0f} kt"
            )
        return None

    @staticmethod
    def assess(image: np.ndarray, settings: GateSettings) -> GateReport:
        planes, channel_reason = CycloneImageGate._planes(image, settings)
        if channel_reason is not None:
            return GateReport(False, channel_reason, 0.0, None)
        field = np.mean(np.stack(planes, axis=0), axis=0).astype(np.float64)
        finite = field[np.isfinite(field)]
        if finite.size == 0:
            return GateReport(False, "the frame has no finite pixels", 0.0, None)
        spread = float(finite.std())
        if spread < settings.min_std:
            return GateReport(
                False,
                f"the frame is too flat to be an infrared cyclone (std {spread:.4f})",
                0.0,
                field,
            )
        candidates = CycloneImageGate._candidates(field, settings)
        passing = [item for item in candidates if item[0]]
        chosen = max(passing or candidates, key=lambda item: item[1])
        accepted, score, reason, metrics, polarity = chosen
        return GateReport(
            accepted=accepted,
            reason=reason,
            score=score if accepted else 0.0,
            field=field,
            symmetry=metrics["symmetry"],
            cold_concentration=metrics["cold_concentration"],
            wall_contrast=metrics["wall_contrast"],
            eye_contrast=metrics["eye_contrast"],
            polarity=polarity,
        )

    @staticmethod
    def _planes(image: np.ndarray, settings: GateSettings) -> tuple[list[np.ndarray], str | None]:
        array = np.asarray(image)
        if array.ndim == 2:
            return [np.asarray(array, dtype=np.float64)], None
        if array.ndim != 3:
            return [], f"the array shape {array.shape} is not an infrared frame"
        if array.shape[0] <= 4 and array.shape[1] > 4 and array.shape[2] > 4:
            planes = [np.asarray(array[index], dtype=np.float64) for index in range(array.shape[0])]
        elif array.shape[-1] <= 4 and array.shape[0] > 4 and array.shape[1] > 4:
            planes = [np.asarray(array[..., index], dtype=np.float64) for index in range(array.shape[-1])]
        else:
            return [], f"the array shape {array.shape} is not an infrared frame"
        if len(planes) == 1:
            return planes, None
        if len(planes) not in (3, 4):
            return [], "the frame is not a single infrared channel"
        # Four planes are RGB plus alpha. Alpha is not a colour channel.
        colour = planes[:3]
        reason = CycloneImageGate._colour_reason(colour, settings)
        if reason is not None:
            return [], reason
        return colour, None

    @staticmethod
    def _colour_reason(planes: list[np.ndarray], settings: GateSettings) -> str | None:
        """Refuse a false-colour enhancement. Near-copies stay greyscale IR."""
        mean = np.mean(np.stack(planes, axis=0), axis=0)
        finite = mean[np.isfinite(mean)]
        if finite.size == 0:
            return "the frame has no finite pixels"
        low, high = np.percentile(finite, [5, 95])
        scale = max(float(high - low), 1e-6)
        pair_gaps = [
            float(np.mean(np.abs(planes[i] - planes[j])))
            for i in range(3)
            for j in range(i + 1, 3)
        ]
        ratio = float(np.mean(pair_gaps)) / scale
        if ratio > settings.grayscale_tolerance:
            return (
                "the frame is colour-enhanced rather than greyscale infrared "
                f"(relative channel spread {ratio:.2f})"
            )
        return None

    @staticmethod
    def _candidates(field: np.ndarray, settings: GateSettings) -> list[tuple]:
        if settings.cold_value == "low":
            signed = [(field, "low")]
        elif settings.cold_value == "high":
            signed = [(-field, "high")]
        else:
            signed = [(field, "low"), (-field, "high")]
        reports = []
        for values, polarity in signed:
            reports.append(CycloneImageGate._score_field(values, polarity, settings))
        return reports

    @staticmethod
    def _score_field(field: np.ndarray, polarity: str, settings: GateSettings) -> tuple:
        finite = np.isfinite(field)
        values = field[finite]
        low, high = np.percentile(values, [5, 95])
        scale = max(float(high - low), 1e-6)
        scaled = (field - float(np.median(values))) / scale
        height, width = scaled.shape
        yy, xx = np.mgrid[0:height, 0:width]
        half = min(height, width) / 2.0
        radius = np.hypot(yy - (height - 1) / 2.0, xx - (width - 1) / 2.0) / max(half, 1e-6)
        symmetry = CycloneImageGate._symmetry(scaled, radius, finite, settings.radial_bins)
        tail = float(np.percentile(scaled[finite], settings.cold_tail_fraction * 100.0))
        cold = finite & (scaled <= tail)
        core = radius <= settings.cold_radius_fraction
        concentration = float(cold[core].sum() / max(int(cold.sum()), 1))
        wall_mask = (radius > settings.eye_radius_fraction) & (
            radius <= settings.eyewall_outer_fraction
        )
        outer_mask = (radius > settings.outer_inner_fraction) & (
            radius <= settings.outer_outer_fraction
        )
        eye_mask = radius <= settings.eye_radius_fraction
        wall_mean = CycloneImageGate._masked_mean(scaled, wall_mask & finite)
        outer_mean = CycloneImageGate._masked_mean(scaled, outer_mask & finite)
        eye_mean = CycloneImageGate._masked_mean(scaled, eye_mask & finite)
        wall_contrast = (
            None if wall_mean is None or outer_mean is None else outer_mean - wall_mean
        )
        eye_contrast = None if eye_mean is None or wall_mean is None else eye_mean - wall_mean
        metrics = {
            "symmetry": symmetry,
            "cold_concentration": concentration,
            "wall_contrast": wall_contrast,
            "eye_contrast": eye_contrast,
        }
        failures: list[str] = []
        if symmetry is None or symmetry < settings.min_symmetry:
            shown = "missing" if symmetry is None else f"{symmetry:.2f}"
            failures.append(f"radial symmetry {shown} is below {settings.min_symmetry:.2f}")
        if concentration < settings.min_cold_concentration:
            failures.append(
                "cold-cloud concentration "
                f"{concentration:.2f} is below {settings.min_cold_concentration:.2f}"
            )
        if wall_contrast is None or wall_contrast < settings.min_wall_contrast:
            shown = "missing" if wall_contrast is None else f"{wall_contrast:.2f}"
            failures.append(f"cold-core contrast {shown} is below {settings.min_wall_contrast:.2f}")
        score = CycloneImageGate._score(symmetry, concentration, wall_contrast, settings)
        if failures:
            reason = "; ".join(failures) + f" ({polarity} values treated as cold)"
            return False, score, reason, metrics, polarity
        reason = (
            "cold cloud is radially organized "
            f"({polarity} values treated as cold)"
        )
        return True, score, reason, metrics, polarity

    @staticmethod
    def _symmetry(scaled, radius, finite, radial_bins: int) -> float | None:
        inside = finite & (radius <= 1)
        if int(inside.sum()) < 8:
            return None
        edges = np.linspace(0.0, 1.0, int(radial_bins) + 1)
        residual = np.zeros(scaled.shape, dtype=np.float64)
        for index in range(int(radial_bins)):
            mask = inside & (radius >= edges[index]) & (radius < edges[index + 1])
            if int(mask.sum()) == 0:
                continue
            residual[mask] = scaled[mask] - float(scaled[mask].mean())
        variance = float(scaled[inside].var())
        if variance <= 1e-8:
            return 1.0
        return float(1.0 - residual[inside].var() / variance)

    @staticmethod
    def _masked_mean(scaled, mask) -> float | None:
        if int(mask.sum()) == 0:
            return None
        return float(scaled[mask].mean())

    @staticmethod
    def _score(symmetry, concentration, wall_contrast, settings: GateSettings) -> float:
        if symmetry is None or wall_contrast is None:
            return 0.0
        return float(
            np.clip(
                min(
                    1.0,
                    max(float(symmetry), 0.0),
                    max(float(wall_contrast), 0.0),
                    max(float(concentration), 0.0) / settings.concentration_reference,
                ),
                0.0,
                1.0,
            )
        )


class ConfidenceScore:
    """Gate score, dropout spread, and residual quantiles when a checkpoint has them.

    The label stays uncalibrated unless a matching real-data run record
    carries a coverage check. This is not a prediction interval.
    """

    @staticmethod
    def interval_width(raw) -> float | None:
        if not isinstance(raw, dict) or "low_kt" not in raw or "high_kt" not in raw:
            return None
        low = _finite(raw["low_kt"], "residual_quantiles.low_kt")
        high = _finite(raw["high_kt"], "residual_quantiles.high_kt")
        if high < low:
            return None
        return high - low

    @staticmethod
    def combine(
        config: dict,
        gate_score: float,
        dropout_std: float | None,
        residual_quantiles,
    ) -> float:
        cfg = _section(config, "predict", "confidence")
        weights = {
            "gate": _finite(cfg["gate_weight"], "predict.confidence.gate_weight"),
            "mc": _finite(cfg["mc_weight"], "predict.confidence.mc_weight"),
            "interval": _finite(cfg["interval_weight"], "predict.confidence.interval_weight"),
        }
        if min(weights.values()) <= 0:
            raise ValueError("config predict.confidence weights must be positive")
        terms = [(weights["gate"], float(np.clip(gate_score, 0.0, 1.0)))]
        if dropout_std is not None:
            scale = _finite(cfg["mc_std_scale_kt"], "predict.confidence.mc_std_scale_kt")
            if scale <= 0:
                raise ValueError("config predict.confidence.mc_std_scale_kt must be positive")
            terms.append((weights["mc"], 1.0 / (1.0 + float(dropout_std) / scale)))
        width = ConfidenceScore.interval_width(residual_quantiles)
        if width is not None:
            scale = _finite(cfg["interval_scale_kt"], "predict.confidence.interval_scale_kt")
            if scale <= 0:
                raise ValueError("config predict.confidence.interval_scale_kt must be positive")
            interval = float(np.clip(1.0 - width / scale, 0.0, 1.0))
            terms.append((weights["interval"], interval))
        total = sum(weight for weight, _value in terms)
        score = sum(weight * value for weight, value in terms) / total
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def status(config: dict, record: dict | None) -> str:
        cfg = _section(config, "predict", "confidence")
        uncalibrated = str(cfg["uncalibrated_label"]).strip()
        recorded = str(cfg["coverage_label"]).strip()
        if not isinstance(record, dict):
            return uncalibrated
        check = record.get("coverage_check")
        if not isinstance(check, dict) or check.get("real_data") is not True:
            return uncalibrated
        try:
            count = int(check["n"])
            nominal = float(check["nominal"])
            observed = float(check["observed"])
        except (KeyError, TypeError, ValueError):
            return uncalibrated
        if count < 1 or not math.isfinite(nominal) or not math.isfinite(observed):
            return uncalibrated
        return recorded


class IntensityGrouping:
    """Project bins. The names are not an official IMD class."""

    @staticmethod
    def note(config: dict) -> str:
        return str(_section(config, "data", "grouping_note")).strip()

    @staticmethod
    def code(wind_kt: float, edges: tuple[float, ...], names: tuple[str, ...]) -> str | None:
        return wind_to_imd5(float(wind_kt), edges, names)


class RunRecord:
    """A checkpoint may be shown only when a real-data record matches it.

    A synthetic checkpoint is always flagged. A checkpoint that claims to
    be real and has no matching record produces no wind speed.
    """

    @staticmethod
    def load_match(config: dict, payload: dict) -> tuple[bool, dict | None]:
        run_id = payload.get("run_id")
        if run_id in (None, ""):
            return False, None
        run_id = str(run_id)
        if "/" in run_id or "\\" in run_id or ".." in run_id:
            return False, None
        path = Path(_section(config, "paths", "run_records")) / f"{run_id}.json"
        if not path.is_file():
            return False, None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeError):
            return False, None
        if not isinstance(record, dict):
            return False, None
        if str(record.get("run_id")) != run_id:
            return False, None
        if bool(record.get("is_synthetic", True)):
            return False, None
        if str(record.get("data_kind")) != "real":
            return False, None
        try:
            same_model = str(record.get("model_name")) == str(payload.get("model_name"))
            same_size = int(record.get("image_size")) == int(payload.get("image_size"))
        except (TypeError, ValueError):
            return False, None
        if not same_model or not same_size:
            return False, None
        return True, record


class CoastalContext:
    """Nearest configured coastal reference points. Not a landfall forecast."""

    @staticmethod
    def describe(config: dict, latitude, longitude, metadata) -> tuple:
        position = CoastalContext.position(config, latitude, longitude, metadata)
        if position is None:
            return None, None, None
        lat, lon = position
        count = int(_section(config, "predict", "coastal_count"))
        if count < 1:
            raise ValueError("config predict.coastal_count must be at least 1")
        radius = _finite(_section(config, "predict", "earth_radius_km"), "predict.earth_radius_km")
        places = _section(config, "predict", "coast")
        if not isinstance(places, list) or not places:
            raise ValueError("config predict.coast must list reference points")
        ranked = sorted(
            (
                (
                    CoastalContext.distance_km(lat, lon, float(place["lat"]), float(place["lon"]), radius),
                    str(place["name"]),
                )
                for place in places
            ),
            key=lambda item: item[0],
        )
        names = tuple(name for _distance, name in ranked[:count])
        note = str(_section(config, "predict", "landfall_note")).strip()
        reference = str(_section(config, "predict", "coast_reference_note")).strip()
        return names, note, reference

    @staticmethod
    def position(config: dict, latitude, longitude, metadata) -> tuple[float, float] | None:
        meta = metadata or {}
        lat = latitude
        lon = longitude
        if lat is None:
            for key in _section(config, "predict", "latitude_keys"):
                if key in meta and meta[key] is not None:
                    lat = meta[key]
                    break
        if lon is None:
            for key in _section(config, "predict", "longitude_keys"):
                if key in meta and meta[key] is not None:
                    lon = meta[key]
                    break
        if lat is None and lon is None:
            return None
        if lat is None or lon is None:
            raise ValueError("latitude and longitude must be supplied together")
        lat_f = float(lat)
        lon_f = float(lon)
        if not (-90.0 <= lat_f <= 90.0 and -180.0 <= lon_f <= 180.0):
            raise ValueError("latitude or longitude is outside the geographic range")
        return lat_f, lon_f

    @staticmethod
    def distance_km(lat1: float, lon1: float, lat2: float, lon2: float, radius_km: float) -> float:
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        d_phi = math.radians(lat2 - lat1)
        d_lon = math.radians(lon2 - lon1)
        haversine = (
            math.sin(d_phi / 2.0) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(d_lon / 2.0) ** 2
        )
        return 2.0 * radius_km * math.asin(min(1.0, math.sqrt(haversine)))


class CheckpointStore:
    """Load a ``.pt`` state dict or a ``.joblib`` estimator. No other suffix."""

    @staticmethod
    def load(path: Path) -> dict:
        if path.suffix == ".joblib":
            import joblib

            payload = joblib.load(path)
            if not isinstance(payload, dict) or "estimator" not in payload:
                raise ValueError("joblib checkpoint must be a dict with an estimator")
            loaded = dict(payload)
            loaded["_kind"] = "sklearn"
            return loaded
        if path.suffix == ".pt":
            import torch

            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                payload = torch.load(path, map_location="cpu")
            if not isinstance(payload, dict):
                raise ValueError("torch checkpoint must be a dict")
            loaded = dict(payload)
            loaded["_kind"] = "torch"
            return loaded
        raise ValueError("checkpoint must be a .pt or .joblib file")


@dataclass(frozen=True)
class Estimate:
    """What ``predict_images`` turns into a :class:`cloudie.predict.Prediction`."""

    outcome: str
    banner: str
    is_synthetic: bool
    wind_kt: float | None = None
    wind_kmh: float | None = None
    dropout_std_kt: float | None = None
    samples: tuple[float, ...] = ()
    confidence: float | None = None
    confidence_status: str | None = None
    grouping_note: str | None = None
    operational_disclaimer: str | None = None
    coastal_heading: str | None = None
    nearest_coastal_states: tuple[str, ...] | None = None
    landfall_note: str | None = None
    coast_reference_note: str | None = None
    gate_reason: str | None = None
    gate_score: float | None = None
    matched_run: bool = False
    outside_training: bool = False
    wind_averaging: str | None = None
    edges: tuple[float, ...] = ()
    names: tuple[str, ...] = ()
    honesty: str | None = None


class IntensityEstimate:
    """Run the gate, then the checkpoint, and refuse a wind when either fails.

    This is not an operational forecast. Dropout spread and residual
    quantiles are not a calibrated interval. Nearest coastal states are
    context for a supplied position, not a landfall forecast.
    """

    @staticmethod
    def predict(
        config: dict,
        path: Path,
        images: list[np.ndarray],
        *,
        hemisphere: str | None,
        latitude,
        longitude,
        metadata,
    ) -> Estimate:
        ctx = IntensityEstimate._context(config, latitude, longitude, metadata)
        payload = CheckpointStore.load(path)
        IntensityEstimate._check_identity(config, payload)
        synthetic = bool(payload.get("is_synthetic"))
        matched, record = RunRecord.load_match(config, payload)
        if synthetic:
            matched, record = False, None
        expected = int(payload.get("n_frames", 1))
        if len(images) != expected:
            banner = (
                f"This checkpoint expects {expected} infrared frames and received "
                f"{len(images)}. There is no model output. This is not a forecast."
            )
            return Estimate("frame_count", banner, synthetic, matched_run=matched, **ctx)
        frames = IntensityEstimate._unflipped(images, hemisphere)
        missing = IntensityEstimate._missing_fraction(frames, float(payload["max_missing_fraction"]))
        if missing is not None:
            return Estimate("missing_pixels", "", synthetic, matched_run=matched, **ctx)
        settings = GateSettings.from_config(config)
        reports = [CycloneImageGate.assess(frame, settings) for frame in frames]
        refused = next((report for report in reports if not report.accepted), None)
        if refused is not None:
            return Estimate(
                "not_cyclone",
                IntensityEstimate._gate_banner(settings, refused.reason),
                synthetic,
                gate_reason=refused.reason,
                gate_score=refused.score,
                matched_run=matched,
                **ctx,
            )
        if not synthetic and not matched:
            banner = str(_section(config, "predict", "no_run_record_refusal")).strip()
            return Estimate("no_run_record", banner, synthetic, matched_run=False, **ctx)
        prepared = IntensityEstimate._prepare(config, payload, reports)
        wind_kt, samples, std_kt = IntensityEstimate._wind(config, payload, prepared, reports)
        uncertain = CycloneImageGate.uncertainty_reason(std_kt, settings)
        if uncertain is not None:
            return Estimate(
                "not_cyclone",
                IntensityEstimate._gate_banner(settings, uncertain),
                synthetic,
                gate_reason=uncertain,
                matched_run=matched,
                **ctx,
            )
        edges = tuple(float(edge) for edge in payload.get("edges_kt", config["data"]["imd5_edges_kt"]))
        names = tuple(str(name) for name in payload.get("class_names", config["data"]["imd5_names"]))
        averaging = payload.get("wind_averaging")
        averaging = None if averaging in (None, "") else str(averaging)
        field_mean = IntensityEstimate._mean(reports)
        width = _finite(_section(config, "predict", "training_mean_width"), "predict.training_mean_width")
        scale = max(float(payload["norm_std"]), 1e-6)
        outside = abs(field_mean - float(payload["norm_mean"])) > width * scale
        low, high = (float(bound) for bound in config["data"]["plausible_wind_kt"])
        if outside or not (low <= wind_kt <= high):
            return Estimate(
                "outside_range",
                "",
                synthetic,
                outside_training=outside,
                wind_averaging=averaging,
                edges=edges,
                names=names,
                honesty=payload.get("honesty"),
                matched_run=matched,
                **ctx,
            )
        gate_score = min(report.score for report in reports)
        confidence = ConfidenceScore.combine(
            config, gate_score, std_kt, payload.get("residual_quantiles")
        )
        factor = _finite(_section(config, "data", "kt_to_kmh"), "data.kt_to_kmh")
        if factor <= 0:
            raise ValueError("config data.kt_to_kmh must be positive")
        return Estimate(
            "ok",
            "",
            synthetic,
            wind_kt=float(wind_kt),
            wind_kmh=float(wind_kt) * factor,
            dropout_std_kt=None if std_kt is None else float(std_kt),
            samples=tuple(float(value) for value in samples),
            confidence=confidence,
            confidence_status=ConfidenceScore.status(config, record if matched else None),
            grouping_note=IntensityGrouping.note(config),
            gate_score=gate_score,
            matched_run=matched,
            wind_averaging=averaging,
            edges=edges,
            names=names,
            honesty=None if payload.get("honesty") is None else str(payload.get("honesty")),
            **ctx,
        )

    @staticmethod
    def _context(config: dict, latitude, longitude, metadata) -> dict:
        states, note, reference = CoastalContext.describe(config, latitude, longitude, metadata)
        return {
            "operational_disclaimer": str(_section(config, "predict", "operational_disclaimer")).strip(),
            "coastal_heading": str(_section(config, "predict", "coastal_heading")).strip(),
            "nearest_coastal_states": states,
            "landfall_note": note,
            "coast_reference_note": reference,
        }

    @staticmethod
    def _check_identity(config: dict, payload: dict) -> None:
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

    @staticmethod
    def _unflipped(images: list[np.ndarray], hemisphere: str | None) -> list[np.ndarray]:
        from cloudie.preprocessing import leave_unflipped

        return [leave_unflipped(image, hemisphere) for image in images]

    @staticmethod
    def _missing_fraction(images: list[np.ndarray], limit: float) -> str | None:
        from cloudie.preprocessing import as_channel_first, missing_fraction

        for image in images:
            if missing_fraction(as_channel_first(image)) > limit:
                return "missing"
        return None

    @staticmethod
    def _gate_banner(settings: GateSettings, reason: str) -> str:
        return (
            f"{settings.refusal}: {reason}. "
            "There is no wind speed. This is not a forecast."
        )

    @staticmethod
    def _prepare(config: dict, payload: dict, reports: list[GateReport]) -> list[np.ndarray]:
        from cloudie.preprocessing import NormStats, prepare_image

        mode = FrameGeometry.mode(config)
        pad = str(_section(config, "data", "pad")).strip()
        stats = NormStats(
            mean=float(payload["norm_mean"]),
            std=float(payload["norm_std"]),
            source_split="train",
        )
        image_size = int(payload["image_size"])
        limit = float(payload["max_missing_fraction"])
        prepared = []
        for report in reports:
            field = report.field if mode == "stretch" else FrameGeometry.pad_square(report.field, pad)
            prepared.append(prepare_image(field, image_size, stats, limit))
        return prepared

    @staticmethod
    def _mean(reports: list[GateReport]) -> float:
        values = [np.asarray(report.field, dtype=np.float64) for report in reports]
        finite = [image[np.isfinite(image)] for image in values]
        finite = [image for image in finite if image.size]
        if not finite:
            raise ValueError("image has no finite pixels")
        return float(np.concatenate(finite).mean())

    @staticmethod
    def _wind(config: dict, payload: dict, prepared: list[np.ndarray], reports: list[GateReport]):
        if payload["_kind"] == "sklearn":
            wind = IntensityEstimate._sklearn_wind(config, payload, reports[0])
            return wind, np.asarray([wind], dtype=np.float64), None
        return IntensityEstimate._torch_wind(config, payload, prepared)

    @staticmethod
    def _sklearn_wind(config: dict, payload: dict, report: GateReport) -> float:
        expected = [str(name) for name in _section(config, "predict", "sklearn_features")]
        names = [str(name) for name in payload["feature_names"]]
        if names != expected:
            raise ValueError("joblib feature_names do not match predict.sklearn_features")
        estimator = payload["estimator"]
        if not hasattr(estimator, "predict"):
            raise ValueError("joblib checkpoint estimator has no predict")
        features = report.feature_map()
        vector = np.asarray([[features[name] for name in names]], dtype=np.float64)
        wind = float(np.asarray(estimator.predict(vector), dtype=np.float64).reshape(-1)[0])
        return wind

    @staticmethod
    def _torch_wind(config: dict, payload: dict, prepared: list[np.ndarray]):
        import torch

        count = int(_section(config, "model", "mc_samples"))
        if count < 1:
            raise ValueError("model.mc_samples must be at least 1")
        stack = np.stack(prepared, axis=0)
        tensor = torch.from_numpy(np.ascontiguousarray(stack))
        model_input = tensor if len(prepared) == 1 else tensor.unsqueeze(0)
        model = ModelAccess.build_net(config, size_head=bool(payload.get("size_head", False)))
        model.load_state_dict(payload["state_dict"])
        seed = int(config.get("project", {}).get("seed", 0))
        dropout = ModelAccess.mc_dropout()
        state = torch.get_rng_state()
        try:
            model.eval()
            with torch.no_grad():
                point = IntensityEstimate._as_wind(model(model_input))
                if count == 1:
                    return point, np.asarray([point], dtype=np.float64), 0.0
                dropout.enable(model)
                torch.manual_seed(seed)
                draws = [
                    IntensityEstimate._as_wind(model(model_input)) for _ in range(count)
                ]
            samples = np.asarray(draws, dtype=np.float64)
            return point, samples, float(samples.std(ddof=1))
        finally:
            model.eval()
            torch.set_rng_state(state)

    @staticmethod
    def _as_wind(output) -> float:
        wind = output[0] if isinstance(output, tuple) else output
        return float(wind.detach().cpu().reshape(-1)[0])
