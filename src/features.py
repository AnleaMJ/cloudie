"""Physically motivated image features for classical intensity baselines.

Distributional statistics are called ``tb_*`` only when the caller has set
``features.calibrated``. Otherwise they are ``pixel_*``. Nothing here converts
digital numbers to brightness temperature.

The simplified DAV terms are the circular variance of the angle between the
image gradient and the radial direction, at fixed fractions of the crop
half-width. That is not operational deviation-angle variance: there is no
radius search, no flight-level wind, and no agency radius product.

Radial, eye, and DAV features assume the storm is already near the image
center. This module does not recenter with a best-track fix.

Temporal deltas use the previous frame of the same storm after cross-split
links have been removed. Wind, R34, RMW, and class labels are not features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer

from src.data_loader import load_image, resolve_image
from src.preprocessing import MAX_MISSING_FRACTION, as_channel_first, missing_fraction
from src.schema import FORBIDDEN_FEATURES, parse_is_synthetic, wind_to_imd5
from src.splits import LeakageError, add_previous_frame_id, null_cross_split_links

GLCM_NAMES = (
    "glcm_contrast",
    "glcm_dissimilarity",
    "glcm_homogeneity",
    "glcm_energy",
    "glcm_correlation",
    "glcm_asm",
)
HAAR_NAMES = (
    "haar_energy_ll",
    "haar_energy_lh",
    "haar_energy_hl",
    "haar_energy_hh",
)
EYE_NAMES = (
    "eye_center_mean",
    "cdo_annulus_mean",
    "eye_cdo_contrast",
    "eye_present",
)


@dataclass(frozen=True)
class FeatureSpec:
    """Geometry of one feature vector. Thresholds are in image units."""

    calibrated: bool = False
    cold_thresholds: tuple[float, ...] = (200.0, 210.0, 220.0, 230.0, 240.0)
    radial_bins: int = 8
    eye_radius_fraction: float = 0.12
    cdo_outer_fraction: float = 0.35
    eye_contrast_min: float = 2.0
    dav_radii_fractions: tuple[float, ...] = (0.15, 0.30, 0.45, 0.60)
    dav_annulus_width_fraction: float = 0.08
    glcm_levels: int = 16
    glcm_distance: int = 1

    def __post_init__(self) -> None:
        if not self.cold_thresholds:
            raise ValueError("cold_thresholds must not be empty")
        if self.radial_bins < 2:
            raise ValueError("radial_bins must be at least 2")
        if not 0 < self.eye_radius_fraction < self.cdo_outer_fraction <= 1:
            raise ValueError("eye radius must sit inside the CDO annulus")
        if self.eye_contrast_min < 0:
            raise ValueError("eye_contrast_min must be >= 0")
        if any(not 0 < radius < 1 for radius in self.dav_radii_fractions):
            raise ValueError("DAV radii must be fractions of the half-width")
        if not 0 < self.dav_annulus_width_fraction < 0.5:
            raise ValueError("DAV annulus width fraction is out of range")
        if not 8 <= self.glcm_levels <= 32:
            raise ValueError("glcm_levels must be between 8 and 32")
        if self.glcm_distance < 1:
            raise ValueError("glcm_distance must be >= 1")

    @property
    def prefix(self) -> str:
        return "tb" if self.calibrated else "pixel"

    @classmethod
    def from_config(cls, config: dict) -> FeatureSpec:
        raw = dict(config.get("features") or {})
        raw.pop("selection", None)
        if "cold_thresholds" in raw:
            raw["cold_thresholds"] = tuple(float(value) for value in raw["cold_thresholds"])
        if "dav_radii_fractions" in raw:
            raw["dav_radii_fractions"] = tuple(float(value) for value in raw["dav_radii_fractions"])
        return cls(
            calibrated=bool(raw.get("calibrated", False)),
            cold_thresholds=raw.get("cold_thresholds", cls.cold_thresholds),
            radial_bins=int(raw.get("radial_bins", 8)),
            eye_radius_fraction=float(raw.get("eye_radius_fraction", 0.12)),
            cdo_outer_fraction=float(raw.get("cdo_outer_fraction", 0.35)),
            eye_contrast_min=float(raw.get("eye_contrast_min", 2.0)),
            dav_radii_fractions=raw.get("dav_radii_fractions", cls.dav_radii_fractions),
            dav_annulus_width_fraction=float(raw.get("dav_annulus_width_fraction", 0.08)),
            glcm_levels=int(raw.get("glcm_levels", 16)),
            glcm_distance=int(raw.get("glcm_distance", 1)),
        )

    def image_feature_names(self) -> tuple[str, ...]:
        prefix = self.prefix
        names = [
            f"{prefix}_min",
            f"{prefix}_max",
            f"{prefix}_mean",
            f"{prefix}_std",
            f"{prefix}_p10",
            f"{prefix}_p25",
            f"{prefix}_p50",
            f"{prefix}_p75",
            f"{prefix}_p90",
            f"{prefix}_iqr",
        ]
        names.extend(f"cold_frac_{_token(threshold)}" for threshold in self.cold_thresholds)
        names.extend(f"radial_mean_{index:02d}" for index in range(self.radial_bins))
        names.extend(EYE_NAMES)
        names.extend(
            f"dav_simplified_r{index:02d}" for index in range(len(self.dav_radii_fractions))
        )
        names.extend(GLCM_NAMES)
        names.extend(HAAR_NAMES)
        return tuple(names)

    def feature_names(self) -> tuple[str, ...]:
        prefix = self.prefix
        temporal = (
            "has_prev",
            f"delta_{prefix}_mean",
            f"delta_{prefix}_min",
            "delta_cold_primary",
            "delta_eye_cdo_contrast",
            "delta_radial_mean_00",
        )
        return self.image_feature_names() + temporal


@dataclass(frozen=True)
class FeatureSelection:
    names: tuple[str, ...]
    source_split: str
    importances: dict[str, float]
    dropped_variance: tuple[str, ...]
    dropped_correlation: tuple[str, ...]


def _token(value: float) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return str(number).replace("-", "m").replace(".", "p")


def _single_channel(image: np.ndarray) -> np.ndarray:
    return as_channel_first(image)[0]


def _finite_fill(image: np.ndarray) -> np.ndarray:
    filled = np.array(image, dtype=np.float64, copy=True)
    finite = np.isfinite(filled)
    fill = float(np.median(filled[finite])) if finite.any() else 0.0
    filled[~finite] = fill
    return filled


def _distribution(image: np.ndarray, prefix: str) -> dict[str, float]:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        keys = (
            "min",
            "max",
            "mean",
            "std",
            "p10",
            "p25",
            "p50",
            "p75",
            "p90",
            "iqr",
        )
        return {f"{prefix}_{key}": float("nan") for key in keys}
    percentiles = np.percentile(finite, [10, 25, 50, 75, 90])
    return {
        f"{prefix}_min": float(finite.min()),
        f"{prefix}_max": float(finite.max()),
        f"{prefix}_mean": float(finite.mean()),
        f"{prefix}_std": float(finite.std()),
        f"{prefix}_p10": float(percentiles[0]),
        f"{prefix}_p25": float(percentiles[1]),
        f"{prefix}_p50": float(percentiles[2]),
        f"{prefix}_p75": float(percentiles[3]),
        f"{prefix}_p90": float(percentiles[4]),
        f"{prefix}_iqr": float(percentiles[3] - percentiles[1]),
    }


def _cold_coverage(image: np.ndarray, thresholds: tuple[float, ...]) -> dict[str, float]:
    finite = image[np.isfinite(image)]
    out: dict[str, float] = {}
    for threshold in thresholds:
        key = f"cold_frac_{_token(threshold)}"
        if finite.size == 0:
            out[key] = float("nan")
        else:
            out[key] = float(np.mean(finite < threshold))
    return out


def _radial_and_eye(image: np.ndarray, spec: FeatureSpec) -> dict[str, float]:
    height, width = image.shape
    yy, xx = np.mgrid[0:height, 0:width]
    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0
    radius = np.hypot(yy - center_y, xx - center_x)
    max_radius = max(float(radius.max()), 1e-6)
    edges = np.linspace(0.0, max_radius + 1e-6, spec.radial_bins + 1)
    out: dict[str, float] = {}
    for index in range(spec.radial_bins):
        mask = (radius >= edges[index]) & (radius < edges[index + 1]) & np.isfinite(image)
        out[f"radial_mean_{index:02d}"] = (
            float(image[mask].mean()) if mask.any() else float("nan")
        )
    half = min(height, width) / 2.0
    eye = radius <= spec.eye_radius_fraction * half
    annulus = (radius > spec.eye_radius_fraction * half) & (
        radius <= spec.cdo_outer_fraction * half
    )
    eye_values = image[eye & np.isfinite(image)]
    annulus_values = image[annulus & np.isfinite(image)]
    eye_mean = float(eye_values.mean()) if eye_values.size else float("nan")
    annulus_mean = float(annulus_values.mean()) if annulus_values.size else float("nan")
    contrast = eye_mean - annulus_mean
    eye_present = (
        0.0
        if not np.isfinite(contrast)
        else float(contrast >= spec.eye_contrast_min)
    )
    out["eye_center_mean"] = eye_mean
    out["cdo_annulus_mean"] = annulus_mean
    out["eye_cdo_contrast"] = float(contrast)
    out["eye_present"] = eye_present
    return out


def _deviation_angle_variance(image: np.ndarray, spec: FeatureSpec) -> dict[str, float]:
    """Circular variance of gradient angle minus radial angle. Not operational DAV."""
    height, width = image.shape
    yy, xx = np.mgrid[0:height, 0:width]
    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0
    dy = yy - center_y
    dx = xx - center_x
    radius = np.hypot(dy, dx)
    half = min(height, width) / 2.0
    gradient_y, gradient_x = np.gradient(image)
    magnitude = np.hypot(gradient_y, gradient_x)
    radial_angle = np.arctan2(dy, dx)
    gradient_angle = np.arctan2(gradient_y, gradient_x)
    deviation = np.arctan2(
        np.sin(gradient_angle - radial_angle),
        np.cos(gradient_angle - radial_angle),
    )
    # Keep the ring at least two pixels thick. On a small crop the fractional
    # width can otherwise fall between grid points and the statistic is empty.
    width_px = max(spec.dav_annulus_width_fraction * half, 2.0)
    out: dict[str, float] = {}
    for index, fraction in enumerate(spec.dav_radii_fractions):
        radius_px = fraction * half
        annulus = (radius >= radius_px - width_px / 2.0) & (
            radius <= radius_px + width_px / 2.0
        )
        # A zero gradient has no direction. arctan2(0, 0) is 0 and would
        # look like a full circle of disagreement.
        selected = deviation[annulus & (magnitude > 1e-6)]
        if selected.size < 8:
            out[f"dav_simplified_r{index:02d}"] = float("nan")
            continue
        mean_vector = np.exp(1j * selected).mean()
        out[f"dav_simplified_r{index:02d}"] = float(1.0 - np.abs(mean_vector))
    return out


def _pair_counts(
    bins: np.ndarray, valid: np.ndarray, dy: int, dx: int, levels: int
) -> np.ndarray | None:
    height, width = bins.shape
    if abs(dy) >= height or abs(dx) >= width:
        return None
    if dy >= 0:
        src_y, dst_y = slice(0, height - dy), slice(dy, height)
    else:
        src_y, dst_y = slice(-dy, height), slice(0, height + dy)
    if dx >= 0:
        src_x, dst_x = slice(0, width - dx), slice(dx, width)
    else:
        src_x, dst_x = slice(-dx, width), slice(0, width + dx)
    mask = valid[src_y, src_x] & valid[dst_y, dst_x]
    if not mask.any():
        return None
    source = bins[src_y, src_x][mask].ravel()
    dest = bins[dst_y, dst_x][mask].ravel()
    flat = source.astype(np.int64) * levels + dest.astype(np.int64)
    counts = np.bincount(flat, minlength=levels * levels).reshape(levels, levels)
    total = float(counts.sum())
    if total <= 0:
        return None
    return counts.astype(np.float64) / total


def _haralick(probs: np.ndarray) -> dict[str, float]:
    levels = probs.shape[0]
    rows, cols = np.indices((levels, levels))
    contrast = float(((rows - cols) ** 2 * probs).sum())
    dissimilarity = float((np.abs(rows - cols) * probs).sum())
    homogeneity = float((probs / (1.0 + (rows - cols) ** 2)).sum())
    asm = float((probs**2).sum())
    energy = float(np.sqrt(asm))
    mean_i = float((rows * probs).sum())
    mean_j = float((cols * probs).sum())
    std_i = float(np.sqrt(((rows - mean_i) ** 2 * probs).sum()))
    std_j = float(np.sqrt(((cols - mean_j) ** 2 * probs).sum()))
    if std_i < 1e-12 or std_j < 1e-12:
        correlation = 0.0
    else:
        correlation = float(
            ((rows - mean_i) * (cols - mean_j) * probs).sum() / (std_i * std_j)
        )
    return {
        "glcm_contrast": contrast,
        "glcm_dissimilarity": dissimilarity,
        "glcm_homogeneity": homogeneity,
        "glcm_energy": energy,
        "glcm_correlation": correlation,
        "glcm_asm": asm,
    }


def _glcm_features(image: np.ndarray, levels: int, distance: int) -> dict[str, float]:
    height, width = image.shape
    empty = {name: float("nan") for name in GLCM_NAMES}
    if height <= distance or width <= distance:
        return empty
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return empty
    low = float(finite.min())
    high = float(finite.max())
    if high - low < 1e-6:
        bins = np.zeros(image.shape, dtype=np.int32)
    else:
        scaled = np.clip((image - low) / (high - low), 0.0, 1.0)
        bins = np.minimum((scaled * levels).astype(np.int32), levels - 1)
    valid = np.isfinite(image)
    offsets = (
        (0, distance),
        (-distance, distance),
        (-distance, 0),
        (-distance, -distance),
    )
    collected = []
    for dy, dx in offsets:
        probs = _pair_counts(bins, valid, dy, dx, levels)
        if probs is not None:
            collected.append(_haralick(probs))
    if not collected:
        return empty
    return {
        name: float(np.mean([item[name] for item in collected])) for name in GLCM_NAMES
    }


def _haar_energies(image: np.ndarray) -> dict[str, float]:
    band = image
    if band.shape[0] % 2:
        band = band[:-1, :]
    if band.shape[1] % 2:
        band = band[:, :-1]
    if min(band.shape) < 2:
        return {name: float("nan") for name in HAAR_NAMES}
    even_even = band[0::2, 0::2]
    even_odd = band[0::2, 1::2]
    odd_even = band[1::2, 0::2]
    odd_odd = band[1::2, 1::2]
    low_low = (even_even + even_odd + odd_even + odd_odd) / 4.0
    low_high = (even_even + odd_even - even_odd - odd_odd) / 4.0
    high_low = (even_even - odd_even + even_odd - odd_odd) / 4.0
    high_high = (even_even - odd_even - even_odd + odd_odd) / 4.0
    energies = {
        "haar_energy_ll": float(np.mean(low_low**2)),
        "haar_energy_lh": float(np.mean(low_high**2)),
        "haar_energy_hl": float(np.mean(high_low**2)),
        "haar_energy_hh": float(np.mean(high_high**2)),
    }
    total = sum(energies.values())
    if total <= 1e-12:
        return {name: 0.0 for name in HAAR_NAMES}
    return {name: value / total for name, value in energies.items()}


def extract_image_features(
    image: np.ndarray,
    spec: FeatureSpec,
    max_missing_fraction: float = MAX_MISSING_FRACTION,
) -> dict[str, float]:
    """Features of one centered IR crop. The label is not an input."""
    channel = _single_channel(image)
    fraction = missing_fraction(channel)
    if fraction > max_missing_fraction:
        raise ValueError(
            f"missing fraction {fraction:.2f} exceeds {max_missing_fraction:.2f}"
        )
    if not np.isfinite(channel).any():
        raise ValueError("image has no finite pixels")
    dense = _finite_fill(channel)
    features: dict[str, float] = {}
    features.update(_distribution(channel, spec.prefix))
    features.update(_cold_coverage(channel, spec.cold_thresholds))
    features.update(_radial_and_eye(channel, spec))
    features.update(_deviation_angle_variance(dense, spec))
    features.update(_glcm_features(dense, spec.glcm_levels, spec.glcm_distance))
    features.update(_haar_energies(dense))
    expected = spec.image_feature_names()
    missing = [name for name in expected if name not in features]
    if missing:
        raise RuntimeError(f"feature extractor omitted {missing}")
    return {name: float(features[name]) for name in expected}


def _temporal_defaults(spec: FeatureSpec) -> dict[str, float]:
    prefix = spec.prefix
    return {
        "has_prev": 0.0,
        f"delta_{prefix}_mean": float("nan"),
        f"delta_{prefix}_min": float("nan"),
        "delta_cold_primary": float("nan"),
        "delta_eye_cdo_contrast": float("nan"),
        "delta_radial_mean_00": float("nan"),
    }


def extract_feature_table(
    frame: pd.DataFrame,
    manifest_path: str,
    spec: FeatureSpec,
    max_missing_fraction: float = MAX_MISSING_FRACTION,
) -> tuple[pd.DataFrame, int]:
    """Extract features and same-split temporal deltas.

    Previous-frame ids are recomputed from ``storm_id`` and ``timestamp``,
    then any link that crosses ``split`` is dropped before the delta is taken.
    """
    if "split" not in frame.columns:
        raise LeakageError("feature extraction requires a split column")
    linked = add_previous_frame_id(frame)
    linked, dropped = null_cross_split_links(linked)
    cache: dict[str, tuple[str, dict[str, float]]] = {}
    packed: list[tuple[dict[str, object], dict[str, float]]] = []
    for row in linked.to_dict(orient="records"):
        sample_id = str(row["sample_id"])
        path = resolve_image(manifest_path, str(row["image_path"]))
        image_features = extract_image_features(
            load_image(path), spec, max_missing_fraction
        )
        cache[sample_id] = (str(row["split"]), image_features)
        packed.append((row, image_features))

    prefix = spec.prefix
    cold_key = f"cold_frac_{_token(spec.cold_thresholds[0])}"
    rows: list[dict[str, object]] = []
    for row, image_features in packed:
        temporal = _temporal_defaults(spec)
        prev = row.get("prev_sample_id")
        if not pd.isna(prev) and str(prev) in cache:
            prev_split, previous = cache[str(prev)]
            if prev_split != str(row["split"]):
                raise LeakageError(
                    f"temporal feature for {row['sample_id']} crossed a split"
                )
            temporal["has_prev"] = 1.0
            temporal[f"delta_{prefix}_mean"] = (
                image_features[f"{prefix}_mean"] - previous[f"{prefix}_mean"]
            )
            temporal[f"delta_{prefix}_min"] = (
                image_features[f"{prefix}_min"] - previous[f"{prefix}_min"]
            )
            temporal["delta_cold_primary"] = image_features[cold_key] - previous[cold_key]
            temporal["delta_eye_cdo_contrast"] = (
                image_features["eye_cdo_contrast"] - previous["eye_cdo_contrast"]
            )
            temporal["delta_radial_mean_00"] = (
                image_features["radial_mean_00"] - previous["radial_mean_00"]
            )
        klass = row.get("imd5_class")
        if pd.isna(klass):
            klass = wind_to_imd5(float(row["wind_speed_kt"]))
        built: dict[str, object] = {
            "sample_id": str(row["sample_id"]),
            "storm_id": str(row["storm_id"]),
            "split": str(row["split"]),
            "wind_speed_kt": float(row["wind_speed_kt"]),
            "imd5_class": klass,
        }
        if "is_synthetic" in row:
            built["is_synthetic"] = parse_is_synthetic(row["is_synthetic"])
        built.update(image_features)
        built.update(temporal)
        rows.append(built)
    table = pd.DataFrame(rows)
    banned = set(spec.feature_names()) & set(FORBIDDEN_FEATURES)
    if banned:
        raise LeakageError(f"feature names include labels: {sorted(banned)}")
    return table, dropped


def model_matrix(frame: pd.DataFrame, names: tuple[str, ...] | list[str]) -> np.ndarray:
    """Numeric feature matrix. Raises if a label column is requested."""
    banned = set(names) & set(FORBIDDEN_FEATURES)
    if banned:
        raise LeakageError(f"forbidden features in model matrix: {sorted(banned)}")
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise KeyError(f"missing feature columns: {missing}")
    return frame.loc[:, list(names)].to_numpy(dtype=np.float64)


def _impute_matrix(values: np.ndarray) -> np.ndarray:
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    filled = imputer.fit_transform(values)
    return np.where(np.isfinite(filled), filled, 0.0)


def select_features(
    frame: pd.DataFrame,
    feature_names: tuple[str, ...] | list[str],
    target: np.ndarray,
    *,
    source_split: str,
    correlation_threshold: float = 0.95,
    max_features: int = 25,
    method: str = "correlation_rf",
    n_estimators: int = 40,
    seed: int = 42,
) -> FeatureSelection:
    """Choose columns from a train frame only.

    Correlation filtering does not look at the label. The forest importance
    and RFE paths do, so the frame must already be the training split.
    """
    if source_split != "train":
        raise LeakageError("feature selection is allowed on the train split only")
    if "split" in frame.columns:
        present = set(frame["split"].astype(str).unique())
        if present != {"train"}:
            raise LeakageError(
                f"selection frame splits were {sorted(present)}, expected train"
            )
    names = tuple(feature_names)
    banned = set(names) & set(FORBIDDEN_FEATURES)
    if banned:
        raise LeakageError(f"forbidden features in selection: {sorted(banned)}")
    if len(frame) < 5:
        raise ValueError("feature selection needs at least 5 training rows")
    if len(target) != len(frame):
        raise ValueError("target must align with the selection frame")
    method = method.lower().strip()
    if method not in {"correlation_rf", "rfe", "none"}:
        raise ValueError(f"unknown selection method: {method}")

    values = _impute_matrix(model_matrix(frame, names))
    variance = values.var(axis=0)
    kept_mask = variance > 1e-8
    dropped_variance = tuple(name for name, keep in zip(names, kept_mask, strict=True) if not keep)
    kept_names = [name for name, keep in zip(names, kept_mask, strict=True) if keep]
    kept_values = values[:, kept_mask]
    if not kept_names:
        raise ValueError("every candidate feature was constant on the train split")

    dropped_correlation: list[str] = []
    if method in {"correlation_rf", "rfe", "none"} and len(kept_names) > 1:
        corr = np.corrcoef(kept_values, rowvar=False)
        corr = np.where(np.isfinite(corr), corr, 0.0)
        drop_flags = np.zeros(len(kept_names), dtype=bool)
        for left in range(len(kept_names)):
            if drop_flags[left]:
                continue
            for right in range(left + 1, len(kept_names)):
                if drop_flags[right]:
                    continue
                if abs(corr[left, right]) > correlation_threshold:
                    drop_flags[right] = True
                    dropped_correlation.append(kept_names[right])
        kept_names = [name for name, drop in zip(kept_names, drop_flags, strict=True) if not drop]
        kept_values = kept_values[:, ~drop_flags]

    y = np.asarray(target, dtype=np.float64).ravel()
    cap = max(1, int(max_features))
    if method == "none":
        chosen = kept_names[:cap]
        importances: dict[str, float] = {}
    elif method == "correlation_rf":
        rank_cap = cap if len(kept_names) > cap else len(kept_names)
        ranked, importances = _rank_by_forest(
            kept_values, y, kept_names, rank_cap, n_estimators, seed
        )
        chosen = ranked if len(kept_names) > cap else kept_names
    else:
        chosen, importances = _rank_by_rfe(
            kept_values, y, kept_names, min(cap, len(kept_names)), n_estimators, seed
        )

    return FeatureSelection(
        names=tuple(chosen),
        source_split=source_split,
        importances=importances,
        dropped_variance=dropped_variance,
        dropped_correlation=tuple(dropped_correlation),
    )


def _rank_by_forest(
    values: np.ndarray,
    target: np.ndarray,
    names: list[str],
    cap: int,
    n_estimators: int,
    seed: int,
) -> tuple[list[str], dict[str, float]]:
    forest = RandomForestRegressor(
        n_estimators=max(10, int(n_estimators)),
        max_depth=4,
        random_state=seed,
        n_jobs=1,
    )
    forest.fit(values, target)
    importance = np.asarray(forest.feature_importances_, dtype=np.float64)
    order = sorted(
        range(len(names)),
        key=lambda index: (-importance[index], names[index]),
    )
    chosen = [names[index] for index in order[:cap]]
    return chosen, {names[index]: float(importance[index]) for index in order}


def _rank_by_rfe(
    values: np.ndarray,
    target: np.ndarray,
    names: list[str],
    cap: int,
    n_estimators: int,
    seed: int,
) -> tuple[list[str], dict[str, float]]:
    forest = RandomForestRegressor(
        n_estimators=max(10, int(n_estimators)),
        max_depth=4,
        random_state=seed,
        n_jobs=1,
    )
    step = max(1, len(names) // 10)
    selector = RFE(
        forest,
        n_features_to_select=max(1, min(cap, len(names))),
        step=step,
    )
    selector.fit(values, target)
    ranking = np.asarray(selector.ranking_)
    chosen = [name for name, keep in zip(names, selector.support_, strict=True) if keep]
    importance = {name: float(1.0 / rank) for name, rank in zip(names, ranking, strict=True)}
    return chosen, importance
