"""Torch-free calibration diagnostics for uncertainty intervals.

Compares raw MC Dropout intervals with validation-fit scaled conformal
intervals.  Computes MACE, ENCE, 90% coverage and width, and Spearman
rank correlation between predicted uncertainty and absolute error.

This module does **not** import torch.  All inputs are NumPy arrays.

Terminology
-----------
- *Raw MC*: central percentile interval from MC-dropout samples.
- *Scaled conformal*: MC-dropout intervals whose half-widths are rescaled
  by a single scalar fit on validation residuals so that empirical coverage
  matches the nominal level on the validation set.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IntervalResult:
    """Coverage and width for one interval method at one nominal level."""

    nominal: float
    empirical_coverage: float
    mean_width_kt: float
    median_width_kt: float
    n: int


@dataclass(frozen=True)
class CalibrationReport:
    """Full calibration comparison between raw MC and scaled conformal."""

    # Per-level results
    raw_mc: dict[str, IntervalResult] = field(default_factory=dict)
    scaled_conformal: dict[str, IntervalResult] = field(default_factory=dict)

    # Aggregate scalar metrics
    mace: float = float("nan")
    ence: float = float("nan")
    coverage_90_raw: float = float("nan")
    width_90_raw_kt: float = float("nan")
    coverage_90_scaled: float = float("nan")
    width_90_scaled_kt: float = float("nan")
    spearman_rho: float = float("nan")
    spearman_p: float = float("nan")

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON output."""
        out: dict = {
            "mace": _safe_float(self.mace),
            "ence": _safe_float(self.ence),
            "spearman_rho": _safe_float(self.spearman_rho),
            "spearman_p": _safe_float(self.spearman_p),
            "coverage_90_raw": _safe_float(self.coverage_90_raw),
            "width_90_raw_kt": _safe_float(self.width_90_raw_kt),
            "coverage_90_scaled": _safe_float(self.coverage_90_scaled),
            "width_90_scaled_kt": _safe_float(self.width_90_scaled_kt),
        }
        for label, bucket in (("raw_mc", self.raw_mc), ("scaled_conformal", self.scaled_conformal)):
            out[label] = {
                key: {
                    "nominal": result.nominal,
                    "empirical_coverage": result.empirical_coverage,
                    "mean_width_kt": result.mean_width_kt,
                    "median_width_kt": result.median_width_kt,
                    "n": result.n,
                }
                for key, result in bucket.items()
            }
        return out


def _safe_float(value: float) -> float | None:
    """Return None for nan/inf so JSON serialization works."""
    if not np.isfinite(value):
        return None
    return float(value)


# ---------------------------------------------------------------------------
# Low-level interval helpers (no torch)
# ---------------------------------------------------------------------------

def _mc_intervals(
    samples: np.ndarray, level: float
) -> tuple[np.ndarray, np.ndarray]:
    """Central percentile interval from MC-dropout draws.

    Parameters
    ----------
    samples : array, shape (n_draws, n_rows)
    level : float in (0, 1)

    Returns
    -------
    lower, upper : arrays of shape (n_rows,)
    """
    draws = np.asarray(samples, dtype=np.float64)
    if draws.ndim != 2 or draws.shape[0] < 2:
        raise ValueError("samples must have shape (n_draws, n_rows) with >= 2 draws")
    alpha = (1.0 - level) / 2.0
    lower = np.quantile(draws, alpha, axis=0, method="linear")
    upper = np.quantile(draws, 1.0 - alpha, axis=0, method="linear")
    return np.asarray(lower, np.float64), np.asarray(upper, np.float64)


def _coverage_and_width(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[float, float, float]:
    """Empirical coverage, mean width, median width."""
    covered = (y_true >= lower) & (y_true <= upper)
    widths = upper - lower
    return float(np.mean(covered)), float(np.mean(widths)), float(np.median(widths))


def _fit_conformal_scale(
    y_true_val: np.ndarray,
    mc_samples_val: np.ndarray,
    level: float,
) -> float:
    """Fit a single multiplicative scalar on validation residuals.

    The scalar *s* is chosen so that the rescaled MC intervals (center ± s·hw)
    achieve exactly the requested coverage on the validation set, using
    conformalized quantile regression with a single global scale.
    """
    draws = np.asarray(mc_samples_val, dtype=np.float64)
    means = draws.mean(axis=0)
    lower, upper = _mc_intervals(draws, level)
    half_widths = (upper - lower) / 2.0
    # Avoid division by zero for constant predictions.
    half_widths = np.maximum(half_widths, 1e-12)
    conformity_scores = np.abs(y_true_val - means) / half_widths
    # The conformal quantile: the smallest s such that coverage >= level.
    quantile = np.quantile(conformity_scores, level, method="higher")
    return float(quantile)


def _scaled_intervals(
    mc_samples: np.ndarray, scale: float, level: float
) -> tuple[np.ndarray, np.ndarray]:
    """Apply conformal scale to MC intervals."""
    draws = np.asarray(mc_samples, dtype=np.float64)
    means = draws.mean(axis=0)
    lower_raw, upper_raw = _mc_intervals(draws, level)
    half_widths = (upper_raw - lower_raw) / 2.0
    lower = means - scale * half_widths
    upper = means + scale * half_widths
    return lower, upper


# ---------------------------------------------------------------------------
# Aggregate calibration metrics
# ---------------------------------------------------------------------------

def compute_mace(
    y_true: np.ndarray,
    mc_samples: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Mean Absolute Calibration Error (MACE).

    Evaluates the raw MC intervals at ``n_bins`` equally spaced nominal
    levels from 0.1 to 0.9 and returns the mean absolute gap between
    nominal and empirical coverage.
    """
    levels = np.linspace(0.1, 0.9, n_bins)
    gaps: list[float] = []
    for level in levels:
        lower, upper = _mc_intervals(mc_samples, float(level))
        cov, _, _ = _coverage_and_width(y_true, lower, upper)
        gaps.append(abs(cov - float(level)))
    return float(np.mean(gaps))


def compute_ence(
    y_true: np.ndarray,
    mc_samples: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Normalized Calibration Error (ENCE).

    Groups rows by predicted uncertainty (MC std), computes per-bin RMSE and
    mean predicted std, then averages the relative gap.
    """
    draws = np.asarray(mc_samples, dtype=np.float64)
    pred_mean = draws.mean(axis=0)
    pred_std = draws.std(axis=0, ddof=0)
    n = len(y_true)
    if n < n_bins:
        n_bins = max(1, n)
    order = np.argsort(pred_std)
    bin_edges = np.array_split(order, n_bins)
    ence_terms: list[float] = []
    for indices in bin_edges:
        if len(indices) == 0:
            continue
        rmse = float(np.sqrt(np.mean((y_true[indices] - pred_mean[indices]) ** 2)))
        mean_std = float(np.mean(pred_std[indices]))
        if mean_std > 0:
            ence_terms.append(abs(rmse - mean_std) / mean_std)
        else:
            ence_terms.append(0.0 if rmse == 0 else float("inf"))
    return float(np.mean(ence_terms)) if ence_terms else float("nan")


def compute_spearman(
    y_true: np.ndarray,
    mc_samples: np.ndarray,
) -> tuple[float, float]:
    """Spearman ρ between predicted uncertainty (MC std) and |error|.

    A higher ρ means the model's uncertainty tracks its actual errors.
    """
    draws = np.asarray(mc_samples, dtype=np.float64)
    pred_mean = draws.mean(axis=0)
    pred_std = draws.std(axis=0, ddof=0)
    abs_error = np.abs(y_true - pred_mean)
    if pred_std.std() == 0 or abs_error.std() == 0:
        return 0.0, 1.0
    result = sp_stats.spearmanr(pred_std, abs_error)
    return float(result.statistic), float(result.pvalue)


# ---------------------------------------------------------------------------
# Main calibration comparison
# ---------------------------------------------------------------------------

NOMINAL_LEVELS = (0.50, 0.80, 0.90, 0.95)


def compare_calibration(
    y_true_val: np.ndarray,
    mc_samples_val: np.ndarray,
    y_true_test: np.ndarray,
    mc_samples_test: np.ndarray,
    levels: tuple[float, ...] = NOMINAL_LEVELS,
) -> CalibrationReport:
    """Compare raw MC Dropout vs validation-fit scaled conformal intervals.

    Parameters
    ----------
    y_true_val, mc_samples_val
        Validation ground truth and MC draws (n_draws, n_val).
    y_true_test, mc_samples_test
        Test ground truth and MC draws (n_draws, n_test).
    levels
        Nominal coverage levels to evaluate.

    Returns
    -------
    CalibrationReport
    """
    y_val = np.asarray(y_true_val, dtype=np.float64).ravel()
    y_test = np.asarray(y_true_test, dtype=np.float64).ravel()
    mc_val = np.asarray(mc_samples_val, dtype=np.float64)
    mc_test = np.asarray(mc_samples_test, dtype=np.float64)

    if mc_val.ndim != 2 or mc_val.shape[0] < 2:
        raise ValueError("mc_samples_val must have shape (n_draws, n_val)")
    if mc_test.ndim != 2 or mc_test.shape[0] < 2:
        raise ValueError("mc_samples_test must have shape (n_draws, n_test)")
    if mc_val.shape[1] != y_val.size:
        raise ValueError("mc_samples_val columns must match y_true_val length")
    if mc_test.shape[1] != y_test.size:
        raise ValueError("mc_samples_test columns must match y_true_test length")

    raw_results: dict[str, IntervalResult] = {}
    scaled_results: dict[str, IntervalResult] = {}

    for level in levels:
        key = f"{level:.2f}"

        # Raw MC on test
        raw_lo, raw_hi = _mc_intervals(mc_test, level)
        cov_r, mw_r, mdw_r = _coverage_and_width(y_test, raw_lo, raw_hi)
        raw_results[key] = IntervalResult(
            nominal=level,
            empirical_coverage=cov_r,
            mean_width_kt=mw_r,
            median_width_kt=mdw_r,
            n=y_test.size,
        )

        # Scaled conformal on test (scale fit on val)
        scale = _fit_conformal_scale(y_val, mc_val, level)
        sc_lo, sc_hi = _scaled_intervals(mc_test, scale, level)
        cov_s, mw_s, mdw_s = _coverage_and_width(y_test, sc_lo, sc_hi)
        scaled_results[key] = IntervalResult(
            nominal=level,
            empirical_coverage=cov_s,
            mean_width_kt=mw_s,
            median_width_kt=mdw_s,
            n=y_test.size,
        )

    # Aggregate metrics on test set using raw MC
    mace = compute_mace(y_test, mc_test)
    ence = compute_ence(y_test, mc_test)
    rho, pval = compute_spearman(y_test, mc_test)

    cov_90_raw = raw_results.get("0.90", IntervalResult(0.9, float("nan"), float("nan"), float("nan"), 0)).empirical_coverage
    w_90_raw = raw_results.get("0.90", IntervalResult(0.9, float("nan"), float("nan"), float("nan"), 0)).mean_width_kt
    cov_90_sc = scaled_results.get("0.90", IntervalResult(0.9, float("nan"), float("nan"), float("nan"), 0)).empirical_coverage
    w_90_sc = scaled_results.get("0.90", IntervalResult(0.9, float("nan"), float("nan"), float("nan"), 0)).mean_width_kt

    return CalibrationReport(
        raw_mc=raw_results,
        scaled_conformal=scaled_results,
        mace=mace,
        ence=ence,
        coverage_90_raw=cov_90_raw,
        width_90_raw_kt=w_90_raw,
        coverage_90_scaled=cov_90_sc,
        width_90_scaled_kt=w_90_sc,
        spearman_rho=rho,
        spearman_p=pval,
    )
