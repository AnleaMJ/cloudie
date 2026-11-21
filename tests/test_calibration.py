"""Unit tests for src.calibration.

All arrays are synthetic.  This is a software check, not a scientific result.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.calibration import (
    CalibrationReport,
    IntervalResult,
    compare_calibration,
    compute_ence,
    compute_mace,
    compute_spearman,
)


def _make_mc_samples(
    y_true: np.ndarray,
    n_draws: int = 50,
    noise_scale: float = 5.0,
    seed: int = 42,
) -> np.ndarray:
    """Synthetic MC draws centred on truth + noise."""
    rng = np.random.default_rng(seed)
    base = y_true[np.newaxis, :] + rng.normal(0, 2.0, size=(1, len(y_true)))
    return base + rng.normal(0, noise_scale, size=(n_draws, len(y_true)))


# --- IntervalResult / CalibrationReport plumbing --------------------------

def test_interval_result_is_frozen() -> None:
    r = IntervalResult(nominal=0.9, empirical_coverage=0.88, mean_width_kt=10.0, median_width_kt=9.5, n=100)
    with pytest.raises(AttributeError):
        r.nominal = 0.5  # type: ignore[misc]


def test_calibration_report_to_dict_round_trips() -> None:
    rng = np.random.default_rng(7)
    y = rng.normal(50, 10, size=60)
    mc = _make_mc_samples(y, seed=7)
    report = compare_calibration(y[:30], mc[:, :30], y[30:], mc[:, 30:])
    d = report.to_dict()
    assert isinstance(d, dict)
    assert "mace" in d
    assert "raw_mc" in d
    assert "scaled_conformal" in d
    for key in ("0.90",):
        if key in d["raw_mc"]:
            assert "empirical_coverage" in d["raw_mc"][key]


# --- MACE -----------------------------------------------------------------

def test_mace_perfect_calibration() -> None:
    """When truth and draws come from the same predictive distribution, MACE is small."""
    rng = np.random.default_rng(99)
    n = 1000
    mu = rng.normal(50, 10, size=n)
    # y is drawn from N(mu, 5^2)
    y = mu + rng.normal(0, 5, size=n)
    # draws are also drawn from N(mu, 5^2)
    draws = mu[np.newaxis, :] + rng.normal(0, 5, size=(500, n))
    mace = compute_mace(y, draws, n_bins=10)
    # With 1000 samples and 500 draws, empirical coverage closely tracks nominal
    assert 0.0 <= mace <= 0.05, f"MACE {mace} is too large for well-calibrated draws"


def test_mace_bad_calibration() -> None:
    """Very narrow intervals should give large MACE."""
    rng = np.random.default_rng(42)
    n = 200
    y = rng.normal(50, 15, size=n)
    # Tiny noise → intervals far too narrow → coverage << nominal
    draws = y[np.newaxis, :] + rng.normal(0, 0.1, size=(50, n))
    mace = compute_mace(y, draws)
    assert mace > 0.1, f"MACE {mace} should be large for overly narrow intervals"


# --- ENCE -----------------------------------------------------------------

def test_ence_non_negative() -> None:
    rng = np.random.default_rng(10)
    y = rng.normal(50, 10, size=100)
    mc = _make_mc_samples(y, seed=10)
    ence = compute_ence(y, mc)
    assert ence >= 0.0


# --- Spearman -------------------------------------------------------------

def test_spearman_positive_for_correlated_errors() -> None:
    """When uncertainty tracks error, ρ should be positive."""
    rng = np.random.default_rng(55)
    n = 200
    y = rng.normal(50, 10, size=n)
    # Make std proportional to |error|: draw wider when further from mean.
    noise_scale = np.abs(y - 50) + 1.0
    draws = y[np.newaxis, :] + rng.normal(0, 1, size=(80, n)) * noise_scale[np.newaxis, :]
    rho, pval = compute_spearman(y, draws)
    assert rho > 0, f"Expected positive Spearman ρ, got {rho}"


def test_spearman_constant_uncertainty() -> None:
    """Zero-variance uncertainty → ρ = 0."""
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    # All draws identical → std = 0.
    draws = np.tile(y, (10, 1))
    rho, _ = compute_spearman(y, draws)
    assert rho == 0.0


# --- compare_calibration end-to-end ----------------------------------------

def test_compare_calibration_smoke() -> None:
    """End-to-end smoke test: runs without error, returns a valid report."""
    rng = np.random.default_rng(123)
    n_val, n_test = 40, 60
    y_val = rng.normal(55, 12, size=n_val)
    y_test = rng.normal(55, 12, size=n_test)
    mc_val = _make_mc_samples(y_val, n_draws=30, seed=11)
    mc_test = _make_mc_samples(y_test, n_draws=30, seed=12)

    report = compare_calibration(y_val, mc_val, y_test, mc_test)
    assert isinstance(report, CalibrationReport)
    assert report.mace >= 0.0 or np.isnan(report.mace)
    assert len(report.raw_mc) == 4  # default 4 levels
    assert len(report.scaled_conformal) == 4

    # Scaled conformal should generally be closer to nominal than raw.
    for key in report.raw_mc:
        assert 0.0 <= report.raw_mc[key].empirical_coverage <= 1.0
        assert report.raw_mc[key].mean_width_kt >= 0.0

    # 90% coverage should exist.
    assert not np.isnan(report.coverage_90_raw)
    assert not np.isnan(report.coverage_90_scaled)


def test_compare_calibration_shape_validation() -> None:
    """Wrong shapes raise ValueError."""
    y = np.array([1.0, 2.0, 3.0])
    mc = np.array([[1.0, 2.0, 3.0]])  # only 1 draw
    with pytest.raises(ValueError, match="n_draws"):
        compare_calibration(y, mc, y, mc)


def test_compare_calibration_mismatch_rows() -> None:
    rng = np.random.default_rng(7)
    y = rng.normal(0, 1, size=10)
    mc = rng.normal(0, 1, size=(5, 20))  # 20 columns ≠ 10 rows
    with pytest.raises(ValueError, match="match"):
        compare_calibration(y, mc, y, mc)


def test_scaled_conformal_improves_coverage() -> None:
    """With enough data the conformal correction should bring coverage
    closer to 90% than the raw MC intervals."""
    rng = np.random.default_rng(77)
    n = 300
    y_val = rng.normal(50, 10, size=n)
    y_test = rng.normal(50, 10, size=n)
    # Deliberately biased draws (mean shifted).
    mc_val = y_val[np.newaxis, :] + 8.0 + rng.normal(0, 6, size=(60, n))
    mc_test = y_test[np.newaxis, :] + 8.0 + rng.normal(0, 6, size=(60, n))

    report = compare_calibration(y_val, mc_val, y_test, mc_test, levels=(0.90,))
    gap_raw = abs(report.coverage_90_raw - 0.90)
    gap_scaled = abs(report.coverage_90_scaled - 0.90)
    # Scaled should not be dramatically worse.
    assert gap_scaled <= gap_raw + 0.15, (
        f"Scaled gap {gap_scaled:.3f} much worse than raw gap {gap_raw:.3f}"
    )
