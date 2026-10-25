"""Held-out intensity scores for supplied wind predictions.

Bins are our own grouping, IMD-style names at 17/34/48/64/91 kt. VSCS and
ESCS/SuCS are always reported. IMD class definitions use 3-minute winds.
NASA HURDAT2 labels are 1-minute winds. ``wind_averaging`` is stored on the
report (for example ``1min`` for NASA) and mixed periods are refused. This
module does not convert one averaging period into another.

The evaluated storms are the test side of ``storm_ids_by_split``. Pixel
normalization uses ``fit_split_normalization`` (defined on ``cloudie.train``,
which already calls the splits helper). Mean and ridge reuse
``src.baselines``. Persistence is the previous wind inside the same storm.

Monte Carlo dropout coverage reads a NumPy array of draws. This module does
not import torch. Residual-quantile widths are fit on validation storms and
scored on test storms. A coverage gap larger than the tolerance is labeled
``uncalibrated``. ``quote_as_result`` stays ``no``. This is not a forecast
and not an operational-readiness claim.

Checkpoint scoring stays in ``cloudie.evaluate``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

from src.baselines import baseline_params, fit_numeric_pipeline
from src.config import load_config
from src.features import model_matrix
from src.metrics import mean_baseline, metrics_by_class, regression_metrics
from src.plots import save_regression_scatter
from src.schema import (
    DEFAULT_EDGES_KT,
    DEFAULT_NAMES,
    recorded_wind_averaging,
    require_valid,
    wind_to_imd5,
)
from src.splits import (
    LeakageError,
    assert_no_forbidden_inputs,
    assign_storms,
    storm_ids_by_split,
)

BINNING_LABEL = "our own grouping, IMD-style names"
HIGHLIGHT_CODES = ("VSCS", "ESCS_SUCS")
HIGHLIGHT_DISPLAY = {"VSCS": "VSCS", "ESCS_SUCS": "ESCS/SuCS"}
WIND_AVERAGING_NOTE = (
    "IMD class definitions use 3-minute winds. "
    "NASA HURDAT2 labels are 1-minute winds and can be recorded as "
    "wind_averaging='1min'. This report stores the period it was given "
    "and does not convert between averaging periods."
)
BANNER = (
    "Software check only. Do not quote these numbers as experimental results. "
    "Bins are our own grouping, IMD-style names. "
    "IMD class definitions use 3-minute winds. NASA HURDAT2 labels are "
    "1-minute winds. wind_averaging is recorded and is not converted. "
    "Monte Carlo dropout coverage is not a calibrated forecast interval. "
    "Residual-quantile intervals are labeled uncalibrated when held-out "
    "coverage misses the nominal level by more than the tolerance. "
    "This is not a forecast, a warning, or an operational-readiness claim."
)
_EMPTY_METRICS = {"n": 0.0, "mae_kt": None, "rmse_kt": None, "bias_kt": None}
_LEVELS = (0.80, 0.90)
_TOLERANCE = 0.05
# Float slack so a gap that lands on the tolerance is not marked uncalibrated.
_GAP_SLACK = 1e-9


def _pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _level_key(level: float) -> str:
    return f"{float(level):.2f}"


def _check_levels(levels: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(level) for level in levels)
    if not values:
        raise ValueError("at least one nominal coverage level is required")
    for level in values:
        if not 0.0 < level < 1.0:
            raise ValueError(f"coverage level must be between 0 and 1, got {level}")
    return values


def interval_status(empirical: float, nominal: float, tolerance: float) -> str:
    """Label an interval ``uncalibrated`` when coverage misses by more than ``tolerance``."""
    if tolerance < 0:
        raise ValueError("coverage tolerance must be non-negative")
    gap = abs(float(empirical) - float(nominal))
    if gap > float(tolerance) + _GAP_SLACK:
        return "uncalibrated"
    return "within_tolerance"


def _bin_spec(config: dict) -> tuple[tuple[float, ...], tuple[str, ...]]:
    data = dict(config.get("data") or {})
    edges = tuple(float(edge) for edge in data.get("imd5_edges_kt", DEFAULT_EDGES_KT))
    names = tuple(str(name) for name in data.get("imd5_names", DEFAULT_NAMES))
    if len(edges) != len(names) or not edges:
        raise ValueError("intensity edges and names must be non-empty and the same length")
    return edges, names


def assign_bins(
    winds: np.ndarray,
    edges: Sequence[float] = DEFAULT_EDGES_KT,
    names: Sequence[str] = DEFAULT_NAMES,
) -> np.ndarray:
    """Project target wind onto our own grouping, IMD-style names.

    Winds below the first edge are None. The names are not a warning product.
    """
    edge_tuple = tuple(float(edge) for edge in edges)
    name_tuple = tuple(str(name) for name in names)
    values = np.asarray(winds, dtype=np.float64).ravel()
    return np.asarray(
        [wind_to_imd5(float(wind), edge_tuple, name_tuple) for wind in values],
        dtype=object,
    )


def _empty_metrics() -> dict[str, float | None]:
    return dict(_EMPTY_METRICS)


def _highlight(by_bin: dict[str, dict]) -> dict[str, dict]:
    highlighted: dict[str, dict] = {}
    for code in HIGHLIGHT_CODES:
        metrics = dict(by_bin.get(code) or _empty_metrics())
        metrics["display_name"] = HIGHLIGHT_DISPLAY[code]
        highlighted[code] = metrics
    return highlighted


def score_winds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    edges: Sequence[float] = DEFAULT_EDGES_KT,
    names: Sequence[str] = DEFAULT_NAMES,
) -> dict:
    """MAE, RMSE, and bias overall and per target-wind bin.

    Empty bins, including VSCS and ESCS/SuCS when absent, keep null errors.
    A zero error is not invented for an empty bin.
    """
    truth = np.asarray(y_true, dtype=np.float64).ravel()
    pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if truth.size == 0:
        by_bin = {str(name): _empty_metrics() for name in names}
        for code in HIGHLIGHT_CODES:
            by_bin.setdefault(code, _empty_metrics())
        return {
            "overall": _empty_metrics(),
            "by_bin": by_bin,
            "highlight": _highlight(by_bin),
            "n_below_lowest_edge": 0,
        }
    labels = assign_bins(truth, edges, names)
    found = metrics_by_class(truth, pred, labels)
    by_bin: dict[str, dict] = {}
    for name in tuple(names) + HIGHLIGHT_CODES:
        key = str(name)
        if key in by_bin:
            continue
        by_bin[key] = dict(found[key]) if key in found else _empty_metrics()
    below = int(sum(label is None for label in labels))
    return {
        "overall": regression_metrics(truth, pred),
        "by_bin": by_bin,
        "highlight": _highlight(by_bin),
        "n_below_lowest_edge": below,
    }


def persistence_predictions(frame: pd.DataFrame) -> np.ndarray:
    """Previous ``wind_speed_kt`` inside the same storm.

    Rows are ordered by ``timestamp`` when that column exists, otherwise by
    the given row order. The first row of each storm is NaN. Wind from another
    storm is never used. This is a baseline forecast, not a model feature.
    """
    if "storm_id" not in frame.columns or "wind_speed_kt" not in frame.columns:
        raise ValueError("persistence needs storm_id and wind_speed_kt")
    winds = pd.to_numeric(frame["wind_speed_kt"], errors="coerce").to_numpy(dtype=np.float64)
    table = pd.DataFrame(
        {
            "storm_id": frame["storm_id"].astype(str).to_numpy(),
            "wind": winds,
            "row": np.arange(len(frame)),
        }
    )
    if "timestamp" in frame.columns:
        parsed = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
        if bool(parsed.isna().any()):
            table["order"] = frame["timestamp"].astype(str).to_numpy()
        else:
            table["order"] = parsed.to_numpy()
    else:
        table["order"] = table["row"]
    table = table.sort_values(["storm_id", "order", "row"], kind="mergesort")
    table["prediction"] = table.groupby("storm_id", sort=False)["wind"].shift(1)
    table = table.sort_values("row", kind="mergesort")
    return table["prediction"].to_numpy(dtype=np.float64)


def mc_dropout_intervals(
    samples: np.ndarray, level: float
) -> tuple[np.ndarray, np.ndarray]:
    """Central interval from draws shaped ``(n_draws, n_rows)``.

    The draws are an array. Nothing here runs a network.
    """
    (level,) = _check_levels((level,))
    draws = np.asarray(samples, dtype=np.float64)
    if draws.ndim != 2 or draws.shape[0] < 2:
        raise ValueError("samples must have shape (n_draws, n_rows) with at least 2 draws")
    if not np.isfinite(draws).all():
        raise ValueError("MC-dropout samples must be finite")
    alpha = (1.0 - level) / 2.0
    lower = np.quantile(draws, alpha, axis=0, method="linear")
    upper = np.quantile(draws, 1.0 - alpha, axis=0, method="linear")
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def _sample_matrix(samples: np.ndarray, n_rows: int) -> np.ndarray:
    array = np.asarray(samples, dtype=np.float64)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError("MC-dropout samples must have shape (n_draws, n_rows)")
    if array.shape[0] == n_rows and array.shape[1] != n_rows:
        raise ValueError("samples must have shape (n_draws, n_rows), with draws on axis 0")
    if array.shape[0] < 2 or array.shape[1] != n_rows:
        raise ValueError(
            "samples must have shape (n_draws, n_rows) with at least 2 draws; "
            f"got {array.shape} for {n_rows} rows"
        )
    if not np.isfinite(array).all():
        raise ValueError("MC-dropout samples must be finite")
    return array


def mc_dropout_coverage(
    y_true: np.ndarray,
    samples: np.ndarray,
    levels: Sequence[float] = _LEVELS,
    tolerance: float = _TOLERANCE,
) -> dict[str, dict]:
    """Empirical coverage of central intervals drawn from ``samples``.

    ``samples`` is ``(n_draws, n_rows)`` aligned with ``y_true``. A level whose
    coverage misses the nominal rate by more than ``tolerance`` is
    ``uncalibrated``.
    """
    truth = np.asarray(y_true, dtype=np.float64).ravel()
    if truth.size == 0 or not np.isfinite(truth).all():
        raise ValueError("MC-dropout coverage needs finite targets")
    draws = _sample_matrix(samples, truth.size)
    checked = _check_levels(levels)
    report: dict[str, dict] = {}
    for level in checked:
        lower, upper = mc_dropout_intervals(draws, level)
        covered = (truth >= lower) & (truth <= upper)
        empirical = float(np.mean(covered))
        report[_level_key(level)] = {
            "nominal": float(level),
            "empirical_coverage": empirical,
            "n": float(truth.size),
            "n_covered": float(np.sum(covered)),
            "mean_interval_width_kt": float(np.mean(upper - lower)),
            "status": interval_status(empirical, level, tolerance),
        }
    return report


def residual_quantile_coverage(
    y_true_val: np.ndarray,
    y_pred_val: np.ndarray,
    y_true_test: np.ndarray,
    y_pred_test: np.ndarray,
    levels: Sequence[float] = _LEVELS,
    tolerance: float = _TOLERANCE,
) -> dict[str, dict]:
    """Fit quantiles of ``|residual|`` on validation rows and score test rows.

    The half-width is the validation quantile at the nominal level. Test
    coverage is the fraction of test rows with ``|residual|`` at or below that
    width. Validation rows are not included in the coverage count.
    """
    val_truth = np.asarray(y_true_val, dtype=np.float64).ravel()
    val_pred = np.asarray(y_pred_val, dtype=np.float64).ravel()
    test_truth = np.asarray(y_true_test, dtype=np.float64).ravel()
    test_pred = np.asarray(y_pred_test, dtype=np.float64).ravel()
    if val_truth.shape != val_pred.shape or val_truth.size == 0:
        raise ValueError("validation residuals need a non-empty aligned pair")
    if test_truth.shape != test_pred.shape or test_truth.size == 0:
        raise ValueError("test residuals need a non-empty aligned pair")
    if not (
        np.isfinite(val_truth).all()
        and np.isfinite(val_pred).all()
        and np.isfinite(test_truth).all()
        and np.isfinite(test_pred).all()
    ):
        raise ValueError("residual-quantile inputs must be finite")
    val_abs = np.abs(val_pred - val_truth)
    test_abs = np.abs(test_pred - test_truth)
    report: dict[str, dict] = {}
    for level in _check_levels(levels):
        half_width = float(np.quantile(val_abs, level, method="linear"))
        covered = test_abs <= half_width
        empirical = float(np.mean(covered))
        report[_level_key(level)] = {
            "nominal": float(level),
            "absolute_residual_quantile_kt": half_width,
            "empirical_coverage": empirical,
            "n": float(test_truth.size),
            "n_covered": float(np.sum(covered)),
            "fit_split": "val",
            "eval_split": "test",
            "status": interval_status(empirical, level, tolerance),
        }
    return report


def _positions(frame: pd.DataFrame, split_name: str) -> np.ndarray:
    return np.flatnonzero(frame["split"].astype(str).to_numpy() == split_name)


def _ensure_split(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Keep a complete storm split, or assign one. Random frame splits raise."""
    if "split" in frame.columns:
        if bool(frame["split"].isna().any()):
            raise LeakageError("split has nulls; refusing to fill them frame by frame")
        unknown = set(frame["split"].astype(str).unique()) - {"train", "val", "test"}
        if unknown:
            raise LeakageError(f"unknown splits: {sorted(unknown)}")
        return frame
    split_cfg = dict(config.get("split") or {})
    strategy = str(split_cfg.get("strategy", "storm_wise"))
    holdout_basin = split_cfg.get("holdout_basin") or None
    if holdout_basin in {None, "", "null"}:
        holdout_basin = None
    return assign_storms(
        frame,
        strategy,
        test_fraction=float(split_cfg.get("test_fraction", 0.2)),
        val_fraction=float(split_cfg.get("val_fraction", 0.2)),
        seed=int(split_cfg.get("seed", 0)),
        holdout_years=[int(year) for year in (split_cfg.get("holdout_years") or [])],
        holdout_basin=holdout_basin,
    )


def _require_finite_predictions(frame: pd.DataFrame, column: str, split_name: str) -> np.ndarray:
    if column not in frame.columns:
        raise ValueError(f"missing prediction column {column}")
    positions = _positions(frame, split_name)
    if positions.size == 0:
        raise ValueError(f"split {split_name} has no rows")
    values = pd.to_numeric(frame.iloc[positions][column], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{split_name} predictions must be finite")
    return values


def _train_norm_stats(frame: pd.DataFrame, manifest: str | Path, config: dict):
    """Fit train-only pixel stats through the existing helper."""
    from cloudie.train import fit_split_normalization

    prepared = frame.copy()
    if "_class" not in prepared.columns:
        prepared["_class"] = None
    stats = fit_split_normalization(prepared, manifest, config)
    if stats.source_split != "train":
        raise LeakageError("normalization stats may be fit on the train split only")
    return stats


def _synthetic_label(frame: pd.DataFrame) -> str:
    flags = frame["is_synthetic"].astype(bool)
    if bool(flags.all()):
        return "yes"
    if bool(flags.any()):
        return "mixed"
    return "no"


def _baseline_predictions(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    config: dict,
    train_pos: np.ndarray,
    test_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str], str, str]:
    """Fit mean and ridge on train rows. Return test predictions and fit notes."""
    names = tuple(feature_names)
    if not names:
        raise ValueError("ridge needs at least one feature column")
    train = frame.iloc[train_pos]
    test = frame.iloc[test_pos]
    x_train = model_matrix(train, names)
    x_test = model_matrix(test, names)
    y_train = train["wind_speed_kt"].to_numpy(dtype=np.float64)
    params = baseline_params(config)
    fitted: set[str] = set()
    seed = int(params["seed"])
    mean_hat, _mean_pipe, mean_note = fit_numeric_pipeline(
        x_train,
        y_train,
        x_test,
        task="regression",
        model_name="mean",
        params=params,
        seed=seed,
        train_storms=train["storm_id"].astype(str),
        fitted_storms=fitted,
    )
    ridge_hat, _ridge_pipe, ridge_note = fit_numeric_pipeline(
        x_train,
        y_train,
        x_test,
        task="regression",
        model_name="ridge",
        params=params,
        seed=seed,
        train_storms=train["storm_id"].astype(str),
        fitted_storms=fitted,
    )
    reference = mean_baseline(y_train, test_pos.size)
    if not np.allclose(mean_hat, reference):
        raise RuntimeError("mean baseline disagreed with src.metrics.mean_baseline")
    return (
        np.asarray(mean_hat, dtype=np.float64),
        np.asarray(ridge_hat, dtype=np.float64),
        sorted(fitted),
        str(mean_note),
        str(ridge_note),
    )


def _plain(value):
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not np.isfinite(number):
            return None
        return number
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _csv_metric_row(series: str, bin_name: str, metrics: dict, report: dict) -> dict:
    return {
        "series": series,
        "bin": bin_name,
        "n": metrics.get("n"),
        "mae_kt": metrics.get("mae_kt"),
        "rmse_kt": metrics.get("rmse_kt"),
        "bias_kt": metrics.get("bias_kt"),
        "wind_averaging": report["wind_averaging"],
        "binning": report["binning"],
        "quote_as_result": report["quote_as_result"],
        "synthetic": report["synthetic"],
        "status": "",
    }


def _metric_table(report: dict) -> pd.DataFrame:
    rows: list[dict] = []
    blocks = (
        ("model", report["model"]),
        ("mean", report["baselines"]["mean"]),
        ("ridge", report["baselines"]["ridge"]),
        ("persistence", report["baselines"]["persistence"]),
        ("paired_model", report["baselines"]["paired"]["model"]),
        ("paired_mean", report["baselines"]["paired"]["mean"]),
        ("paired_ridge", report["baselines"]["paired"]["ridge"]),
        ("paired_persistence", report["baselines"]["paired"]["persistence"]),
    )
    for series, block in blocks:
        rows.append(_csv_metric_row(series, "overall", block["overall"], report))
        for name, metrics in block["by_bin"].items():
            rows.append(_csv_metric_row(series, name, metrics, report))
    uncertainty = report["uncertainty"]
    for method in ("mc_dropout", "residual_quantile"):
        payload = uncertainty.get(method)
        if not payload:
            continue
        for level, metrics in payload["levels"].items():
            rows.append(
                {
                    "series": method,
                    "bin": level,
                    "n": metrics.get("n"),
                    "mae_kt": None,
                    "rmse_kt": None,
                    "bias_kt": None,
                    "wind_averaging": report["wind_averaging"],
                    "binning": report["binning"],
                    "quote_as_result": report["quote_as_result"],
                    "synthetic": report["synthetic"],
                    "status": metrics.get("status"),
                    "nominal": metrics.get("nominal"),
                    "empirical_coverage": metrics.get("empirical_coverage"),
                }
            )
    return pd.DataFrame(rows)


def save_error_by_bin(by_bin: dict[str, dict], names: Sequence[str], path: str | Path) -> Path:
    """Bar chart of MAE by bin. Empty bins are marked n=0 and have no bar."""
    plt = _pyplot()
    ordered = [str(name) for name in names]
    for code in HIGHLIGHT_CODES:
        if code not in ordered:
            ordered.append(code)
    values = []
    for name in ordered:
        mae = (by_bin.get(name) or _empty_metrics())["mae_kt"]
        values.append(np.nan if mae is None else float(mae))
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    positions = np.arange(len(ordered))
    finite = [value for value in values if np.isfinite(value)]
    for position, value in zip(positions, values, strict=True):
        if np.isfinite(value):
            color = "#9b2226" if ordered[position] in HIGHLIGHT_CODES else "#1d3557"
            axis.bar(position, value, color=color, width=0.72)
        else:
            axis.text(position, 0.0, "n=0", ha="center", va="bottom", fontsize=8, color="#6c757d")
    labels = [HIGHLIGHT_DISPLAY.get(name, name) for name in ordered]
    axis.set_xticks(positions, labels, rotation=20, ha="right")
    for tick, name in zip(axis.get_xticklabels(), ordered, strict=True):
        if name in HIGHLIGHT_CODES:
            tick.set_fontweight("bold")
    axis.set_ylabel("MAE (kt)")
    axis.set_xlabel(BINNING_LABEL)
    axis.set_title("Error by bin on held-out storms. VSCS and ESCS/SuCS are marked. Not a forecast.")
    if finite:
        axis.set_ylim(0.0, max(finite) * 1.25 + 1.0)
    else:
        axis.set_ylim(0.0, 1.0)
    figure.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=100)
    plt.close(figure)
    return destination


def save_coverage_figure(uncertainty: dict, path: str | Path) -> Path:
    """Nominal versus empirical coverage. Misses are annotated uncalibrated."""
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(5.6, 5.2))
    tolerance = float(uncertainty["coverage_tolerance"])
    grid = np.linspace(0.0, 1.0, 40)
    axis.fill_between(
        grid,
        grid - tolerance,
        grid + tolerance,
        color="#dee2e6",
        label="tolerance",
    )
    axis.plot([0.0, 1.0], [0.0, 1.0], color="#6c757d", linewidth=1)
    series = (
        ("MC dropout", uncertainty.get("mc_dropout"), "#1d3557"),
        ("residual quantile", uncertainty.get("residual_quantile"), "#9b2226"),
    )
    for label, payload, color in series:
        if not payload:
            continue
        nominals = []
        empiricals = []
        for row in payload["levels"].values():
            nominals.append(float(row["nominal"]))
            empiricals.append(float(row["empirical_coverage"]))
            if row["status"] == "uncalibrated":
                axis.annotate(
                    "uncalibrated",
                    (float(row["nominal"]), float(row["empirical_coverage"])),
                    textcoords="offset points",
                    xytext=(6, 6),
                    fontsize=8,
                    color=color,
                )
        axis.plot(nominals, empiricals, marker="o", color=color, label=label)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Nominal coverage")
    axis.set_ylabel("Empirical coverage on test storms")
    axis.set_title("Held-out coverage. Not a calibrated forecast.")
    axis.legend(frameon=False)
    figure.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=100)
    plt.close(figure)
    return destination


def _write_report(report: dict, destination: Path) -> dict[str, str]:
    destination.mkdir(parents=True, exist_ok=True)
    _pyplot()
    (destination / "README.txt").write_text(BANNER + "\n", encoding="utf-8")
    metrics_path = destination / "metrics.json"
    table_path = destination / "metrics.csv"
    _metric_table(report).to_csv(table_path, index=False)
    test_truth = np.asarray(report["_plot_truth"], dtype=np.float64)
    test_pred = np.asarray(report["_plot_pred"], dtype=np.float64)
    parity = save_regression_scatter(
        test_truth,
        test_pred,
        destination / "parity.png",
        title="Held-out parity. Not a forecast.",
    )
    error_path = save_error_by_bin(
        report["model"]["by_bin"],
        report["bin_names"],
        destination / "error_by_bin.png",
    )
    coverage_path = save_coverage_figure(report["uncertainty"], destination / "coverage.png")
    outputs = {
        "metrics_json": str(metrics_path),
        "metrics_csv": str(table_path),
        "readme": str(destination / "README.txt"),
        "parity": str(parity),
        "error_by_bin": str(error_path),
        "coverage": str(coverage_path),
    }
    report["outputs"] = outputs
    public = {key: value for key, value in report.items() if not key.startswith("_")}
    metrics_path.write_text(json.dumps(_plain(public), indent=2) + "\n", encoding="utf-8")
    return outputs


def evaluate(
    frame: pd.DataFrame,
    manifest: str | Path,
    config: dict,
    *,
    feature_names: Sequence[str],
    prediction_column: str = "y_pred",
    out_dir: str | Path | None = None,
    levels: Sequence[float] = _LEVELS,
    coverage_tolerance: float | None = None,
    samples: np.ndarray | None = None,
    samples_aligned_to: str = "test",
) -> dict:
    """Score test storms and write figures plus a metrics file under ``out_dir``.

    ``frame`` carries the storm split, targets, predictions, and the feature
    columns used by ridge. Images referenced by ``manifest`` are used only to
    fit train normalization. Train storms are not part of the evaluated set.
    """
    limits = list((config.get("data") or {}).get("plausible_wind_kt") or [0, 200])
    work = require_valid(
        frame,
        wind_min_kt=float(limits[0]),
        wind_max_kt=float(limits[1]),
    )
    work = _ensure_split(work, config).reset_index(drop=True)
    groups = storm_ids_by_split(work)
    if not groups["train"] or not groups["val"] or not groups["test"]:
        raise LeakageError("train, val, and test storms are all required")
    if groups["train"] & groups["test"] or groups["train"] & groups["val"]:
        raise LeakageError("train storms leaked into a held-out split")
    edges, names = _bin_spec(config)
    wind_averaging = recorded_wind_averaging(work)
    train_pos = _positions(work, "train")
    val_pos = _positions(work, "val")
    test_pos = _positions(work, "test")
    evaluated = frozenset(work.iloc[test_pos]["storm_id"].astype(str))
    if evaluated & groups["train"]:
        raise LeakageError(f"train storms are in the evaluated set: {sorted(evaluated & groups['train'])}")
    if evaluated != groups["test"]:
        raise LeakageError("evaluated storms are not exactly the test storms")
    assert_no_forbidden_inputs(feature_names, "refusing to fit ridge on")
    val_pred = _require_finite_predictions(work, prediction_column, "val")
    test_pred = _require_finite_predictions(work, prediction_column, "test")
    val_truth = work.iloc[val_pos]["wind_speed_kt"].to_numpy(dtype=np.float64)
    test_truth = work.iloc[test_pos]["wind_speed_kt"].to_numpy(dtype=np.float64)
    stats = _train_norm_stats(work, manifest, config)
    checked_levels = _check_levels(levels)
    if coverage_tolerance is None:
        configured = dict(config.get("evaluate") or {})
        tolerance = float(configured.get("coverage_tolerance", _TOLERANCE))
    else:
        tolerance = float(coverage_tolerance)
    if tolerance < 0:
        raise ValueError("coverage tolerance must be non-negative")

    model_block = score_winds(test_truth, test_pred, edges, names)
    model_block["storms"] = sorted(evaluated)
    mean_hat, ridge_hat, fitted_ids, mean_note, ridge_note = _baseline_predictions(
        work, feature_names, config, train_pos, test_pos
    )
    fitted = frozenset(fitted_ids)
    if not fitted <= groups["train"]:
        raise LeakageError("baseline fit saw storms outside the training split")
    held_out = groups["val"] | groups["test"]
    if fitted & held_out:
        leaked = sorted(fitted & held_out)
        raise LeakageError(f"baseline fit saw held-out storms: {leaked}")

    persist_all = persistence_predictions(work)
    persist_test = persist_all[test_pos]
    paired = np.isfinite(persist_test)
    mean_block = score_winds(test_truth, mean_hat, edges, names)
    ridge_block = score_winds(test_truth, ridge_hat, edges, names)
    if int(np.sum(paired)) == 0:
        persistence_block = score_winds([], [], edges, names)
        paired_model = score_winds([], [], edges, names)
        paired_mean = score_winds([], [], edges, names)
        paired_ridge = score_winds([], [], edges, names)
        paired_persistence = score_winds([], [], edges, names)
    else:
        persistence_block = score_winds(test_truth[paired], persist_test[paired], edges, names)
        paired_model = score_winds(test_truth[paired], test_pred[paired], edges, names)
        paired_mean = score_winds(test_truth[paired], mean_hat[paired], edges, names)
        paired_ridge = score_winds(test_truth[paired], ridge_hat[paired], edges, names)
        paired_persistence = persistence_block
    persistence_storms = sorted(
        set(work.iloc[test_pos]["storm_id"].astype(str).to_numpy()[paired])
    )
    mean_block["storms"] = sorted(evaluated)
    ridge_block["storms"] = sorted(evaluated)
    persistence_block["storms"] = persistence_storms

    if samples is None:
        mc_report = None
    else:
        if samples_aligned_to == "test":
            draw_matrix = _sample_matrix(samples, test_pos.size)
            mc_truth = test_truth
        elif samples_aligned_to == "frame":
            draw_matrix = _sample_matrix(samples, len(work))[:, test_pos]
            mc_truth = test_truth
        else:
            raise ValueError("samples_aligned_to must be 'test' or 'frame'")
        mc_report = {
            "source": "sampled predictions",
            "imports_torch": False,
            "levels": mc_dropout_coverage(mc_truth, draw_matrix, checked_levels, tolerance),
        }
    residual_report = {
        "fit_split": "val",
        "eval_split": "test",
        "val_storms": sorted(groups["val"]),
        "test_storms": sorted(groups["test"]),
        "levels": residual_quantile_coverage(
            val_truth, val_pred, test_truth, test_pred, checked_levels, tolerance
        ),
    }
    reports_root = Path((config.get("paths") or {}).get("reports", "reports"))
    destination = Path(out_dir) if out_dir is not None else reports_root / "heldout"
    report = {
        "note": BANNER,
        "quote_as_result": "no",
        "present_as_forecast": False,
        "binning": BINNING_LABEL,
        "reported_especially": [HIGHLIGHT_DISPLAY[code] for code in HIGHLIGHT_CODES],
        "wind_averaging": wind_averaging,
        "wind_averaging_note": WIND_AVERAGING_NOTE,
        "edges_kt": [float(edge) for edge in edges],
        "bin_names": [str(name) for name in names],
        "eval_split": "test",
        "synthetic": _synthetic_label(work),
        "train_storms": sorted(groups["train"]),
        "val_storms": sorted(groups["val"]),
        "test_storms": sorted(groups["test"]),
        "evaluated_storms": sorted(evaluated),
        "normalization": {
            "mean": float(stats.mean),
            "std": float(stats.std),
            "source_split": str(stats.source_split),
            "helper": "cloudie.train.fit_split_normalization",
        },
        "model": model_block,
        "baselines": {
            "fit_split": "train",
            "score_split": "test",
            "fitted_storms": sorted(fitted),
            "mean_fit_notes": mean_note,
            "ridge_fit_notes": ridge_note,
            "persistence_definition": (
                "previous wind_speed_kt in the same storm; "
                "the first frame of a storm is not scored"
            ),
            "mean": mean_block,
            "ridge": ridge_block,
            "persistence": persistence_block,
            "paired": {
                "rows": "test rows with a previous wind in the same storm",
                "model": paired_model,
                "mean": paired_mean,
                "ridge": paired_ridge,
                "persistence": paired_persistence,
            },
        },
        "uncertainty": {
            "coverage_tolerance": tolerance,
            "nominal_levels": [float(level) for level in checked_levels],
            "mc_dropout": mc_report,
            "residual_quantile": residual_report,
        },
        "_plot_truth": test_truth,
        "_plot_pred": test_pred,
    }
    _write_report(report, destination)
    return {key: value for key, value in report.items() if not key.startswith("_")}


def main(argv: list[str] | None = None) -> dict:
    """CLI entry point. Prints paths and the honesty banner, not a score to quote."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--frame",
        type=Path,
        required=True,
        help="CSV with storms, winds, y_pred, and features",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path used to resolve image_path. Defaults to --frame.",
    )
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--prediction-column", default="y_pred")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--levels", nargs="+", type=float, default=list(_LEVELS))
    parser.add_argument("--coverage-tolerance", type=float, default=None)
    parser.add_argument(
        "--samples",
        type=Path,
        default=None,
        help="NPY array of MC-dropout draws, shape (n_draws, n_test_rows)",
    )
    parser.add_argument(
        "--samples-aligned-to",
        choices=("test", "frame"),
        default="test",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    table = pd.read_csv(args.frame)
    samples = None if args.samples is None else np.load(args.samples)
    report = evaluate(
        table,
        args.manifest or args.frame,
        config,
        feature_names=args.features,
        prediction_column=args.prediction_column,
        out_dir=args.out,
        levels=args.levels,
        coverage_tolerance=args.coverage_tolerance,
        samples=samples,
        samples_aligned_to=args.samples_aligned_to,
    )
    print(BANNER)
    print(f"wind_averaging={report['wind_averaging']}")
    print("quote_as_result=no")
    print(f"Wrote {report['outputs']['metrics_json']}")
    return report


if __name__ == "__main__":
    main()
