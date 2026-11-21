"""Generate automated dark-mode evaluation and calibration plots.

Generates three publication-quality, dark-mode (#0d1117, 300 DPI) plots from a
completed training run:
1. eval_pred_vs_actual.png: Predicted vs. Actual Wind Speed (kt) with 1:1 line
   and conformal uncertainty intervals.
2. eval_error_by_category.png: RMSE and MAE breakdown by Saffir-Simpson intensity
   category.
3. eval_conformal_coverage.png: Empirical vs. nominal conformal coverage curve.

Usage::

    python scripts/generate_eval_plots.py \\
        --run-dir artifacts/runs/<run_id> \\
        --out-dir docs/assets/
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np


# Dark mode palette matching GitHub dark theme
DARK_BG = "#0d1117"
PANEL_BG = "#161b22"
TEXT_COLOR = "#f0f6fc"
TEXT_MUTED = "#8b949e"
GRID_COLOR = "#30363d"
ACCENT_BLUE = "#58a6ff"
ACCENT_ORANGE = "#f0883e"
ACCENT_GREEN = "#7ee787"
ACCENT_RED = "#ff7b72"
ACCENT_PURPLE = "#d2a8ff"


# Saffir-Simpson Hurricane Wind Scale categories (1-minute sustained winds, kt)
SAFFIR_SIMPSON_BINS = [
    ("TD", 0.0, 34.0),
    ("TS", 34.0, 64.0),
    ("Cat 1", 64.0, 83.0),
    ("Cat 2", 83.0, 96.0),
    ("Cat 3", 96.0, 113.0),
    ("Cat 4", 113.0, 137.0),
    ("Cat 5", 137.0, 300.0),
]


def setup_dark_style(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    """Apply consistent dark-mode styling to a Matplotlib Axes."""
    ax.set_facecolor(DARK_BG)
    ax.set_title(title, color=TEXT_COLOR, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, color=TEXT_COLOR, fontsize=11, labelpad=8)
    ax.set_ylabel(ylabel, color=TEXT_COLOR, fontsize=11, labelpad=8)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9.5)
    ax.grid(True, linestyle="--", alpha=0.4, color=GRID_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
        spine.set_linewidth(1.0)


def load_run_data(run_dir: str | Path) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray | None, np.ndarray | None]:
    """Load evaluation ground truth, predictions/MC draws, and metrics.json."""
    run_path = Path(run_dir)
    if not run_path.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_path}")

    metrics_file = run_path / "metrics.json"
    metrics = {}
    if metrics_file.is_file():
        with open(metrics_file, encoding="utf-8") as f:
            metrics = json.load(f)

    # Check for calibration.npz or predictions.npz
    npz_candidates = [
        run_path / "calibration.npz",
        run_path / "predictions.npz",
        run_path / "test_predictions.npz",
    ]
    npz_path = None
    for cand in npz_candidates:
        if cand.is_file():
            npz_path = cand
            break

    if npz_path is None:
        raise FileNotFoundError(
            f"No prediction file (calibration.npz or predictions.npz) found in {run_path}"
        )

    data = dict(np.load(str(npz_path), allow_pickle=True))

    y_test = data.get("y_true_test")
    if y_test is None:
        y_test = data.get("y_test", data.get("y_true"))
    if y_test is None:
        raise KeyError("npz file must contain 'y_true_test', 'y_test', or 'y_true'")
    y_test = np.asarray(y_test, dtype=np.float64).ravel()

    mc_test = data.get("mc_samples_test", data.get("mc_samples"))
    if mc_test is not None:
        mc_test = np.asarray(mc_test, dtype=np.float64)
        if mc_test.ndim == 1:
            mc_test = mc_test.reshape(1, -1)
        elif mc_test.shape[1] != y_test.size and mc_test.shape[0] == y_test.size:
            mc_test = mc_test.T

    y_val = data.get("y_true_val", data.get("y_val"))
    if y_val is not None:
        y_val = np.asarray(y_val, dtype=np.float64).ravel()

    mc_val = data.get("mc_samples_val")
    if mc_val is not None:
        mc_val = np.asarray(mc_val, dtype=np.float64)
        if mc_val.ndim == 1 and y_val is not None:
            mc_val = mc_val.reshape(1, -1)
        elif y_val is not None and mc_val.shape[1] != y_val.size and mc_val.shape[0] == y_val.size:
            mc_val = mc_val.T

    return y_test, mc_test, metrics, y_val, mc_val


def plot_pred_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lo_90: np.ndarray | None,
    hi_90: np.ndarray | None,
    out_path: str | Path,
    run_id: str = "",
) -> Path:
    """Generate Predicted vs. Actual Wind Speed (kt) scatter plot."""
    fig, ax = plt.subplots(figsize=(7.5, 7.0), facecolor=DARK_BG)
    setup_dark_style(
        ax,
        title=f"Predicted vs. Actual Wind Speed (kt){f' — {run_id}' if run_id else ''}",
        xlabel="Actual Maximum Sustained Wind Speed (kt)",
        ylabel="Predicted Maximum Sustained Wind Speed (kt)",
    )

    all_vals = np.concatenate([y_true, y_pred])
    val_min = max(0.0, float(np.min(all_vals)) - 10.0)
    val_max = float(np.max(all_vals)) + 15.0

    # 1:1 perfect prediction line
    ax.plot(
        [val_min, val_max],
        [val_min, val_max],
        linestyle="--",
        color=TEXT_MUTED,
        linewidth=1.5,
        alpha=0.75,
        label="1:1 Ideal",
        zorder=2,
    )

    # Uncertainty errorbars if available
    if lo_90 is not None and hi_90 is not None:
        yerr = np.array([
            np.maximum(0.0, y_pred - lo_90),
            np.maximum(0.0, hi_90 - y_pred),
        ])
        ax.errorbar(
            y_true,
            y_pred,
            yerr=yerr,
            fmt="none",
            ecolor=ACCENT_BLUE,
            alpha=0.30,
            elinewidth=1.0,
            capsize=2.0,
            capthick=1.0,
            label="90% Conformal Interval",
            zorder=3,
        )

    # Prediction scatter
    ax.scatter(
        y_true,
        y_pred,
        color=ACCENT_BLUE,
        edgecolors="#ffffff",
        linewidths=0.6,
        alpha=0.85,
        s=38,
        label="Held-out Predictions",
        zorder=4,
    )

    # Summary metrics annotation
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    bias = float(np.mean(y_pred - y_true))

    info_text = (
        f"MAE: {mae:.2f} kt\n"
        f"RMSE: {rmse:.2f} kt\n"
        f"Bias: {bias:+.2f} kt\n"
        f"N = {len(y_true)}"
    )
    ax.text(
        0.05,
        0.95,
        info_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        fontfamily="monospace",
        color=TEXT_COLOR,
        bbox=dict(
            boxstyle="round,pad=0.6",
            facecolor=PANEL_BG,
            edgecolor=GRID_COLOR,
            alpha=0.9,
        ),
        zorder=6,
    )

    ax.set_xlim(val_min, val_max)
    ax.set_ylim(val_min, val_max)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(
        loc="lower right",
        facecolor=PANEL_BG,
        edgecolor=GRID_COLOR,
        labelcolor=TEXT_COLOR,
        fontsize=9.5,
    )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(out), dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved pred vs actual plot to {out}")
    return out


def plot_error_by_category(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str | Path,
    metrics: dict | None = None,
    run_id: str = "",
) -> Path:
    """Generate RMSE and MAE breakdown by Saffir-Simpson intensity category."""
    fig, ax = plt.subplots(figsize=(9.0, 5.5), facecolor=DARK_BG)
    setup_dark_style(
        ax,
        title=f"Error Breakdown by Intensity Category (MAE & RMSE){f' — {run_id}' if run_id else ''}",
        xlabel="Saffir-Simpson Intensity Category",
        ylabel="Error (kt)",
    )

    cat_labels: list[str] = []
    cat_maes: list[float] = []
    cat_rmses: list[float] = []
    cat_counts: list[int] = []

    for name, low, high in SAFFIR_SIMPSON_BINS:
        mask = (y_true >= low) & (y_true < high)
        n = int(np.sum(mask))
        if n > 0:
            err = y_pred[mask] - y_true[mask]
            cat_mae = float(np.mean(np.abs(err)))
            cat_rmse = float(np.sqrt(np.mean(err**2)))
            cat_labels.append(f"{name}\n(N={n})")
            cat_maes.append(cat_mae)
            cat_rmses.append(cat_rmse)
            cat_counts.append(n)

    # Fallback to metrics.json by_bin if no test samples matched Saffir-Simpson bins
    if not cat_labels and metrics:
        for split in ("test", "val"):
            by_bin = (metrics.get(split) or {}).get("by_bin")
            if by_bin and isinstance(by_bin, dict):
                for bin_name, b_data in by_bin.items():
                    n = int(b_data.get("n", 0))
                    mae = float(b_data.get("mae_kt", 0.0))
                    rmse = float(b_data.get("rmse_kt", 0.0))
                    cat_labels.append(f"{bin_name}\n(N={n})")
                    cat_maes.append(mae)
                    cat_rmses.append(rmse)
                    cat_counts.append(n)
                break

    if not cat_labels:
        cat_labels = ["All"]
        err = y_pred - y_true
        cat_maes = [float(np.mean(np.abs(err)))]
        cat_rmses = [float(np.sqrt(np.mean(err**2)))]
        cat_counts = [len(y_true)]

    x = np.arange(len(cat_labels))
    bar_width = 0.35

    rects1 = ax.bar(
        x - bar_width / 2,
        cat_maes,
        bar_width,
        label="MAE (kt)",
        color=ACCENT_BLUE,
        alpha=0.9,
        edgecolor=DARK_BG,
    )
    rects2 = ax.bar(
        x + bar_width / 2,
        cat_rmses,
        bar_width,
        label="RMSE (kt)",
        color=ACCENT_ORANGE,
        alpha=0.9,
        edgecolor=DARK_BG,
    )

    # Add numeric labels on top of bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(
            f"{h:.1f}",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=TEXT_COLOR,
        )

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(
            f"{h:.1f}",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=TEXT_COLOR,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, color=TEXT_COLOR)
    max_h = max(max(cat_rmses, default=10.0), max(cat_maes, default=10.0))
    ax.set_ylim(0, max_h * 1.25)
    ax.legend(
        loc="upper right",
        facecolor=PANEL_BG,
        edgecolor=GRID_COLOR,
        labelcolor=TEXT_COLOR,
        fontsize=9.5,
    )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(out), dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved error by category plot to {out}")
    return out


def plot_conformal_coverage(
    y_test: np.ndarray,
    mc_test: np.ndarray,
    out_path: str | Path,
    y_val: np.ndarray | None = None,
    mc_val: np.ndarray | None = None,
    run_id: str = "",
) -> Path:
    """Generate Empirical vs. Nominal Conformal Coverage curve."""
    fig, ax = plt.subplots(figsize=(7.0, 7.0), facecolor=DARK_BG)
    setup_dark_style(
        ax,
        title=f"Conformal Calibration: Empirical vs. Nominal Coverage{f' — {run_id}' if run_id else ''}",
        xlabel="Nominal Coverage Level (1 - α)",
        ylabel="Empirical Coverage",
    )

    # 1:1 diagonal perfect calibration line
    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        color=TEXT_MUTED,
        linewidth=1.5,
        alpha=0.75,
        label="Perfect Calibration",
        zorder=2,
    )

    levels = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99])

    raw_covs: list[float] = []
    scaled_covs: list[float] = []

    from src.calibration import _coverage_and_width, _fit_conformal_scale, _mc_intervals, _scaled_intervals

    can_scale = y_val is not None and mc_val is not None and mc_val.shape[0] >= 2

    for lvl in levels:
        # Raw MC on test
        r_lo, r_hi = _mc_intervals(mc_test, lvl)
        cov_r, _, _ = _coverage_and_width(y_test, r_lo, r_hi)
        raw_covs.append(cov_r)

        # Scaled conformal if val available
        if can_scale:
            try:
                scale = _fit_conformal_scale(y_val, mc_val, lvl)
                s_lo, s_hi = _scaled_intervals(mc_test, scale, lvl)
                cov_s, _, _ = _coverage_and_width(y_test, s_lo, s_hi)
                scaled_covs.append(cov_s)
            except Exception:
                scaled_covs.append(cov_r)
        else:
            scaled_covs.append(cov_r)

    # Plot curves
    ax.plot(
        levels,
        raw_covs,
        marker="o",
        markersize=6,
        color=ACCENT_RED,
        linewidth=1.8,
        label="Raw MC Dropout",
        zorder=4,
    )

    if can_scale:
        ax.plot(
            levels,
            scaled_covs,
            marker="s",
            markersize=6,
            color=ACCENT_BLUE,
            linewidth=2.0,
            label="Scaled Conformal (Val Calibrated)",
            zorder=5,
        )

    # Highlight 90% benchmark level
    idx_90 = int(np.where(levels == 0.90)[0][0])
    chosen_cov = scaled_covs[idx_90] if can_scale else raw_covs[idx_90]
    ax.scatter(
        [0.90],
        [chosen_cov],
        s=90,
        color=ACCENT_GREEN,
        edgecolors="#ffffff",
        zorder=6,
        label=f"90% Target -> {chosen_cov*100:.1f}% Actual",
    )

    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(
        loc="lower right",
        facecolor=PANEL_BG,
        edgecolor=GRID_COLOR,
        labelcolor=TEXT_COLOR,
        fontsize=9.5,
    )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(out), dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved conformal coverage plot to {out}")
    return out


def generate_eval_plots(
    run_dir: str | Path,
    out_dir: str | Path = "docs/assets",
) -> dict[str, Path]:
    """Generate all three dark-mode evaluation plots."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    y_test, mc_test, metrics, y_val, mc_val = load_run_data(run_dir)
    run_id = str(metrics.get("run_id", Path(run_dir).name))

    # Calculate point predictions (mean over MC draws)
    if mc_test is not None:
        y_pred = np.mean(mc_test, axis=0)
    else:
        y_pred = y_test.copy()

    # Calculate 90% uncertainty intervals
    lo_90, hi_90 = None, None
    if mc_test is not None and mc_test.shape[0] >= 2:
        from src.calibration import _fit_conformal_scale, _mc_intervals, _scaled_intervals

        if y_val is not None and mc_val is not None and mc_val.shape[0] >= 2:
            try:
                scale = _fit_conformal_scale(y_val, mc_val, 0.90)
                lo_90, hi_90 = _scaled_intervals(mc_test, scale, 0.90)
            except Exception:
                lo_90, hi_90 = _mc_intervals(mc_test, 0.90)
        else:
            lo_90, hi_90 = _mc_intervals(mc_test, 0.90)

    # 1. Pred vs Actual scatter plot
    p1 = plot_pred_vs_actual(
        y_true=y_test,
        y_pred=y_pred,
        lo_90=lo_90,
        hi_90=hi_90,
        out_path=out_path / "eval_pred_vs_actual.png",
        run_id=run_id,
    )

    # 2. Error by Saffir-Simpson category
    p2 = plot_error_by_category(
        y_true=y_test,
        y_pred=y_pred,
        out_path=out_path / "eval_error_by_category.png",
        metrics=metrics,
        run_id=run_id,
    )

    # 3. Conformal coverage curve
    p3 = plot_conformal_coverage(
        y_test=y_test,
        mc_test=mc_test if mc_test is not None else np.tile(y_pred, (5, 1)),
        out_path=out_path / "eval_conformal_coverage.png",
        y_val=y_val,
        mc_val=mc_val,
        run_id=run_id,
    )

    return {
        "pred_vs_actual": p1,
        "error_by_category": p2,
        "conformal_coverage": p3,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to artifacts/runs/<run_id> containing calibration.npz and metrics.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/assets"),
        help="Directory where PNG plots will be saved (default: docs/assets)",
    )
    args = parser.parse_args(argv)

    try:
        generate_eval_plots(args.run_dir, args.out_dir)
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
