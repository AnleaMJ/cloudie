"""Plot calibration curves: raw MC Dropout vs scaled conformal intervals.

This script MUST assert ``quote_as_result == "yes"`` and refuses to run on
smoke data.  It reads an ``.npz`` file produced by a real training run and
generates a calibration reliability diagram.

Usage::

    python scripts/plot_calibration.py \\
        --npz artifacts/runs/<run_id>/calibration.npz \\
        --out reports/calibration_plot.png

The .npz file must contain:

- ``y_true_val``, ``mc_samples_val``: validation ground truth and MC draws.
- ``y_true_test``, ``mc_samples_test``: test ground truth and MC draws.
- ``is_synthetic``: boolean flag; must be False for real data.
- ``quote_as_result``: string; must be "yes".
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np


def _refuse_smoke(data: dict) -> None:
    """Hard gate: refuse to plot if the data is synthetic or not approved."""
    quote = str(data.get("quote_as_result", "no"))
    assert quote == "yes", (
        f"quote_as_result is {quote!r}, not 'yes'. "
        "This script refuses to generate plots from smoke or unapproved data. "
        "Set quote_as_result='yes' in the .npz only after a real training run "
        "has been reviewed and approved."
    )

    is_synthetic = data.get("is_synthetic", True)
    if isinstance(is_synthetic, np.ndarray):
        is_synthetic = bool(is_synthetic.item())
    assert not is_synthetic, (
        "is_synthetic is True. This script refuses to plot synthetic/smoke data."
    )


def plot(npz_path: str | Path, out_path: str | Path) -> Path:
    """Generate a calibration reliability diagram."""
    import matplotlib.pyplot as plt

    from src.calibration import NOMINAL_LEVELS, compare_calibration

    data = dict(np.load(str(npz_path), allow_pickle=True))
    _refuse_smoke(data)

    y_val = data["y_true_val"]
    mc_val = data["mc_samples_val"]
    y_test = data["y_true_test"]
    mc_test = data["mc_samples_test"]

    report = compare_calibration(y_val, mc_val, y_test, mc_test, levels=NOMINAL_LEVELS)

    # --- Reliability diagram ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: coverage vs nominal
    ax = axes[0]
    nominals = [report.raw_mc[f"{l:.2f}"].nominal for l in NOMINAL_LEVELS]
    raw_covs = [report.raw_mc[f"{l:.2f}"].empirical_coverage for l in NOMINAL_LEVELS]
    sc_covs = [report.scaled_conformal[f"{l:.2f}"].empirical_coverage for l in NOMINAL_LEVELS]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.plot(nominals, raw_covs, "o-", color="tab:red", label="Raw MC Dropout")
    ax.plot(nominals, sc_covs, "s-", color="tab:blue", label="Scaled conformal")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Calibration reliability diagram")
    ax.legend(loc="lower right")
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0.4, 1.0)
    ax.set_aspect("equal")

    # Right: interval widths
    ax2 = axes[1]
    raw_widths = [report.raw_mc[f"{l:.2f}"].mean_width_kt for l in NOMINAL_LEVELS]
    sc_widths = [report.scaled_conformal[f"{l:.2f}"].mean_width_kt for l in NOMINAL_LEVELS]
    x_pos = np.arange(len(NOMINAL_LEVELS))
    width = 0.35
    ax2.bar(x_pos - width / 2, raw_widths, width, label="Raw MC", color="tab:red", alpha=0.7)
    ax2.bar(x_pos + width / 2, sc_widths, width, label="Scaled conformal", color="tab:blue", alpha=0.7)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"{l:.0%}" for l in NOMINAL_LEVELS])
    ax2.set_xlabel("Nominal level")
    ax2.set_ylabel("Mean interval width (kt)")
    ax2.set_title("Interval widths")
    ax2.legend()

    # Annotation box
    text = (
        f"MACE = {report.mace:.4f}\n"
        f"ENCE = {report.ence:.4f}\n"
        f"Spearman ρ = {report.spearman_rho:.3f} (p = {report.spearman_p:.2e})\n"
        f"90% raw coverage = {report.coverage_90_raw:.3f}, width = {report.width_90_raw_kt:.1f} kt\n"
        f"90% scaled coverage = {report.coverage_90_scaled:.3f}, width = {report.width_90_scaled_kt:.1f} kt"
    )
    fig.text(0.5, -0.02, text, ha="center", fontsize=9, family="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved calibration plot to {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=str, required=True, help="Path to calibration .npz")
    parser.add_argument("--out", type=str, default="reports/calibration_plot.png",
                        help="Output path for the plot")
    args = parser.parse_args()
    plot(args.npz, args.out)


if __name__ == "__main__":
    main()
