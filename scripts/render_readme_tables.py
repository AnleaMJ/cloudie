"""Render markdown result tables for README.md from a metrics.json file.

This script MUST assert ``quote_as_result == "yes"`` and refuses to run on
smoke data.  It reads ``metrics.json`` from a real training run and writes
a markdown snippet with regression metrics and calibration summary.

Usage::

    python scripts/render_readme_tables.py \\
        --metrics artifacts/runs/<run_id>/metrics.json \\
        --out docs/readme_tables.md

The metrics.json must contain:

- ``quote_as_result``: must be "yes".
- ``is_synthetic``: must be false.
- ``val`` or ``test`` blocks with regression metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _refuse_smoke(metrics: dict) -> None:
    """Hard gate: refuse to render tables from smoke or unapproved data."""
    quote = str(metrics.get("quote_as_result", "no"))
    assert quote == "yes", (
        f"quote_as_result is {quote!r}, not 'yes'. "
        "This script refuses to render tables from smoke or unapproved data. "
        "Set quote_as_result to 'yes' in metrics.json only after a real "
        "training run has been reviewed and approved."
    )

    is_synthetic = metrics.get("is_synthetic", True)
    if isinstance(is_synthetic, str):
        is_synthetic = is_synthetic.lower() in ("true", "1", "yes")
    assert not is_synthetic, (
        "is_synthetic is True. This script refuses to render tables "
        "from synthetic/smoke data."
    )


def _fmt(value: object, decimals: int = 2) -> str:
    """Format a numeric value for the table."""
    if value is None:
        return "—"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not __import__("math").isfinite(f):
        return "—"
    return f"{f:.{decimals}f}"


def render(metrics_path: str | Path, out_path: str | Path) -> Path:
    """Read metrics.json and write a markdown results table."""
    with open(metrics_path, encoding="utf-8") as fh:
        metrics = json.load(fh)

    _refuse_smoke(metrics)

    lines: list[str] = []
    lines.append("## Results\n")
    lines.append("> These numbers are from a reviewed training run. "
                 "Bins are this project's own grouping (17/34/48/64/91 kt), "
                 "not official IMD classes.\n")

    # --- Overall regression metrics ---
    for split_name in ("val", "test"):
        block = metrics.get(split_name) or metrics.get(f"{split_name}_raw")
        if not block:
            continue
        lines.append(f"\n### {split_name.capitalize()} split\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for key in ("mae_kt", "rmse_kt", "bias_kt"):
            label = key.replace("_kt", " (kt)").upper().replace("_", " ")
            lines.append(f"| {label} | {_fmt(block.get(key))} |")
        n = block.get("n")
        if n is not None:
            lines.append(f"| N | {int(float(n))} |")

    # --- Per-class metrics ---
    for split_name in ("val", "test"):
        block = metrics.get(split_name)
        if not block:
            continue
        by_bin = block.get("by_bin")
        if not by_bin:
            continue
        lines.append(f"\n### {split_name.capitalize()} per-class\n")
        lines.append("| Class | N | MAE (kt) | RMSE (kt) | Bias (kt) |")
        lines.append("|-------|---|----------|-----------|-----------|")
        for cls_name, cls_metrics in by_bin.items():
            n = _fmt(cls_metrics.get("n"), 0) if cls_metrics.get("n") is not None else "0"
            lines.append(
                f"| {cls_name} | {n} | "
                f"{_fmt(cls_metrics.get('mae_kt'))} | "
                f"{_fmt(cls_metrics.get('rmse_kt'))} | "
                f"{_fmt(cls_metrics.get('bias_kt'))} |"
            )

    # --- Calibration summary ---
    cal = metrics.get("calibration")
    if cal:
        lines.append("\n### Calibration\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for key in ("mace", "ence", "spearman_rho",
                     "coverage_90_raw", "width_90_raw_kt",
                     "coverage_90_scaled", "width_90_scaled_kt"):
            label = key.replace("_", " ").title()
            lines.append(f"| {label} | {_fmt(cal.get(key), 4)} |")

    # --- MC dropout coverage ---
    mc = metrics.get("mc_dropout_coverage")
    if mc:
        lines.append("\n### MC dropout coverage\n")
        lines.append("| Level | Empirical | Width (kt) | Status |")
        lines.append("|-------|-----------|------------|--------|")
        for key, entry in mc.items():
            lines.append(
                f"| {_fmt(entry.get('nominal'), 2)} | "
                f"{_fmt(entry.get('empirical_coverage'), 3)} | "
                f"{_fmt(entry.get('mean_interval_width_kt'), 1)} | "
                f"{entry.get('status', '')} |"
            )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote README tables to {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=str, required=True,
                        help="Path to metrics.json from a training run")
    parser.add_argument("--out", type=str, default="docs/readme_tables.md",
                        help="Output markdown file path")
    args = parser.parse_args()
    render(args.metrics, args.out)


if __name__ == "__main__":
    main()
