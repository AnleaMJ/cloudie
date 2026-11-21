"""Finalize a reviewed training run.

One-command pipeline that accepts --run-dir (pointing to a real
artifacts/runs/<run_id> directory), strictly refuses synthetic/smoke runs,
and executes:
1. scripts/plot_calibration.py (if calibration predictions .npz exists)
2. scripts/render_readme_tables.py to update README.md between
   <!-- BENCHMARK:START --> and <!-- BENCHMARK:END -->
3. Renders a populated MODEL_CARD.md from docs/HF_MODEL_CARD_TEMPLATE.md
   using the run's metrics.json for Anlea/cloudie-weights.

Usage::

    python scripts/finalize_run.py --run-dir artifacts/runs/<run_id>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _refuse_smoke(metrics: dict) -> None:
    """Strictly refuse synthetic or unapproved smoke data."""
    is_synthetic = metrics.get("is_synthetic", True)
    if isinstance(is_synthetic, str):
        is_synthetic = is_synthetic.lower() in ("true", "1", "yes")
    if is_synthetic:
        raise ValueError(
            "Refusing to finalize: run is synthetic/smoke. "
            "Only real, reviewed training runs can be published."
        )

    quote = str(metrics.get("quote_as_result", "no")).strip().lower()
    if quote != "yes":
        raise ValueError(
            f"Refusing to finalize: quote_as_result is {quote!r}, not 'yes'. "
            "Set quote_as_result='yes' in metrics.json only after the run has been "
            "thoroughly audited and approved."
        )


def _fmt(value: object, decimals: int = 2) -> str:
    """Format numeric values safely."""
    if value is None:
        return "—"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not __import__("math").isfinite(f):
        return "—"
    return f"{f:.{decimals}f}"


def build_benchmark_table_markdown(metrics: dict) -> str:
    """Build markdown tables for the benchmark section."""
    lines: list[str] = []
    run_id = metrics.get("run_id", "reviewed-run")
    model = metrics.get("model", "cloudie")
    dataset = metrics.get("dataset", "cyclone")

    lines.append(f"> **Verified Run Results:** `{run_id}` ({dataset.upper()} | Model: `{model}`)\n")
    lines.append("> All metrics use the held-out storm protocol with zero data leakage. "
                 "Bins are this project's own grouping (17/34/48/64/91 kt), not official IMD classes.\n")

    # Overall regression metrics
    for split_name in ("val", "test"):
        block = metrics.get(split_name) or metrics.get(f"{split_name}_raw")
        if not block or not isinstance(block, dict) or not block.get("scored", True):
            continue
        lines.append(f"### {split_name.capitalize()} Split Performance\n")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        for key in ("mae_kt", "rmse_kt", "bias_kt"):
            label = key.replace("_kt", " (kt)").upper().replace("_", " ")
            lines.append(f"| {label} | {_fmt(block.get(key))} |")
        n = block.get("n")
        if n is not None:
            lines.append(f"| N (frames/windows) | {int(float(n))} |")
        lines.append("")

    # Per-class metrics
    for split_name in ("val", "test"):
        block = metrics.get(split_name)
        if not block or not isinstance(block, dict):
            continue
        by_bin = block.get("by_bin")
        if not by_bin:
            continue
        lines.append(f"### {split_name.capitalize()} Per-Class Breakdown\n")
        lines.append("| Class | N | MAE (kt) | RMSE (kt) | Bias (kt) |")
        lines.append("|---|---|---|---|---|")
        for cls_name, cls_metrics in by_bin.items():
            n = _fmt(cls_metrics.get("n"), 0) if cls_metrics.get("n") is not None else "0"
            lines.append(
                f"| `{cls_name}` | {n} | "
                f"{_fmt(cls_metrics.get('mae_kt'))} | "
                f"{_fmt(cls_metrics.get('rmse_kt'))} | "
                f"{_fmt(cls_metrics.get('bias_kt'))} |"
            )
        lines.append("")

    # Calibration summary if present
    cal = metrics.get("calibration")
    if cal and isinstance(cal, dict):
        lines.append("### Uncertainty Calibration\n")
        lines.append("| Metric | Value | Interpretation |")
        lines.append("|---|---|---|")
        lines.append(f"| MACE | {_fmt(cal.get('mace'), 4)} | Mean Absolute Calibration Error |")
        lines.append(f"| ENCE | {_fmt(cal.get('ence'), 4)} | Expected Normalized Calibration Error |")
        lines.append(f"| Spearman ρ | {_fmt(cal.get('spearman_rho'), 3)} | Uncertainty vs Error Rank Corr |")
        lines.append(f"| 90% Raw MC Coverage | {_fmt(cal.get('coverage_90_raw'), 3)} | Interval Width: {_fmt(cal.get('width_90_raw_kt'), 1)} kt |")
        lines.append(f"| 90% Scaled Conformal Coverage | {_fmt(cal.get('coverage_90_scaled'), 3)} | Interval Width: {_fmt(cal.get('width_90_scaled_kt'), 1)} kt |")
        lines.append("")

    return "\n".join(lines).strip()


def update_readme_benchmark(metrics: dict, readme_path: str | Path) -> Path:
    """Update README.md between <!-- BENCHMARK:START --> and <!-- BENCHMARK:END -->."""
    path = Path(readme_path)
    if not path.is_file():
        raise FileNotFoundError(f"README file not found at {path}")

    text = path.read_text(encoding="utf-8")
    start_tag = "<!-- BENCHMARK:START -->"
    end_tag = "<!-- BENCHMARK:END -->"

    if start_tag not in text or end_tag not in text:
        raise ValueError(f"Could not find {start_tag} and {end_tag} in {path}")

    table_md = build_benchmark_table_markdown(metrics)
    replacement = f"{start_tag}\n## Benchmark & Baselines\n\n{table_md}\n{end_tag}"

    pattern = re.compile(
        rf"{re.escape(start_tag)}.*?{re.escape(end_tag)}",
        re.DOTALL,
    )
    new_text = pattern.sub(replacement, text)
    path.write_text(new_text, encoding="utf-8")
    print(f"Updated benchmark section in {path}")
    return path


def render_model_card(
    metrics: dict,
    template_path: str | Path,
    out_path: str | Path,
) -> Path:
    """Populate docs/HF_MODEL_CARD_TEMPLATE.md for Anlea/cloudie-weights."""
    template_file = Path(template_path)
    if not template_file.is_file():
        raise FileNotFoundError(f"Template not found at {template_file}")

    content = template_file.read_text(encoding="utf-8")
    run_id = str(metrics.get("run_id", "run-001"))

    # Replace <run_id> with the concrete run id
    populated = content.replace("<run_id>", run_id)

    # Append run performance section to model card
    summary_md = build_benchmark_table_markdown(metrics)
    populated += f"\n\n---\n\n## Verified Metrics for `{run_id}`\n\n{summary_md}\n"

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(populated, encoding="utf-8")
    print(f"Rendered populated model card to {out}")
    return out


def finalize_run(
    run_dir: str | Path,
    readme_path: str | Path = "README.md",
    template_path: str | Path = "docs/HF_MODEL_CARD_TEMPLATE.md",
) -> dict:
    """Execute the full post-training publishing pipeline."""
    run_path = Path(run_dir)
    if not run_path.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_path}")

    metrics_file = run_path / "metrics.json"
    if not metrics_file.is_file():
        raise FileNotFoundError(f"metrics.json missing in {run_path}")

    with open(metrics_file, encoding="utf-8") as f:
        metrics = json.load(f)

    # 1. Gate check: strictly refuse synthetic / unapproved runs
    _refuse_smoke(metrics)

    results = {"run_dir": str(run_path), "run_id": metrics.get("run_id")}

    # 2. Plot calibration if calibration predictions exist
    cal_npz = run_path / "calibration.npz"
    if cal_npz.is_file():
        try:
            from scripts.plot_calibration import plot as plot_calibration

            plot_out = run_path / "calibration_plot.png"
            plot_calibration(cal_npz, plot_out)
            results["calibration_plot"] = str(plot_out)
        except Exception as exc:
            print(f"Warning: Calibration plotting skipped: {exc}")
    else:
        print(f"Notice: No calibration.npz in {run_path}; skipping calibration plot.")

    # 3. Update README.md benchmark section
    updated_readme = update_readme_benchmark(metrics, readme_path)
    results["readme"] = str(updated_readme)

    # 4. Render MODEL_CARD.md
    model_card_out = run_path / "MODEL_CARD.md"
    rendered_card = render_model_card(metrics, template_path, model_card_out)
    results["model_card"] = str(rendered_card)

    print(f"Successfully finalized run {metrics.get('run_id')}!")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to artifacts/runs/<run_id> directory",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=Path("README.md"),
        help="Path to README.md to update",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("docs/HF_MODEL_CARD_TEMPLATE.md"),
        help="Path to Hugging Face Model Card template",
    )
    args = parser.parse_args(argv)

    try:
        finalize_run(args.run_dir, readme_path=args.readme, template_path=args.template)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
