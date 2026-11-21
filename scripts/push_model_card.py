"""Push a populated Model Card (README.md) to Hugging Face Hub.

Reads docs/HF_MODEL_CARD_TEMPLATE.md and a completed run's metrics.json,
strictly verifies real evaluation values, prepends valid Hugging Face YAML
frontmatter, and uploads to Anlea/cloudie-weights (or outputs locally with --dry-run).

Usage::

    python scripts/push_model_card.py \\
        --run-dir artifacts/runs/<run_id> \\
        --dry-run \\
        --out artifacts/runs/<run_id>/README.md
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path


DEFAULT_REPO_ID = "Anlea/cloudie-weights"
DEFAULT_TEMPLATE = "docs/HF_MODEL_CARD_TEMPLATE.md"


def validate_metrics(metrics: dict) -> None:
    """Validate that metrics.json contains required real metrics and zero synthetic numbers."""
    if not isinstance(metrics, dict):
        raise ValueError("metrics.json must be a JSON object")

    is_synth = metrics.get("is_synthetic", True)
    if isinstance(is_synth, str):
        is_synth = is_synth.lower() in ("true", "1", "yes")
    if is_synth:
        raise ValueError(
            "Refusing to push model card: run is synthetic/smoke. "
            "Only real, reviewed training runs can be published."
        )

    quote = str(metrics.get("quote_as_result", "no")).strip().lower()
    if quote != "yes":
        raise ValueError(
            f"Refusing to push model card: quote_as_result is {quote!r}, not 'yes'. "
            "Set quote_as_result='yes' in metrics.json only after the run has been "
            "thoroughly audited and approved."
        )

    required_top = ["run_id", "model", "dataset"]
    for key in required_top:
        val = metrics.get(key)
        if not val or not isinstance(val, str) or val.strip() in ("", "<run_id>"):
            raise ValueError(f"metrics.json missing required top-level key: {key}")

    # Must contain at least one evaluation split with valid metrics
    eval_found = False
    for split in ("test", "val"):
        block = metrics.get(split)
        if isinstance(block, dict) and block.get("scored", True):
            mae = block.get("mae_kt")
            rmse = block.get("rmse_kt")
            if mae is not None and rmse is not None:
                try:
                    f_mae = float(mae)
                    f_rmse = float(rmse)
                    if math.isfinite(f_mae) and math.isfinite(f_rmse):
                        eval_found = True
                        break
                except (ValueError, TypeError):
                    pass

    if not eval_found:
        raise ValueError("metrics.json must contain at least one scored split (test or val) with valid mae_kt and rmse_kt")


def format_frontmatter(
    license_type: str = "mit",
    library_name: str = "pytorch",
    tags: list[str] | None = None,
) -> str:
    """Generate Hugging Face Model Card YAML frontmatter."""
    tag_list = tags or [
        "tropical-cyclone",
        "remote-sensing",
        "pytorch",
        "conformal-prediction",
    ]
    tag_lines = "\n".join(f"- {t}" for t in tag_list)
    return (
        "---\n"
        f"license: {license_type}\n"
        f"library_name: {library_name}\n"
        "tags:\n"
        f"{tag_lines}\n"
        "language:\n"
        "- en\n"
        "datasets:\n"
        "- incyde\n"
        "- tcir\n"
        "pipeline_tag: image-to-text\n"
        "---\n\n"
    )


def render_model_card_content(
    template_text: str,
    metrics: dict,
) -> str:
    """Populate template strictly with real values from metrics.json."""
    validate_metrics(metrics)

    run_id = str(metrics["run_id"])
    model_name = str(metrics["model"])
    dataset_name = str(metrics["dataset"])

    # Strip existing frontmatter from template if present
    content = template_text
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            content = parts[2].lstrip()

    # Prepend normalized valid HF YAML frontmatter
    frontmatter = format_frontmatter()
    content = frontmatter + content

    # Replace <run_id> placeholders with real run_id
    content = content.replace("<run_id>", run_id)

    # Append verified benchmark metrics table
    from scripts.finalize_run import build_benchmark_table_markdown

    table_md = build_benchmark_table_markdown(metrics)
    content = (
        content.rstrip()
        + f"\n\n---\n\n## Verified Model Performance (`{run_id}`)\n\n"
        f"**Dataset:** `{dataset_name.upper()}` | **Architecture:** `{model_name}`\n\n"
        f"{table_md}\n"
    )

    return content


def push_model_card(
    run_dir: str | Path,
    template_path: str | Path = DEFAULT_TEMPLATE,
    out_path: str | Path | None = None,
    repo_id: str = DEFAULT_REPO_ID,
    dry_run: bool = False,
    commit_message: str | None = None,
) -> Path:
    """Read metrics, render model card, and push to HF or write locally."""
    run_path = Path(run_dir)
    metrics_file = run_path / "metrics.json"
    if not metrics_file.is_file():
        raise FileNotFoundError(f"metrics.json not found in {run_path}")

    with open(metrics_file, encoding="utf-8") as f:
        metrics = json.load(f)

    template_file = Path(template_path)
    if not template_file.is_file():
        raise FileNotFoundError(f"Template file not found at {template_file}")

    template_text = template_file.read_text(encoding="utf-8")
    rendered = render_model_card_content(template_text, metrics)

    target_out = Path(out_path) if out_path else (run_path / "README.md")
    target_out.parent.mkdir(parents=True, exist_ok=True)
    target_out.write_text(rendered, encoding="utf-8")
    print(f"Rendered model card saved to {target_out}")

    if dry_run:
        print(f"Dry-run enabled: skipped Hub upload to {repo_id}")
        return target_out

    # Upload to Hugging Face Hub
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to upload model card. Install with `pip install huggingface_hub`."
        ) from exc

    msg = commit_message or f"docs: update model card for run {metrics['run_id']}"
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(target_out),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message=msg,
    )
    print(f"Successfully uploaded model card to https://huggingface.co/{repo_id}")
    return target_out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to artifacts/runs/<run_id> directory containing metrics.json",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path(DEFAULT_TEMPLATE),
        help=f"Path to model card template (default: {DEFAULT_TEMPLATE})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Path to write rendered README.md locally",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help=f"Target Hugging Face model repository (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render model card locally without uploading to Hugging Face",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default=None,
        help="Commit message for Hugging Face upload",
    )
    args = parser.parse_args(argv)

    try:
        push_model_card(
            run_dir=args.run_dir,
            template_path=args.template,
            out_path=args.out,
            repo_id=args.repo_id,
            dry_run=args.dry_run,
            commit_message=args.commit_message,
        )
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
