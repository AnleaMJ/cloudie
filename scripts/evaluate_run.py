"""Evaluate an existing training run directory and compute held-out residual quantiles.

Computes held-out residual-quantile intervals (fit on val storms, evaluated on test storms)
and merges the resulting block into the run's metrics.json, matching the schema required
by ConfidenceScore._held_out_storm_check.

Usage::

    python scripts/evaluate_run.py --run-dir artifacts/runs/<run_id>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate import _LEVELS, _TOLERANCE, evaluate_run_directory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate an existing training run directory and compute held-out residual quantiles."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to an existing run directory (e.g. artifacts/runs/<run_id>)",
    )
    parser.add_argument(
        "--coverage-tolerance",
        "--tolerance",
        dest="tolerance",
        type=float,
        default=_TOLERANCE,
        help=f"Coverage tolerance for calibration status (default: {_TOLERANCE})",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        type=float,
        default=list(_LEVELS),
        help=f"Nominal coverage levels (default: {list(_LEVELS)})",
    )
    args = parser.parse_args(argv)
    evaluate_run_directory(
        args.run_dir,
        tolerance=args.tolerance,
        levels=args.levels,
    )
    print(f"Updated metrics.json with held-out residual quantiles in {args.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
