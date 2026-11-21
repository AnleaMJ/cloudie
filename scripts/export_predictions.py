"""Standalone CLI to generate predictions.npz and evaluate residual quantiles.

Usage::

    python scripts/export_predictions.py --run-dir artifacts/runs/<run_id>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate import _LEVELS, _TOLERANCE, evaluate_run_directory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate predictions.npz and evaluate held-out residual quantiles."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to an existing run directory",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=_TOLERANCE,
        help="Coverage tolerance for calibration",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        type=float,
        default=list(_LEVELS),
        help="Nominal coverage levels",
    )
    args = parser.parse_args(argv)
    evaluate_run_directory(
        args.run_dir,
        tolerance=args.tolerance,
        levels=args.levels,
    )
    print(f"Successfully processed {args.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
