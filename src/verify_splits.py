"""CLI that runs a 6-point data-leakage audit on the configured splits.

The six checks are:

1. storm_wise splits keep storms whole and disjoint.
2. chronological splits keep storms whole and disjoint.
3. holdout_basin splits keep storms whole and disjoint.
4. Random / frame / shuffle strategies raise ``LeakageError``.
5. No forbidden feature name leaks into the model matrix.
6. Cross-split previous-frame links are nulled.

Usage::

    python -m src.verify_splits --config config/config_smoke.yaml

The exit code is 0 when all six checks pass.  This module does not import
torch and does not download imagery.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.make_smoke_data import write_smoke
from src.schema import require_valid
from src.splits import (
    LeakageError,
    add_previous_frame_id,
    assert_no_forbidden_inputs,
    assign_storms,
    null_cross_split_links,
    storm_ids_by_split,
)


def _build_smoke_frame(config: dict) -> pd.DataFrame:
    """Return a validated manifest from the smoke config's data path."""
    manifest_path = Path(config["paths"]["manifest"])
    if not manifest_path.exists():
        # Generate smoke data on the fly so the audit is self-contained.
        out_dir = manifest_path.parent
        write_smoke(out_dir, seed=config.get("project", {}).get("seed", 7))
    frame = pd.read_csv(manifest_path)
    return require_valid(frame)


def _assert_disjoint(assigned: pd.DataFrame) -> dict[str, frozenset[str]]:
    """Verify that storm-id sets across splits are disjoint."""
    groups = storm_ids_by_split(assigned)
    assert groups["train"], "train is empty"
    assert groups["val"], "val is empty"
    assert groups["test"], "test is empty"
    assert groups["train"].isdisjoint(groups["val"]), "train ∩ val is non-empty"
    assert groups["train"].isdisjoint(groups["test"]), "train ∩ test is non-empty"
    assert groups["val"].isdisjoint(groups["test"]), "val ∩ test is non-empty"
    # Every storm is in exactly one split.
    counts = assigned.groupby(assigned["storm_id"].astype(str))["split"].nunique()
    assert int(counts.max()) == 1, "a storm appears in more than one split"
    return groups


def _run_strategy(frame: pd.DataFrame, strategy: str, config: dict) -> None:
    """Check that a strategy assigns splits correctly."""
    split_cfg = dict(config.get("split", {}))
    kwargs: dict = {
        "test_fraction": float(split_cfg.get("test_fraction", 0.2)),
        "val_fraction": float(split_cfg.get("val_fraction", 0.2)),
        "seed": int(split_cfg.get("seed", 42)),
    }
    if strategy == "chronological":
        holdout_years = [int(y) for y in (split_cfg.get("holdout_years") or [])]
        if not holdout_years:
            # Use at least one year for the audit.
            if "year" in frame.columns:
                years = sorted(frame["year"].dropna().unique())
                holdout_years = [int(years[-1])] if years else [2021]
            else:
                holdout_years = [2021]
        kwargs["holdout_years"] = holdout_years
    if strategy == "holdout_basin":
        holdout_basin = split_cfg.get("holdout_basin") or None
        if not holdout_basin:
            if "basin" in frame.columns:
                basins = sorted(frame["basin"].dropna().unique())
                holdout_basin = basins[0] if basins else "NIO"
            else:
                holdout_basin = "NIO"
        kwargs["holdout_basin"] = holdout_basin

    assigned = assign_storms(frame, strategy, **kwargs)
    # Deterministic: same call same result.
    again = assign_storms(frame, strategy, **kwargs)
    assert assigned["split"].tolist() == again["split"].tolist(), (
        f"{strategy}: non-deterministic assignment"
    )
    _assert_disjoint(assigned)


def audit(config: dict) -> list[str]:
    """Run the 6-point leakage audit.  Returns a list of failure messages."""
    frame = _build_smoke_frame(config)
    failures: list[str] = []

    # --- Check 1: storm_wise ---
    try:
        _run_strategy(frame, "storm_wise", config)
        print("  [PASS] 1. storm_wise splits are disjoint")
    except Exception as exc:
        failures.append(f"1. storm_wise: {exc}")
        print(f"  [FAIL] 1. storm_wise: {exc}")

    # --- Check 2: chronological ---
    try:
        _run_strategy(frame, "chronological", config)
        print("  [PASS] 2. chronological splits are disjoint")
    except Exception as exc:
        failures.append(f"2. chronological: {exc}")
        print(f"  [FAIL] 2. chronological: {exc}")

    # --- Check 3: holdout_basin ---
    try:
        _run_strategy(frame, "holdout_basin", config)
        print("  [PASS] 3. holdout_basin splits are disjoint")
    except Exception as exc:
        failures.append(f"3. holdout_basin: {exc}")
        print(f"  [FAIL] 3. holdout_basin: {exc}")

    # --- Check 4: random splits raise LeakageError ---
    random_ok = True
    for name in ("random", "frame", "shuffle", "random_frame"):
        try:
            assign_storms(frame, name)
            failures.append(f"4. {name} did not raise LeakageError")
            random_ok = False
        except LeakageError:
            pass
        except Exception as exc:
            failures.append(f"4. {name} raised unexpected {type(exc).__name__}: {exc}")
            random_ok = False
    if random_ok:
        print("  [PASS] 4. Random splits raise LeakageError")
    else:
        print("  [FAIL] 4. Random splits did not all raise LeakageError")

    # --- Check 5: no forbidden features in model matrix ---
    try:
        from src.features import FeatureSpec

        feature_names = list(FeatureSpec().feature_names())
        assert_no_forbidden_inputs(feature_names, "model features")
        print("  [PASS] 5. No forbidden features in default feature set")
    except LeakageError as exc:
        failures.append(f"5. forbidden features: {exc}")
        print(f"  [FAIL] 5. {exc}")
    except Exception as exc:
        failures.append(f"5. feature check error: {exc}")
        print(f"  [FAIL] 5. {exc}")

    # --- Check 6: cross-split links are nulled ---
    try:
        split_cfg = dict(config.get("split", {}))
        assigned = assign_storms(
            frame,
            "storm_wise",
            test_fraction=float(split_cfg.get("test_fraction", 0.2)),
            val_fraction=float(split_cfg.get("val_fraction", 0.2)),
            seed=int(split_cfg.get("seed", 42)),
        )
        linked = add_previous_frame_id(assigned)
        linked["split"] = assigned.loc[linked.index, "split"]
        # Artificially set a cross-split link.
        train_ids = linked.loc[linked["split"] == "train", "sample_id"].tolist()
        test_ids = linked.loc[linked["split"] == "test", "sample_id"].tolist()
        if train_ids and test_ids:
            dirty = linked.copy()
            dirty.loc[dirty["sample_id"] == train_ids[0], "prev_sample_id"] = test_ids[0]
            cleaned, dropped = null_cross_split_links(dirty)
            assert dropped >= 1, "cross-split link was not removed"
            assert pd.isna(
                cleaned.loc[cleaned["sample_id"] == train_ids[0], "prev_sample_id"]
            ).all(), "cross-split link survived cleaning"
        print("  [PASS] 6. Cross-split previous-frame links are nulled")
    except Exception as exc:
        failures.append(f"6. cross-split links: {exc}")
        print(f"  [FAIL] 6. {exc}")

    return failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a 6-point data-leakage audit on the split strategy."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config_smoke.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    print(f"Running 6-point leakage audit with {args.config}")
    failures = audit(config)
    if failures:
        print(f"\n{len(failures)} check(s) failed:")
        for msg in failures:
            print(f"  - {msg}")
        sys.exit(1)
    else:
        print("\nAll 6 checks passed.")


if __name__ == "__main__":
    main()
