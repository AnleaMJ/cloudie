"""Intensity-bin weights from train originals. No downloads and no torch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.incyde import assign_incyde_split
from src.train import (
    _score_split,
    align_bin_weights,
    annotate_balance,
    frequency_rows,
    intensity_balance,
    is_on_fix,
    loss_weighting_mode,
    on_fix_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["D_DD", "CS", "SCS", "VSCS", "ESCS_SUCS"]
EDGES = [17, 34, 48, 64, 91]


def _config(mode: str = "inverse_freq", balanced: bool = False, strategy: str = "storm_wise") -> dict:
    return {
        "data": {"imd5_edges_kt": EDGES, "imd5_names": NAMES, "wind_averaging": "3min"},
        "train": {"loss_weighting": mode, "balanced_sampling": balanced},
        "split": {"strategy": strategy},
    }


def _row(storm: str, split: str, wind: float, synthetic: bool = False) -> dict:
    return {
        "storm_id": storm,
        "split": split,
        "is_synthetic": synthetic,
        "wind_kt_3min": wind,
        "wind_speed_kt": wind,
        "wind_averaging": "3min",
    }


def _skewed_frame() -> pd.DataFrame:
    rows = []
    rows.extend(_row("d", "train", 20.0) for _ in range(4))
    rows.append(_row("d", "train", 20.0, synthetic=True))
    rows.extend(_row("c", "train", 40.0) for _ in range(2))
    rows.append(_row("s", "train", 55.0))
    rows.append(_row("v", "train", 70.0))
    rows.append(_row("e", "train", 100.0))
    rows.extend(_row("e", "train", 100.0, synthetic=True) for _ in range(10))
    rows.extend(_row("val-storm", "val", 20.0) for _ in range(50))
    rows.extend(_row("test-storm", "test", 100.0) for _ in range(50))
    rows.append(_row("low", "train", 10.0))
    return pd.DataFrame(rows)


def test_default_switches_match_the_unweighted_loader() -> None:
    for name in ("config.yaml", "config_smoke.yaml"):
        config = load_config(ROOT / "config" / name)
        assert config["train"]["loss_weighting"] == "none"
        assert config["train"]["balanced_sampling"] is False
        assert config["incyde"]["on_fix_only_eval"] is False
        assert config["labels"]["near_land_policy"] == "flag"
        assert float(config["labels"]["open_water_km"]) == 20.0
        assert loss_weighting_mode(config["train"]["loss_weighting"]) == "none"


def test_weights_use_train_originals_only() -> None:
    frame = _skewed_frame()
    balance = intensity_balance(frame, _config("inverse_freq"), "wind_kt_3min")
    assert balance["counts"] == {
        "D_DD": 4,
        "CS": 2,
        "SCS": 1,
        "VSCS": 1,
        "ESCS_SUCS": 1,
    }
    assert balance["n_frequency_rows"] == 10  # 9 binned originals plus the 10 kt row
    assert balance["n_below_lowest_edge"] == 1
    assert balance["weights"]["D_DD"] == pytest.approx(9 / (5 * 4))
    assert balance["weights"]["CS"] == pytest.approx(9 / (5 * 2))
    assert balance["weights"]["ESCS_SUCS"] == pytest.approx(9 / 5)
    counted, _note = frequency_rows(frame)
    assert set(counted["split"]) == {"train"}
    assert not counted["is_synthetic"].astype(bool).any()
    assert "val-storm" not in set(counted["storm_id"])
    assert "test-storm" not in set(counted["storm_id"])

    sqrt_balance = intensity_balance(frame, _config("sqrt_inverse_freq"), "wind_kt_3min")
    assert sqrt_balance["weights"]["D_DD"] == pytest.approx((9 / 20) ** 0.5)
    none = intensity_balance(frame, _config("none"), "wind_kt_3min")
    assert none["sampling_weights"] is None
    assert none["weights"]["D_DD"] == 1.0
    annotated = annotate_balance(frame, none, _config("none"))
    assert np.allclose(annotated["loss_weight"], 1.0)
    weighted = annotate_balance(frame, balance, _config("inverse_freq"))
    assert weighted.loc[weighted["split"] != "train", "loss_weight"].eq(1.0).all()
    d_rows = weighted.loc[weighted["storm_id"] == "d", "loss_weight"]
    assert np.allclose(d_rows, balance["weights"]["D_DD"])

    moved = frame.copy()
    moved.loc[moved["storm_id"] == "e", "split"] = "val"
    held_out = intensity_balance(moved, _config("inverse_freq"), "wind_kt_3min")
    assert held_out["counts"]["ESCS_SUCS"] == 0
    assert held_out["weights"]["ESCS_SUCS"] is None
    labels = ["ESCS_SUCS", "D_DD"]
    assert align_bin_weights(labels, held_out["weights"]).tolist() == pytest.approx([1.0, held_out["weights"]["D_DD"]])


def test_storm_cv_fold_is_not_in_the_frequency_rows() -> None:
    storms = pd.read_csv(ROOT / "tests" / "fixtures" / "incyde" / "storm_split.csv")
    winds = {1: 20.0, 2: 40.0, 3: 100.0, 4: 55.0, 5: 70.0, 6: 91.0}
    rows = []
    for storm in storms.itertuples(index=False):
        rows.append(
            {
                "storm_id": storm.storm_id,
                "proposed_split": storm.proposed_split,
                "cooccurrence_group": int(storm.cooccurrence_group),
                "is_synthetic": False,
                "aug_type": "",
                "wind_kt_3min": winds[int(storm.Sequence)],
                "wind_speed_kt": winds[int(storm.Sequence)],
                "wind_averaging": "3min",
            }
        )
    rows.append(
        {
            "storm_id": "INCYDE-S01_IMD-2018-001",
            "proposed_split": "train",
            "cooccurrence_group": 1,
            "is_synthetic": True,
            "aug_type": "rot_180",
            "wind_kt_3min": 20.0,
            "wind_speed_kt": 20.0,
            "wind_averaging": "3min",
        }
    )
    base = pd.DataFrame(rows)
    folded, _dropped = assign_incyde_split(base, "storm_cv", n_folds=2, cv_val_fold=0, cv_seed=42)
    balance = intensity_balance(folded, _config("inverse_freq", strategy="storm_cv"), "wind_kt_3min")
    counted, _note = frequency_rows(folded)
    val_ids = set(folded.loc[folded["split"] == "val", "storm_id"])
    test_ids = set(folded.loc[folded["split"] == "test", "storm_id"])
    assert val_ids
    assert test_ids
    assert set(counted["storm_id"]).isdisjoint(val_ids)
    assert set(counted["storm_id"]).isdisjoint(test_ids)
    assert not counted["is_synthetic"].astype(bool).any()
    assert balance["split_strategy"] == "storm_cv"
    assert balance["n_frequency_rows"] == counted.shape[0]


def test_on_fix_slice_keeps_the_overall_rows() -> None:
    truth = np.array([10.0, 20.0, 30.0, 40.0])
    pred = np.array([12.0, 20.0, 33.0, 40.0])
    flags = [True, "unknown", False, True]
    config = _config("none")
    overall = _score_split(truth, pred, config)
    subset = on_fix_metrics(truth, pred, flags, config)
    raw_subset = on_fix_metrics(truth, pred + 1.0, flags, config)
    assert overall["n"] == 4
    assert subset["n"] == 2
    assert raw_subset["n"] == 2
    assert subset["mae_kt"] != overall["mae_kt"]
    assert is_on_fix("unknown") is False
    assert is_on_fix(True) is True
    assert is_on_fix(False) is False
    assert is_on_fix(pd.NA) is False
    empty = on_fix_metrics(truth, pred, ["unknown"] * 4, config)
    assert empty["n"] == 0
    assert empty["mae_kt"] is None
