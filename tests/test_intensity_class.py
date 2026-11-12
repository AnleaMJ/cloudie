"""Per-class validation metrics. Arrays are synthetic, not cyclones."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluate import INDICATIVE_STATUS, intensity_class_evaluation
from src.metrics import storm_bootstrap_metrics
from src.schema import DEFAULT_NAMES
from src.splits import LeakageError


def _point(truth: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    error = np.asarray(pred, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
    }


def _quantiles(samples: list[float]) -> tuple[float, float]:
    return (
        float(np.quantile(samples, 0.025, method="linear")),
        float(np.quantile(samples, 0.975, method="linear")),
    )


def _storm_intervals(
    truth: np.ndarray, pred: np.ndarray, storms: list[str], n_resamples: int, seed: int
) -> dict[str, float]:
    ids = [str(storm) for storm in storms]
    order = list(dict.fromkeys(ids))
    groups = [np.flatnonzero(np.asarray(ids) == storm) for storm in order]
    draws = np.random.default_rng(seed).integers(0, len(groups), size=(n_resamples, len(groups)))
    maes: list[float] = []
    rmses: list[float] = []
    biases: list[float] = []
    truth = np.asarray(truth, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    for draw in draws:
        index = np.concatenate([groups[int(item)] for item in draw])
        error = pred[index] - truth[index]
        maes.append(float(np.mean(np.abs(error))))
        rmses.append(float(np.sqrt(np.mean(error**2))))
        biases.append(float(np.mean(error)))
    mae_low, mae_high = _quantiles(maes)
    rmse_low, rmse_high = _quantiles(rmses)
    bias_low, bias_high = _quantiles(biases)
    return {
        "mae_low": mae_low,
        "mae_high": mae_high,
        "rmse_low": rmse_low,
        "rmse_high": rmse_high,
        "bias_low": bias_low,
        "bias_high": bias_high,
    }


def _row_intervals(
    truth: np.ndarray, pred: np.ndarray, n_resamples: int, seed: int
) -> dict[str, float]:
    truth = np.asarray(truth, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    draws = np.random.default_rng(seed).integers(0, truth.size, size=(n_resamples, truth.size))
    maes: list[float] = []
    for draw in draws:
        error = pred[draw] - truth[draw]
        maes.append(float(np.mean(np.abs(error))))
    mae_low, mae_high = _quantiles(maes)
    return {"mae_low": mae_low, "mae_high": mae_high}


def _evaluate(**overrides) -> pd.DataFrame:
    truth = np.array([10.0, 10.0, 14.0, 40.0])
    pred = np.array([12.0, 10.0, 11.0, 46.0])
    storms = np.array(["a", "a", "b", "c"])
    classes = np.array(["D_DD", "D_DD", "D_DD", "CS"])
    payload = {
        "y_true": truth,
        "y_pred": pred,
        "storm_ids": storms,
        "classes": classes,
        "splits": np.array(["val", "val", "val", "val"]),
        "n_resamples": 80,
        "seed": 1,
    }
    payload.update(overrides)
    return intensity_class_evaluation(**payload)


def _row(table: pd.DataFrame, **query) -> pd.Series:
    mask = np.ones(len(table), dtype=bool)
    for key, value in query.items():
        mask &= table[key].astype(str).eq(str(value)).to_numpy()
    found = table.loc[mask]
    assert len(found) == 1, query
    return found.iloc[0]


def test_per_class_rmse_mae_bias_and_storm_counts() -> None:
    table = _evaluate(per_storm_breakdown=True)
    overall = _point([10.0, 10.0, 14.0, 40.0], [12.0, 10.0, 11.0, 46.0])
    summary = _row(table, subset="all", row_kind="overall", intensity_class="overall")
    assert summary["n_rows"] == 4
    assert summary["n_storms"] == 3
    assert summary["mae_kt"] == pytest.approx(overall["mae"])
    assert summary["rmse_kt"] == pytest.approx(overall["rmse"])
    assert summary["bias_kt"] == pytest.approx(overall["bias"])
    assert summary["bias_kt"] == pytest.approx(1.25)
    assert summary["rmse_kt"] == pytest.approx(3.5)
    assert summary["status"] == ""

    dd = _point([10.0, 10.0, 14.0], [12.0, 10.0, 11.0])
    class_row = _row(table, subset="all", row_kind="class", intensity_class="D_DD")
    assert class_row["n_rows"] == 3
    assert class_row["n_storms"] == 2
    assert class_row["mae_kt"] == pytest.approx(dd["mae"])
    assert class_row["mae_kt"] == pytest.approx(5.0 / 3.0)
    assert class_row["rmse_kt"] == pytest.approx(dd["rmse"])
    assert class_row["bias_kt"] == pytest.approx(dd["bias"])
    assert class_row["bias_kt"] == pytest.approx(-1.0 / 3.0)
    assert class_row["status"] == INDICATIVE_STATUS

    storm_a = _row(table, subset="all", row_kind="storm", intensity_class="D_DD", storm_id="a")
    assert storm_a["n_rows"] == 2
    assert storm_a["n_storms"] == 1
    assert storm_a["mae_kt"] == pytest.approx(1.0)
    assert storm_a["bias_kt"] == pytest.approx(1.0)
    assert storm_a["rmse_kt"] == pytest.approx(np.sqrt(2.0))
    assert storm_a["status"] == INDICATIVE_STATUS
    storm_b = _row(table, subset="all", row_kind="storm", intensity_class="D_DD", storm_id="b")
    assert storm_b["mae_kt"] == pytest.approx(3.0)
    assert storm_b["bias_kt"] == pytest.approx(-3.0)

    cs = _row(table, subset="all", row_kind="class", intensity_class="CS")
    assert cs["n_rows"] == 1
    assert cs["n_storms"] == 1
    assert cs["mae_kt"] == pytest.approx(6.0)
    assert cs["rmse_kt"] == pytest.approx(6.0)
    assert cs["bias_kt"] == pytest.approx(6.0)
    assert cs["mae_kt_ci_low"] == pytest.approx(cs["mae_kt"])
    assert cs["rmse_kt_ci_high"] == pytest.approx(cs["rmse_kt"])
    assert cs["bias_kt_ci_low"] == pytest.approx(cs["bias_kt"])
    assert cs["status"] == INDICATIVE_STATUS

    empty = _row(table, subset="all", row_kind="class", intensity_class="ESCS_SUCS")
    assert empty["n_rows"] == 0
    assert empty["n_storms"] == 0
    assert pd.isna(empty["mae_kt"])
    assert pd.isna(empty["rmse_kt"])
    assert pd.isna(empty["bias_kt"])
    assert empty["status"] == ""
    names = table.loc[table["row_kind"].eq("class") & table["subset"].eq("all"), "intensity_class"]
    assert list(names)[: len(DEFAULT_NAMES)] == list(DEFAULT_NAMES)


def test_bootstrap_resamples_storms_and_is_deterministic() -> None:
    truth = np.array([10.0, 10.0, 14.0])
    pred = np.array([12.0, 10.0, 11.0])
    storms = np.array(["a", "a", "b"])
    direct = storm_bootstrap_metrics(truth, pred, storms, n_resamples=80, seed=1)
    again = storm_bootstrap_metrics(truth, pred, storms, n_resamples=80, seed=1)
    assert again == direct
    expected = _storm_intervals(truth, pred, ["a", "a", "b"], n_resamples=80, seed=1)
    assert direct["mae_kt_ci_low"] == pytest.approx(expected["mae_low"])
    assert direct["mae_kt_ci_high"] == pytest.approx(expected["mae_high"])
    assert direct["rmse_kt_ci_low"] == pytest.approx(expected["rmse_low"])
    assert direct["rmse_kt_ci_high"] == pytest.approx(expected["rmse_high"])
    assert direct["bias_kt_ci_low"] == pytest.approx(expected["bias_low"])
    assert direct["bias_kt_ci_high"] == pytest.approx(expected["bias_high"])
    by_row = _row_intervals(truth, pred, n_resamples=80, seed=1)
    assert direct["mae_kt_ci_low"] != pytest.approx(by_row["mae_low"]) or direct[
        "mae_kt_ci_high"
    ] != pytest.approx(by_row["mae_high"])
    table = _evaluate()
    other = _evaluate()
    pd.testing.assert_frame_equal(table, other)
    class_row = _row(table, subset="all", row_kind="class", intensity_class="D_DD")
    assert class_row["mae_kt_ci_low"] == pytest.approx(direct["mae_kt_ci_low"])
    assert class_row["n_resamples"] == 80
    assert class_row["seed"] == 1
    assert class_row["resample"] == "storm_id"
    other_seed = storm_bootstrap_metrics(truth, pred, storms, n_resamples=80, seed=2)
    assert other_seed["seed"] == 2
    assert other_seed["mae_kt"] == pytest.approx(direct["mae_kt"])


def test_default_resamples_and_three_storms_are_not_indicative() -> None:
    table = intensity_class_evaluation(
        [10.0, 11.0],
        [12.0, 10.0],
        ["s", "s"],
        ["D_DD", "D_DD"],
        ["val", "val"],
        per_storm_breakdown=False,
    )
    assert set(table["n_resamples"]) == {1000}
    assert table["ci_level"].to_numpy() == pytest.approx(0.95)
    wide = intensity_class_evaluation(
        [10.0, 10.0, 10.0, 40.0, 40.0],
        [11.0, 12.0, 13.0, 45.0, 47.0],
        ["a", "b", "c", "d", "d"],
        ["D_DD", "D_DD", "D_DD", "CS", "CS"],
        ["val"] * 5,
        n_resamples=10,
        seed=0,
    )
    dd = _row(wide, subset="all", row_kind="class", intensity_class="D_DD")
    assert dd["n_storms"] == 3
    assert dd["n_rows"] == 3
    assert dd["status"] == ""
    assert not (
        (wide["row_kind"] == "storm") & (wide["intensity_class"] == "D_DD")
    ).any()
    cs = _row(wide, subset="all", row_kind="class", intensity_class="CS")
    assert cs["n_storms"] == 1
    assert cs["status"] == "indicative (n_storms<3)"
    assert list(
        wide.loc[
            (wide["row_kind"] == "storm") & (wide["intensity_class"] == "CS"),
            "storm_id",
        ]
    ) == ["d"]
    quiet = intensity_class_evaluation(
        [10.0, 14.0],
        [12.0, 11.0],
        ["a", "b"],
        ["D_DD", "D_DD"],
        ["val", "val"],
        n_resamples=5,
        seed=0,
        per_storm_breakdown=False,
    )
    assert "storm" not in set(quiet["row_kind"])
    assert _row(quiet, row_kind="class", intensity_class="D_DD")["status"] == INDICATIVE_STATUS


def test_on_fix_subset_keeps_only_best_track_fixes() -> None:
    table = _evaluate(on_fix=[True, False, True, True])
    assert set(table["subset"]) == {"all", "on_fix"}
    kept = _point([10.0, 14.0, 40.0], [12.0, 11.0, 46.0])
    overall = _row(table, subset="on_fix", row_kind="overall", intensity_class="overall")
    assert overall["n_rows"] == 3
    assert overall["n_storms"] == 3
    assert overall["mae_kt"] == pytest.approx(kept["mae"])
    assert overall["bias_kt"] == pytest.approx(kept["bias"])
    dd = _row(table, subset="on_fix", row_kind="class", intensity_class="D_DD")
    assert dd["n_rows"] == 2
    assert dd["n_storms"] == 2
    assert dd["mae_kt"] == pytest.approx(2.5)
    assert dd["bias_kt"] == pytest.approx(-0.5)
    assert dd["rmse_kt"] == pytest.approx(np.sqrt(6.5))
    storm_a = _row(table, subset="on_fix", row_kind="storm", intensity_class="D_DD", storm_id="a")
    assert storm_a["n_rows"] == 1
    assert storm_a["mae_kt"] == pytest.approx(2.0)
    words = _evaluate(on_fix=["yes", "no", 1, "true"])
    pd.testing.assert_frame_equal(table, words)
    absent = _evaluate()
    assert set(absent["subset"]) == {"all"}
    with pytest.raises(ValueError, match="on_fix"):
        _evaluate(on_fix=["maybe", True, True, True])
    with pytest.raises(ValueError, match="on_fix"):
        _evaluate(on_fix=[True, None, True, True])


def test_wind_averaging_is_grouped_and_optional() -> None:
    table = _evaluate(
        wind_averaging=["3-min", "1min", "3-min", "1min"],
        per_storm_breakdown=False,
        n_resamples=5,
    )
    assert set(table["wind_averaging"]) == {"3-min", "1min"}
    first = table["wind_averaging"].iloc[0]
    assert first == "3-min"
    three = _row(
        table,
        wind_averaging="3-min",
        subset="all",
        row_kind="class",
        intensity_class="D_DD",
    )
    assert three["n_rows"] == 2
    assert three["n_storms"] == 2
    one = _row(
        table,
        wind_averaging="1min",
        subset="all",
        row_kind="overall",
        intensity_class="overall",
    )
    assert one["n_rows"] == 2
    assert one["n_storms"] == 2
    pooled = table[(table["row_kind"] == "overall") & (table["n_rows"] == 4)]
    assert pooled.empty
    missing = _evaluate(per_storm_breakdown=False, n_resamples=5)
    assert missing["wind_averaging"].isna().all()


def test_validation_split_is_required() -> None:
    with pytest.raises(LeakageError, match="validation"):
        _evaluate(splits=["test", "test", "test", "test"])
    with pytest.raises(LeakageError, match="test"):
        _evaluate(splits=["val", "val", "val", "test"])
    with pytest.raises(ValueError, match="n_resamples"):
        _evaluate(n_resamples=0)
    with pytest.raises(ValueError, match="finite"):
        _evaluate(y_pred=[12.0, np.nan, 11.0, 46.0])
