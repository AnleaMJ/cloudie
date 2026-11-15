"""INCYDE loader, 3-minute labels, and split rules. Fixtures only. No downloads."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.incyde import (
    ZIP_MD5,
    YEAR_HOLDOUT_NOTE,
    assert_archive_md5,
    assign_incyde_split,
    attach_frame_flags,
    build_cv_folds,
    imd_class_of,
    parse_filename,
    read_exclusions,
    read_frame_flags,
    read_rows,
    read_storm_split,
    storm_year,
    validate_cv_folds,
)
from src.labels import (
    WIND_CONVERSION_NOTE,
    apply_wind_standard,
    back_converted_prediction,
)
from src.preprocessing import rotate_with_circular_mask
from src.splits import LeakageError
from src.train import assert_test_score_allowed, averaging_mismatch, normalization_frame

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "incyde"
AUDIT_SPLIT = ROOT / "incyde" / "audit" / "storm_split.csv"
AUDIT_EXCLUSIONS = ROOT / "incyde" / "audit" / "exclusions.csv"

KEPT_NAMES = (
    "01JAN2018_0030_1.jpg",
    "01JAN2018_0600_2.jpg",
    "01JAN2018_0030_1_rot_180.jpg",
    "15JUN2019_1200_9.jpg",
    "15JUN2019_1200_9_rot_180.jpg",
    "20MAY2021_0900_3.jpg",
    "31DEC2018_2330_4.jpg",
    "01JAN2019_0230_4.jpg",
    "02FEB2018_1100_5.jpg",
)
DROPPED_NAMES = (
    "01JAN2018_0030_1_flip_h.jpg",
    "01JAN2018_0030_1_flip_v.jpg",
    "01JAN2018_0030_1_rot_90.jpg",
    "01JAN2018_0030_1_rot_270.jpg",
    "10SEP2017_0530_1.jpg",
    "10SEP2017_0530_1_rot_180.jpg",
)


def write_images(root: Path, pixels: dict[str, int] | None = None) -> None:
    """Write lossless gray chips under INCYDE filenames. Nothing is downloaded."""
    root.mkdir(parents=True, exist_ok=True)
    values = pixels or {}
    for name in KEPT_NAMES + DROPPED_NAMES:
        level = int(values.get(name, 40))
        image = Image.fromarray(np.full((16, 16), level, dtype=np.uint8), mode="L")
        image.save(root / name, format="PNG")


def load_fixture(root: Path, **kwargs) -> tuple[pd.DataFrame, int]:
    write_images(root, kwargs.pop("pixels", None))
    frame, dropped = read_rows(
        root,
        manifest_csv=FIXTURE / "labels.csv",
        storm_split_csv=FIXTURE / "storm_split.csv",
        exclusions_csv=FIXTURE / "exclusions.csv",
        drop_qc_flags=kwargs.get("drop_qc_flags"),
    )
    return frame, dropped


def test_audit_split_is_sixty_fourteen_seventeen_and_one_year_each() -> None:
    storms = read_storm_split(AUDIT_SPLIT)
    counts = storms["proposed_split"].value_counts().to_dict()
    assert counts == {"train": 60, "val": 14, "test": 17}
    assert storms["storm_id"].is_unique
    spread = storms.groupby("cooccurrence_group")["proposed_split"].nunique()
    assert int((spread > 1).sum()) == 0
    years = storms["storm_id"].map(storm_year)
    assert storms.assign(year=years).groupby("storm_id")["year"].nunique().max() == 1
    groups = storms.groupby("cooccurrence_group")["storm_id"].apply(lambda ids: {storm_year(i) for i in ids})
    assert all(len(years_in_group) == 1 for years_in_group in groups)


def test_committed_exclusions_default_only_the_bad_date_flag() -> None:
    rows = read_exclusions(AUDIT_EXCLUSIONS)
    dropped = rows.loc[rows["default_exclude"], "qc_flag"]
    assert set(dropped) == {"wrong_date_imd_daymonth_swap"}
    assert int(rows["default_exclude"].sum()) > 0
    kept = set(rows.loc[~rows["default_exclude"], "qc_flag"])
    assert "label_ne_imd_interp" in kept
    assert "outside_imd_track_span" in kept


def test_filename_clock_is_ist_and_obj_is_not_the_storm(tmp_path: Path) -> None:
    frame, _dropped = load_fixture(tmp_path)
    parsed = parse_filename("01JAN2018_0030_1.jpg")
    assert parsed["time_ist"] == "2018-01-01T00:30:00+05:30"
    assert parsed["time_utc"] == "2017-12-31T19:00:00Z"
    assert parsed["object_index"] == 1
    row = frame.loc[frame["sample_id"] == "01JAN2018_0030_1"].iloc[0]
    assert row["time_utc"] == "2017-12-31T19:00:00Z"
    assert row["time_ist"].startswith("2018-01-01T00:30:00")
    assert int(row["storm_year"]) == 2018
    pair = frame.loc[frame["incyde_sequence"] == 1]
    assert set(pair["storm_id"]) == {"INCYDE-S01_IMD-2018-001"}
    assert set(pair["object_index"]) == {1, 2}
    assert "1" not in set(pair["storm_id"])
    assert "2" not in set(pair["storm_id"])
    nine = frame.loc[frame["filename"] == "15JUN2019_1200_9.jpg"] if "filename" in frame.columns else frame.loc[frame["sample_id"] == "15JUN2019_1200_9"]
    assert nine["storm_id"].iloc[0] == "INCYDE-S02_IMD-2019-002"
    assert int(nine["object_index"].iloc[0]) == 9
    # Fixture names are not in the committed audit table, so the join stays unknown.
    assert set(frame["on_fix"]) == {"unknown"}
    assert set(frame["near_land"]) == {"unknown"}
    assert frame["distance_to_land_km"].isna().all()
    assert "landfall" not in frame.columns


def test_one_storm_does_not_take_a_second_year_from_the_filename(tmp_path: Path) -> None:
    frame, _dropped = load_fixture(tmp_path)
    storm = frame.loc[frame["storm_id"] == "INCYDE-S04_IMD-2018-004"]
    assert set(storm["storm_year"]) == {2018}
    assert storm["time_ist"].str[:4].nunique() == 2
    assert storm_year("INCYDE-S04_IMD-2018-004") == 2018
    with pytest.raises(ValueError, match="INCYDE-Sxx"):
        storm_year("not-a-storm")


def test_default_split_drops_flips_bars_s42_and_eval_synthetics(tmp_path: Path) -> None:
    rows, _dropped = load_fixture(tmp_path)
    assigned, _synthetic = assign_incyde_split(rows, "storm_wise")
    assert set(assigned["aug_type"]) <= {"", "rot_180"}
    assert not assigned["sample_id"].str.contains("flip_").any()
    assert not assigned["sample_id"].str.contains("rot_90").any()
    assert not assigned["sample_id"].str.contains("rot_270").any()
    assert "10SEP2017_0530_1" not in set(assigned["sample_id"])
    assert "10SEP2017_0530_1_rot_180" not in set(assigned["sample_id"])
    kept_qc = assigned.loc[assigned["sample_id"] == "02FEB2018_1100_5"].iloc[0]
    assert kept_qc["qc_flag"] == "label_ne_imd_interp"
    assert kept_qc["split"] == "train"
    synthetic = assigned.loc[assigned["is_synthetic"].astype(bool)]
    assert set(synthetic["split"]) == {"train"}
    assert set(synthetic["aug_type"]) == {"rot_180"}
    assert set(synthetic["source_frame_id"]) == {"01JAN2018_0030_1.jpg"}
    assert set(synthetic["storm_id"]) == {"INCYDE-S01_IMD-2018-001"}
    val = assigned.loc[assigned["split"] == "val"]
    assert not bool(val["is_synthetic"].astype(bool).any())
    assert "15JUN2019_1200_9_rot_180" not in set(assigned["sample_id"])
    per_storm = assigned.groupby("storm_id")["split"].nunique()
    assert int(per_storm.max()) == 1
    group = assigned.loc[assigned["cooccurrence_group"] == 1, "split"]
    assert set(group) == {"train"}
    assert set(assigned["wind_averaging"]) == {"3min"}
    assert set(assigned["wind_averaging_s"].astype(int)) == {180}
    official = assigned.loc[assigned["sample_id"] == "01JAN2018_0030_1", "incyde_set"].iloc[0]
    assert official == "Test"
    assert assigned.loc[assigned["sample_id"] == "01JAN2018_0030_1", "split"].iloc[0] == "train"


def test_frame_flags_inherit_and_keep_unknown(tmp_path: Path) -> None:
    flags = tmp_path / "frame_flags.csv"
    flags.write_text(
        "filename,on_fix,lat,lon,distance_to_land_km,near_land\n"
        "01JAN2018_0030_1.jpg,true,12.0,80.0,5.0,true\n"
        "01JAN2018_0600_2.jpg,false,12.0,80.0,,unknown\n"
        "15JUN2019_1200_9.jpg,false,10.0,70.0,40.0,false\n",
        encoding="utf-8",
    )
    write_images(tmp_path / "images")
    frame, _dropped = read_rows(
        tmp_path / "images",
        manifest_csv=FIXTURE / "labels.csv",
        storm_split_csv=FIXTURE / "storm_split.csv",
        exclusions_csv=FIXTURE / "exclusions.csv",
        frame_flags_csv=flags,
        check_images=False,
    )
    original = frame.loc[frame["sample_id"] == "01JAN2018_0030_1"].iloc[0]
    copy = frame.loc[frame["sample_id"] == "01JAN2018_0030_1_rot_180"].iloc[0]
    assert original["on_fix"] is True
    assert copy["on_fix"] is True
    assert copy["distance_to_land_km"] == pytest.approx(5.0)
    assert copy["near_land"] == "true"
    assert copy["source_frame_id"] == "01JAN2018_0030_1.jpg"
    blank = frame.loc[frame["sample_id"] == "01JAN2018_0600_2"].iloc[0]
    assert blank["on_fix"] is False
    assert blank["near_land"] == "unknown"
    assert pd.isna(blank["distance_to_land_km"])
    missing = frame.loc[frame["sample_id"] == "20MAY2021_0900_3"].iloc[0]
    assert missing["on_fix"] == "unknown"
    assert missing["near_land"] == "unknown"
    assert pd.isna(missing["distance_to_land_km"])
    assert missing["distance_to_land_km"] != 0
    offshore = frame.loc[frame["sample_id"] == "15JUN2019_1200_9"].iloc[0]
    assert offshore["near_land"] == "false"
    assert offshore["distance_to_land_km"] == pytest.approx(40.0)
    standard = apply_wind_standard(
        frame.loc[frame["sample_id"].isin(["01JAN2018_0030_1", "01JAN2018_0600_2", "20MAY2021_0900_3"])].reset_index(drop=True),
        factor=0.93,
        policy="flag",
        open_water_km=20,
    )
    assert standard["wind_kt_3min"].tolist() == standard["wind_kt"].tolist()
    assert set(standard["wind_kt_scale"]) == {1.0}
    assert standard["near_land"].tolist() == ["true", "unknown", "unknown"]
    assert standard["wind_conversion"].tolist() == ["", "", ""]


def test_committed_frame_flag_is_inherited_by_the_source_name() -> None:
    flags = read_frame_flags()
    phailin = flags.loc[flags["filename"] == "08OCT2013_0830_1.jpg"].iloc[0]
    assert phailin["on_fix"] is True
    assert phailin["near_land"] == "false"
    assert phailin["distance_to_land_km"] == pytest.approx(158.91)
    rows = pd.DataFrame(
        {
            "source_frame_id": [
                "08OCT2013_0830_1.jpg",
                "08OCT2013_0830_1.jpg",
                "not-in-the-audit.jpg",
            ],
            "filename": [
                "08OCT2013_0830_1.jpg",
                "08OCT2013_0830_1_rot_180.jpg",
                "not-in-the-audit.jpg",
            ],
        }
    )
    joined = attach_frame_flags(rows, flags)
    assert joined["on_fix"].tolist() == [True, True, "unknown"]
    assert joined["near_land"].tolist() == ["false", "false", "unknown"]
    assert joined["distance_to_land_km"].iloc[1] == pytest.approx(158.91)
    assert pd.isna(joined["distance_to_land_km"].iloc[2])
    inland = flags.loc[flags["filename"] == "20MAY2020_1730_1.jpg"].iloc[0]
    assert inland["on_fix"] is True
    assert inland["near_land"] == "true"
    assert inland["distance_to_land_km"] == pytest.approx(0.0)
    held = apply_wind_standard(
        pd.DataFrame(
            {
                "wind_speed_kt": [80.0],
                "wind_kt": [80.0],
                "wind_averaging": ["3min"],
                "distance_to_land_km": [float(inland["distance_to_land_km"])],
                "near_land": [inland["near_land"]],
                "on_fix": [inland["on_fix"]],
            }
        ),
        factor=0.93,
        policy="flag",
        open_water_km=20,
    )
    assert held["wind_kt_3min"].iloc[0] == pytest.approx(80.0)
    assert held["near_land"].iloc[0] == "true"
    assert held["wind_conversion"].iloc[0] == ""


def test_drop_qc_flags_removes_an_otherwise_kept_row(tmp_path: Path) -> None:
    frame, _dropped = load_fixture(tmp_path, drop_qc_flags=["label_ne_imd_interp"])
    assert "02FEB2018_1100_5" not in set(frame["sample_id"])


def test_imd_class_uses_the_three_minute_scale() -> None:
    assert imd_class_of(16) == "<17"
    assert imd_class_of(27) == "D"
    assert imd_class_of(33) == "DD"
    assert imd_class_of(47) == "CS"
    assert imd_class_of(63) == "SCS"
    assert imd_class_of(89) == "VSCS"
    assert imd_class_of(119) == "ESCS"
    assert imd_class_of(120) == "SuCS"


def test_incyde_labels_pass_through_and_one_minute_uses_the_factor() -> None:
    incyde = pd.DataFrame(
        {
            "wind_speed_kt": [40.0, 80.0],
            "wind_kt": [40.0, 80.0],
            "wind_averaging": ["3min", "3min"],
        }
    )
    passed = apply_wind_standard(incyde, factor=0.93, policy="flag", open_water_km=20)
    assert passed["wind_kt_3min"].tolist() == [40.0, 80.0]
    assert passed["wind_speed_kt"].tolist() == [40.0, 80.0]
    assert set(passed["wind_conversion"]) == {""}
    assert list(passed["wind_averaging_s"]) == [180, 180]

    one_minute = pd.DataFrame(
        {
            "wind_speed_kt": [100.0, 100.0],
            "wind_averaging": ["1min", "1min"],
            "source": ["nasa", "tcir"],
            "lat": [15.0, -20.0],
            "lon": [80.0, 70.0],
        }
    )
    converted = apply_wind_standard(one_minute, factor=0.93, policy="flag", open_water_km=20)
    assert converted["wind_kt_3min"].tolist() == pytest.approx([93.0, 93.0])
    assert converted["wind_speed_kt"].tolist() == [100.0, 100.0]
    assert set(converted["wind_conversion"]) == {WIND_CONVERSION_NOTE}
    assert list(converted["wind_averaging_s"]) == [60, 60]
    assert set(converted["near_land"]) == {"unknown"}
    assert set(converted["conversion_exposure_flag"]) == {"unknown"}

    custom = apply_wind_standard(one_minute.drop(columns=["lat", "lon"]), factor=0.5, policy="flag", open_water_km=20)
    assert custom["wind_kt_3min"].iloc[0] == pytest.approx(50.0)
    assert custom["wind_conversion"].iloc[0].startswith("0.50, derived from WMO/TD-1555")


def test_near_land_policy_does_not_convert_in_silence() -> None:
    frame = pd.DataFrame(
        {
            "wind_speed_kt": [100.0, 100.0, 100.0],
            "wind_averaging": ["1min", "1min", "1min"],
            "distance_to_land_km": [25.0, 10.0, np.nan],
        }
    )
    flagged = apply_wind_standard(frame, factor=0.93, policy="flag", open_water_km=20)
    assert flagged["near_land"].tolist() == ["false", "true", "unknown"]
    assert flagged["wind_kt_3min"].tolist() == pytest.approx([93.0, 93.0, 93.0])
    assert flagged["conversion_exposure_flag"].tolist() == ["false", "true", "unknown"]
    assert flagged["wind_speed_kt"].tolist() == [100.0, 100.0, 100.0]

    held = apply_wind_standard(frame, factor=0.93, policy="raw", open_water_km=20)
    assert held["wind_kt_3min"].tolist() == pytest.approx([93.0, 100.0, 93.0])
    assert held["conversion_exposure_flag"].tolist() == ["false", "true", "unknown"]
    assert "held raw" in held["wind_conversion"].iloc[1]

    landfall = pd.DataFrame(
        {"wind_speed_kt": [50.0], "wind_averaging": ["1min"], "landfall": [True]}
    )
    marked = apply_wind_standard(landfall, factor=0.93, policy="flag", open_water_km=20)
    assert marked["near_land"].iloc[0] == "true"
    assert marked["conversion_exposure_flag"].iloc[0] == "true"
    assert marked["wind_kt_3min"].iloc[0] == pytest.approx(46.5)

    edge = pd.DataFrame(
        {"wind_speed_kt": [10.0], "wind_averaging": ["1min"], "distance_to_land_km": [20.0]}
    )
    at_twenty = apply_wind_standard(edge, factor=0.93, policy="flag", open_water_km=20)
    assert at_twenty["near_land"].iloc[0] == "true"


def test_back_converted_predictions_divide_by_the_factor() -> None:
    raw = back_converted_prediction(np.array([93.0, 40.0]), np.array([0.93, 1.0]))
    assert raw.tolist() == pytest.approx([100.0, 40.0])


def test_cv_folds_from_seed_and_from_a_csv(tmp_path: Path) -> None:
    storms = read_storm_split(FIXTURE / "storm_split.csv")
    first = build_cv_folds(storms, n_folds=2, seed=42)
    second = build_cv_folds(storms, n_folds=2, seed=42)
    assert first.equals(second)
    validate_cv_folds(first, storms)
    test_ids = set(storms.loc[storms["proposed_split"] == "test", "storm_id"])
    assert set(first["storm_id"]).isdisjoint(test_ids)
    merged = storms.merge(first, on="storm_id", how="inner")
    assert int(merged.groupby("cooccurrence_group")["fold"].nunique().max()) == 1
    assert set(merged.loc[merged["cooccurrence_group"] == 1, "fold"]) == {int(merged.loc[merged["storm_id"] == "INCYDE-S01_IMD-2018-001", "fold"].iloc[0])}

    verbatim = pd.DataFrame(
        {
            "storm_id": [
                "INCYDE-S01_IMD-2018-001",
                "INCYDE-S04_IMD-2018-004",
                "INCYDE-S02_IMD-2019-002",
                "INCYDE-S06_IMD-2018-006",
            ],
            "fold": [3, 3, 1, 4],
        }
    )
    path = tmp_path / "cv_folds.csv"
    verbatim.to_csv(path, index=False)
    present = storms.loc[storms["storm_id"] != "INCYDE-S05_IMD-2017-042"]
    validate_cv_folds(pd.read_csv(path).assign(storm_id=lambda frame: frame["storm_id"].astype(str)), present)
    rows, _dropped = load_fixture(tmp_path / "images")
    assigned, _synthetic = assign_incyde_split(rows, "storm_cv", n_folds=5, cv_val_fold=3, cv_folds_csv=path)
    group = assigned.loc[assigned["cooccurrence_group"] == 1]
    assert set(group["split"]) == {"val"}
    assert set(group["is_synthetic"].astype(bool)) <= {True, False}
    assert not bool(assigned.loc[assigned["split"] != "train", "is_synthetic"].astype(bool).any())
    assert "INCYDE-S03_IMD-2021-003" in set(assigned.loc[assigned["split"] == "test", "storm_id"])
    assert int(assigned.loc[assigned["storm_id"] == "INCYDE-S06_IMD-2018-006", "cv_fold"].iloc[0]) == 4

    broken = verbatim.copy()
    broken.loc[broken["storm_id"] == "INCYDE-S04_IMD-2018-004", "fold"] = 1
    with pytest.raises(LeakageError, match="cooccurrence_group"):
        validate_cv_folds(broken, present)
    leaked = pd.concat([verbatim, pd.DataFrame({"storm_id": ["INCYDE-S03_IMD-2021-003"], "fold": [0]})], ignore_index=True)
    with pytest.raises(LeakageError, match="test storms"):
        validate_cv_folds(leaked, storms)

    audit = read_storm_split(AUDIT_SPLIT)
    audit_folds = build_cv_folds(audit, n_folds=5, seed=42)
    validate_cv_folds(audit_folds, audit)
    assert len(audit_folds) == 74
    assert set(audit_folds["storm_id"]).isdisjoint(set(audit.loc[audit["proposed_split"] == "test", "storm_id"]))


def test_year_holdout_uses_the_storm_id_year_and_refuses_to_score() -> None:
    storms = read_storm_split(FIXTURE / "storm_split.csv")
    years = storms["storm_id"].map(storm_year)
    assert set(years) == {2018, 2019, 2021, 2017}
    frame = storms.rename(columns={"proposed_split": "proposed_split"}).copy()
    frame["is_synthetic"] = False
    frame["aug_type"] = ""
    frame["split"] = frame["proposed_split"]
    assigned, _dropped = assign_incyde_split(frame, "year_holdout_2019")
    by_storm = assigned.groupby("storm_id")["split"].first()
    assert by_storm["INCYDE-S02_IMD-2019-002"] == "test"
    assert by_storm["INCYDE-S03_IMD-2021-003"] == "val"
    assert by_storm["INCYDE-S01_IMD-2018-001"] == "train"
    assert by_storm["INCYDE-S04_IMD-2018-004"] == "train"
    assert "Agrawal" in YEAR_HOLDOUT_NOTE
    assert "2410.08218" in YEAR_HOLDOUT_NOTE
    with pytest.raises(RuntimeError, match="allow-final-test"):
        assert_test_score_allowed("incyde", False)
    assert_test_score_allowed("incyde", True)
    assert_test_score_allowed("nasa", False)


def test_mismatch_flag_follows_the_label_column() -> None:
    assert averaging_mismatch({"1min"}, {"3min"}, "wind_kt_3min", "wind_kt_3min") is False
    assert averaging_mismatch({"1min"}, {"3min"}, None, "wind_kt_3min") is True
    assert averaging_mismatch({"1min"}, {"3min"}, "wind_speed_kt", "wind_speed_kt") is True
    assert averaging_mismatch({"1min"}, {"3min"}, "wind_kt_3min", "wind_kt") is True
    assert averaging_mismatch({"synthetic"}, {"synthetic"}, None, "wind_kt_3min") is False


def test_raw_target_does_not_divide_raw_metrics_by_the_factor(tmp_path: Path) -> None:
    from src.train import _path_records

    frame = pd.DataFrame(
        {
            "split": ["val", "val"],
            "image_path": ["a.npy", "b.npy"],
            "_class": ["CS", "CS"],
            "wind_speed_kt": [100.0, 80.0],
            "wind_kt": [100.0, 80.0],
            "wind_kt_3min": [93.0, 74.4],
            "wind_kt_scale": [0.93, 0.93],
        }
    )
    raw = _path_records(frame, tmp_path / "manifest.csv", "val", "wind_speed_kt")
    assert [row["scale"] for row in raw] == [1.0, 1.0]
    assert [row["wind"] for row in raw] == [100.0, 80.0]
    back = back_converted_prediction(
        np.array([row["wind"] for row in raw]),
        np.array([row["scale"] for row in raw]),
    )
    assert back.tolist() == pytest.approx([100.0, 80.0])
    converted = _path_records(frame, tmp_path / "manifest.csv", "val", "wind_kt_3min")
    assert [row["scale"] for row in converted] == pytest.approx([0.93, 0.93])
    assert back_converted_prediction(
        np.array([93.0, 74.4]), np.array([row["scale"] for row in converted])
    ).tolist() == pytest.approx([100.0, 80.0])


def test_grey_jpeg_and_wide_chip_survive_the_circular_mask() -> None:
    from src.preprocessing import prepare_training_chip

    rgb = np.zeros((16, 16, 3), dtype=np.float32)
    rgb[..., 0] = rgb[..., 1] = rgb[..., 2]
    rgb[:, 8:] = 10.0
    rgb[0, 0] = 255.0
    rotated = rotate_with_circular_mask(rgb, 12.0)
    assert rotated.ndim == 2
    assert np.isnan(rotated[0, 0])
    assert np.isfinite(rotated[8, 8])
    colour = rgb.copy()
    colour[..., 2] = 1.0
    with pytest.raises(ValueError, match="grey IR"):
        rotate_with_circular_mask(colour, 0.0)
    wide = np.full((8, 64, 3), 40.0, dtype=np.float32)
    chip = prepare_training_chip(wide, 16, 25.0, None, 0.30)
    assert chip.shape == (1, 16, 16)
    assert np.isfinite(chip).all()


def test_mixed_rectangles_are_not_one_native_size() -> None:
    from src.train import refuse_mixed_resolution

    frame = pd.DataFrame(
        {
            "orig_size": [-1, -1],
            "native_height": [10, 12],
            "native_width": [40, 18],
            "storm_id": ["a", "b"],
        }
    )
    with pytest.raises(ValueError, match="mix native sizes"):
        refuse_mixed_resolution(frame, "incyde", None)
    refuse_mixed_resolution(frame, "incyde", 16)


def test_audit_manifest_without_sequence_uses_the_storm_table(tmp_path: Path) -> None:
    name = "01JAN2018_0030_1.jpg"
    Image.fromarray(np.full((16, 16), 40, dtype=np.uint8), mode="L").save(tmp_path / name, format="PNG")
    manifest = tmp_path / "audit.csv"
    pd.DataFrame(
        {
            "storm_id": ["INCYDE-S01_IMD-2018-001"],
            "label_kt": [40],
            "file": [name],
        }
    ).to_csv(manifest, index=False)
    frame, _dropped = read_rows(
        tmp_path,
        manifest_csv=manifest,
        storm_split_csv=FIXTURE / "storm_split.csv",
        exclusions_csv=FIXTURE / "exclusions.csv",
        check_images=True,
    )
    assert int(frame["incyde_sequence"].iloc[0]) == 1
    assert frame["storm_id"].iloc[0] == "INCYDE-S01_IMD-2018-001"


def test_rotation_mask_drops_corners_and_does_not_flip() -> None:
    image = np.zeros((16, 16), dtype=np.float32)
    image[:, 8:] = 10.0
    image[0, 0] = 255.0
    image[8, 8] = 4.0
    rotated = rotate_with_circular_mask(image, 12.0)
    assert np.isnan(rotated[0, 0])
    assert np.isnan(rotated[0, -1])
    assert np.isfinite(rotated[8, 8])
    assert float((~np.isfinite(rotated)).mean()) < 0.30
    left = float(np.nanmean(rotated[:, :8]))
    right = float(np.nanmean(rotated[:, 8:]))
    assert right > left


def test_normalization_uses_original_train_frames_only(tmp_path: Path) -> None:
    from src.preprocessing import fit_norm_stats
    from src.train import fit_split_normalization

    train = tmp_path / "train.npy"
    np.save(train, np.full((8, 8), 10.0, dtype=np.float32))
    frame = pd.DataFrame(
        {
            "sample_id": ["orig", "copy", "val"],
            "image_path": [str(train), str(tmp_path / "missing-copy.npy"), str(tmp_path / "missing-val.npy")],
            "storm_id": ["a", "a", "b"],
            "split": ["train", "train", "val"],
            "is_synthetic": [False, True, False],
            "wind_speed_kt": [30.0, 30.0, 90.0],
            "wind_averaging": ["3min", "3min", "3min"],
            "_class": ["CS", "CS", "VSCS"],
        }
    )
    kept = normalization_frame(frame, "incyde")
    assert set(kept["sample_id"]) == {"orig", "val"}
    config = {"model": {"name": "simple_cnn"}, "data": {"image_size": 8, "max_missing_fraction": 0.30}}
    stats = fit_split_normalization(kept, tmp_path / "manifest.csv", config)
    assert stats.source_split == "train"
    assert stats.mean == pytest.approx(10.0)
    dirty = np.full((4, 4), 10.0, dtype=np.float32)
    dirty[0, 0] = np.nan
    finite = fit_norm_stats([dirty], "train")
    assert finite.mean == pytest.approx(10.0)


def test_archive_md5_is_the_zenodo_value_and_a_local_file_must_match(tmp_path: Path) -> None:
    assert ZIP_MD5 == "ed5dac254c689e5a327dee7580324347"
    path = tmp_path / "not-the-zip.bin"
    path.write_bytes(b"not the incyde archive")
    with pytest.raises(ValueError, match=ZIP_MD5):
        assert_archive_md5(path)
    source = (ROOT / "src" / "incyde.py").read_text(encoding="utf-8").lower()
    for token in ("urlopen", "requests", "download=true", "zenodo.org/api"):
        assert token not in source


def test_loader_refuses_a_missing_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not download"):
        read_rows(tmp_path / "absent", manifest_csv=FIXTURE / "labels.csv", storm_split_csv=FIXTURE / "storm_split.csv", exclusions_csv=FIXTURE / "exclusions.csv")
