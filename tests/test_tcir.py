"""TCIR IR1 loader on a tiny synthetic HDF5 written in TCIR's own layout.

No network. The fixture mirrors the real file: ``matrix`` N x 201 x 201 x 4
float32 and ``info`` as a pandas 0.15 fixed frame (numeric block plus a
pickled object block), which current pandas cannot read.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

h5py = pytest.importorskip("h5py")

from src import tcir  # noqa: E402
from src.splits import assert_no_group_leakage, assign_storms  # noqa: E402

# (data_set, ID, time, lon, lat, Vmax)
ROWS = [
    ("WPAC", "200301W", "2003050100", 140.0, 15.0, 35.0),
    ("WPAC", "200301W", "2003050103", 139.5, 15.5, 45.0),
    ("ATLN", "200302L", "2003080100", 300.0, 20.0, 60.0),
    ("ATLN", "200302L", "2003080103", 299.0, 20.5, 65.0),
    ("SH", "200303S", "2003020100", 150.0, -15.0, 80.0),
    ("SH", "200303S", "2003020103", 150.5, -15.5, 90.0),
    ("IO", "200304I", "2003060100", 88.0, 14.0, 30.0),
    ("CPAC", "200305C", "2003090100", 200.0, 16.0, 50.0),
    ("SH", "200306S", "2003030100", 60.0, -12.0, 40.0),  # all NaN IR1
    ("SH", "200306S", "2003030103", 60.5, -12.5, 45.0),  # 50% NaN IR1
    ("EPAC", "200307E", "2003070100", 250.0, 15.0, 25.0),  # 400 K pixel
    ("EPAC", "200307E", "2003070103", 249.0, 15.5, 30.0),  # a few NaNs, kept
]


def _frame(index: int) -> np.ndarray:
    rng = np.random.default_rng(index)
    ir1 = rng.uniform(190.0, 300.0, (201, 201)).astype(np.float32)
    ir1[:100, :] -= 20.0  # top half colder so a north-south flip is visible
    other = np.stack(
        [
            np.full((201, 201), 1000.0, np.float32),  # WV: out of IR1 range on purpose
            np.full((201, 201), np.nan, np.float32),  # VIS: all NaN on purpose
            np.full((201, 201), 9.96921e36, np.float32),  # PMW fill value
        ],
        axis=-1,
    )
    return np.concatenate([ir1[..., None], other], axis=-1)


def write_fixture(path) -> np.ndarray:  # noqa: ANN001
    n = len(ROWS)
    matrix = np.stack([_frame(i) for i in range(n)])
    matrix[8, :, :, 0] = np.nan
    matrix[9, :100, :, 0] = np.nan
    matrix[9, 100, :2, 0] = np.nan
    matrix[10, 5, 5, 0] = 400.0
    matrix[11, 0, :3, 0] = np.nan
    numeric = np.array([[r[3], r[4], r[5], 0.0, 1000.0] for r in ROWS], dtype=np.float64)
    objects = np.array([[r[0], r[1], r[2]] for r in ROWS], dtype=object)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("matrix", data=matrix)
        info = handle.create_group("info")
        info.attrs["nblocks"] = np.int64(2)
        info.attrs["pandas_type"] = np.bytes_(b"frame")
        info.create_dataset(
            "axis0",
            data=np.array([b"data_set", b"ID", b"lon", b"lat", b"time", b"Vmax", b"R35_4qAVG", b"MSLP"]),
        )
        info.create_dataset("axis1", data=np.arange(n, dtype=np.int64))
        info.create_dataset("block0_items", data=np.array([b"lon", b"lat", b"Vmax", b"R35_4qAVG", b"MSLP"]))
        info.create_dataset("block0_values", data=numeric)
        info.create_dataset("block1_items", data=np.array([b"data_set", b"ID", b"time"]))
        blob = np.frombuffer(pickle.dumps(objects, protocol=2), dtype=np.uint8)
        vl = info.create_dataset("block1_values", (1,), dtype=h5py.vlen_dtype(np.uint8))
        vl[0] = blob
    return matrix


@pytest.fixture()
def fixture_h5(tmp_path):
    path = tmp_path / "TCIR-TINY.h5"
    matrix = write_fixture(path)
    return path, matrix


def test_read_info_decodes_fixed_frame(fixture_h5):
    path, _ = fixture_h5
    info = tcir.read_info(path)
    assert list(info.columns[:8]) == ["data_set", "ID", "lon", "lat", "time", "Vmax", "R35_4qAVG", "MSLP"]
    assert info["ID"].tolist()[0] == "200301W"
    assert info["h5_index"].tolist() == list(range(len(ROWS)))


def test_restricted_unpickler_refuses_other_globals(tmp_path):
    payload = pickle.dumps(pd.Timestamp("2020-01-01"))
    with pytest.raises(pickle.UnpicklingError):
        tcir._NumpyOnlyUnpickler(__import__("io").BytesIO(payload)).load()


def test_reads_only_ir1(fixture_h5):
    path, matrix = fixture_h5
    for start, frames in tcir.iter_ir1(path, block=5):
        assert frames.shape[1:] == (201, 201)
        np.testing.assert_array_equal(frames, matrix[start : start + len(frames), :, :, 0])


def test_manifest_columns_exclusions_and_no_split(fixture_h5, tmp_path):
    path, matrix = fixture_h5
    manifest, excluded, pixel = tcir.build_manifest([path], tmp_path / "out", block=4)
    assert "split" not in manifest.columns
    assert "source_split" not in manifest.columns
    for column in ("storm_id", "timestamp", "wind_speed_kt", "basin", "lat", "lon", "hemisphere"):
        assert manifest[column].notna().all()
    assert set(manifest["source"]) == {"tcir"}
    assert set(manifest["wind_averaging"]) == {"1min"}
    assert not manifest["is_synthetic"].any()
    assert set(manifest["channel"]) == {"IR1"} and set(manifest["channel_index"]) == {0}
    assert manifest.loc[0, "timestamp"] == "2003-05-01T00:00:00Z"
    assert manifest.loc[manifest["basin"] == "ATLN", "lon"].iloc[0] == pytest.approx(-60.0)
    assert not {"MSLP", "R35_4qAVG", "mslp_hpa"} & set(manifest.columns)
    documented = manifest.set_index("basin")["label_source_documented"]
    assert bool(documented["WPAC"].all()) and not bool(documented["IO"])
    # exclusions: all-NaN, >30% NaN, out-of-range value
    assert excluded["h5_index"].tolist() == [8, 9, 10]
    assert excluded["reason"].str.startswith("ir1 all NaN").sum() == 1
    assert excluded["reason"].str.contains("NaN fraction").sum() == 1
    assert excluded["reason"].str.contains("outside").sum() == 1
    # VIS NaNs and WV/PMW values in other channels do not affect IR1 gating
    assert len(manifest) == len(ROWS) - 3
    kept_nan = manifest.loc[manifest["h5_index"] == 11, "ir1_nan_fraction"].iloc[0]
    assert 0 < kept_nan < 0.001
    assert pixel["kept_frames_with_some_nan"] == 1
    assert 150 <= pixel["min"] and pixel["max"] <= 340
    saved = np.load(tmp_path / "out" / manifest.loc[0, "image_path"])
    assert saved.shape == (201, 201)
    np.testing.assert_array_equal(saved, matrix[0, :, :, 0])


def test_southern_hemisphere_not_flipped_by_default(fixture_h5, tmp_path):
    path, matrix = fixture_h5
    manifest, _, _ = tcir.build_manifest([path], tmp_path / "out")
    south = manifest[manifest["hemisphere"] == "S"]
    assert len(south) == 2 and set(south["basin"]) == {"SH"}
    assert not manifest["flipped"].any()
    for row in south.itertuples():
        saved = np.load(tmp_path / "out" / row.image_path)
        np.testing.assert_array_equal(saved, matrix[row.h5_index, :, :, 0])


def test_flip_is_explicit_opt_in(fixture_h5, tmp_path):
    path, matrix = fixture_h5
    manifest, _, _ = tcir.build_manifest([path], tmp_path / "out", flip_southern=True)
    assert manifest["flipped"].tolist() == (manifest["hemisphere"] == "S").tolist()
    for row in manifest.itertuples():
        saved = np.load(tmp_path / "out" / row.image_path)
        expected = matrix[row.h5_index, :, :, 0]
        expected = expected[::-1, :] if row.hemisphere == "S" else expected
        np.testing.assert_array_equal(saved, expected)


def test_orient_leaves_north_alone():
    frame = np.arange(9, dtype=np.float32).reshape(3, 3)
    same, flipped = tcir.orient(frame, "N", flip_southern=True)
    assert not flipped and same is frame
    same, flipped = tcir.orient(frame, "S")
    assert not flipped and same is frame


def test_storm_wise_split_has_no_leakage(fixture_h5, tmp_path):
    path, _ = fixture_h5
    manifest, _, _ = tcir.build_manifest([path], tmp_path / "out", export=False)
    assert manifest["image_path"].str.contains("#").all()
    assigned = assign_storms(manifest, "storm_wise", seed=0)
    parts = [assigned[assigned["split"] == name] for name in ("train", "val", "test")]
    assert_no_group_leakage(*parts)
    assert tcir.storm_split_check(manifest)["leakage"] == "none"
    with pytest.raises(ValueError):
        tcir.storm_split_check(assigned)


def test_summary_counts(fixture_h5, tmp_path):
    path, _ = fixture_h5
    manifest, excluded, pixel = tcir.build_manifest([path], tmp_path / "out", export=False)
    summary = tcir.summarize(manifest, excluded, pixel)
    assert summary["rows"] == 9
    assert summary["by_hemisphere"]["S"]["rows"] == 2
    assert summary["excluded"]["count"] == 3
    assert summary["flipped_rows"] == 0
    assert summary["has_split_column"] is False
