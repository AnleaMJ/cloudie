"""Both YAML configs name every path and hyperparameter the code reads."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_NAMES = ("config.yaml", "config_smoke.yaml")
LOADERS = ("src.config",)
PROJECT_EDGES = [17.0, 34.0, 48.0, 64.0, 91.0]
REQUIRED = {
    "project": ["name", "seed", "honesty"],
    "paths": ["manifest", "artifacts", "reports", "checkpoint_name"],
    "data": [
        "image_size",
        "channels",
        "normalize",
        "max_missing_fraction",
        "plausible_wind_kt",
        "wind_averaging",
        "imd5_edges_kt",
        "imd5_names",
    ],
    "split": [
        "strategy",
        "test_fraction",
        "val_fraction",
        "holdout_years",
        "holdout_basin",
        "seed",
    ],
    "model": ["name", "pretrained", "dropout", "mc_samples", "n_frames", "hidden"],
    "train": [
        "batch_size",
        "epochs",
        "lr",
        "weight_decay",
        "patience",
        "class_loss_weight",
        "size_loss_weight",
    ],
    "predict": ["advisory_jump_kt"],
    "features": [
        "calibrated",
        "cold_thresholds",
        "radial_bins",
        "eye_radius_fraction",
        "cdo_outer_fraction",
        "eye_contrast_min",
        "dav_radii_fractions",
        "dav_annulus_width_fraction",
        "glcm_levels",
        "glcm_distance",
        "selection",
    ],
    "baselines": [
        "n_splits",
        "seed",
        "regression_models",
        "classification_models",
        "use_smote",
        "ridge_alpha",
        "knn_neighbors",
        "svr_c",
        "svc_c",
        "rf_estimators",
        "rf_max_depth",
        "xgb_estimators",
        "xgb_max_depth",
        "xgb_learning_rate",
        "max_iter",
    ],
}
SELECTION_KEYS = [
    "correlation_threshold",
    "max_features",
    "method",
    "n_estimators",
    "seed",
]


def _load(loader: str, name: str) -> dict:
    module = importlib.import_module(loader)
    return module.load_config(ROOT / "config" / name)


@pytest.mark.parametrize("loader", LOADERS)
@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_configs_have_required_keys(loader: str, name: str) -> None:
    config = _load(loader, name)
    missing = []
    for section, keys in REQUIRED.items():
        block = config.get(section)
        if not isinstance(block, dict):
            missing.append(section)
            continue
        missing.extend(f"{section}.{key}" for key in keys if key not in block)
    if "forbidden_features" not in config or not config["forbidden_features"]:
        missing.append("forbidden_features")
    selection = config.get("features", {}).get("selection")
    if not isinstance(selection, dict):
        missing.append("features.selection")
    else:
        missing.extend(f"features.selection.{key}" for key in SELECTION_KEYS if key not in selection)
    assert missing == []
    assert config["split"]["strategy"] == "storm_wise"
    assert config["data"]["normalize"] == "zscore"
    assert [float(edge) for edge in config["data"]["imd5_edges_kt"]] == PROJECT_EDGES
    assert list(config["data"]["imd5_names"]) == ["D_DD", "CS", "SCS", "VSCS", "ESCS_SUCS"]
    checkpoint_name = str(config["paths"]["checkpoint_name"])
    assert checkpoint_name == Path(checkpoint_name).name
    assert checkpoint_name.endswith(".pt")


def test_nasa_config_keeps_native_chips_and_one_minute_winds() -> None:
    from src.config import checkpoint_path, parse_image_size

    config = _load("src.config", "config.yaml")
    assert parse_image_size(config["data"]["image_size"]) is None
    assert config["data"]["wind_averaging"] == "1min"
    assert checkpoint_path(config) == Path("artifacts") / "checkpoint.pt"
    assert config["data"]["plausible_wind_kt"] == [0, 200]


def test_smoke_config_stays_small_and_synthetic() -> None:
    from src.config import checkpoint_path, parse_image_size

    config = _load("src.config", "config_smoke.yaml")
    size = parse_image_size(config["data"]["image_size"])
    assert isinstance(size, int)
    assert 8 <= size <= 64
    assert config["data"]["wind_averaging"] == "synthetic"
    assert checkpoint_path(config) == Path("artifacts/smoke") / "checkpoint.pt"


def test_project_edges_match_the_schema() -> None:
    module = importlib.import_module("src.schema")
    assert list(module.DEFAULT_EDGES_KT) == PROJECT_EDGES
    assert module.DEFAULT_NAMES == ("D_DD", "CS", "SCS", "VSCS", "ESCS_SUCS")


def test_bins_are_documented_as_the_project_grouping() -> None:
    paths = [
        ROOT / "README.md",
        ROOT / "docs" / "DATA_SOURCES.md",
        ROOT / "docs" / "PROJECT_DOCUMENTATION.md",
        ROOT / "docs" / "AGENT_HANDOFF.md",
        ROOT / "src" / "schema.py",
        ROOT / "config" / "config.yaml",
        ROOT / "config" / "config_smoke.yaml",
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8").lower()
        assert "3-minute" in text, path
    nasa = (ROOT / "config" / "config.yaml").read_text(encoding="utf-8")
    assert "wind_averaging: 1min" in nasa
    assert "HURDAT2" in nasa
    assert "366" in nasa
    sources = (ROOT / "docs" / "DATA_SOURCES.md").read_text(encoding="utf-8")
    assert "1min" in sources
    assert "366" in sources
    assert "3-minute" in sources


def test_training_code_does_not_keep_hyperparameter_fallbacks() -> None:
    for relative in ("src/train.py", "cloudie/train.py"):
        train = (ROOT / relative).read_text(encoding="utf-8")
        for snippet in ("1e-3", 'get("batch_size"', 'get("epochs"', 'get("lr"', "checkpoint.pt"):
            assert snippet not in train, relative
    baselines = (ROOT / "src" / "baselines.py").read_text(encoding="utf-8")
    assert '"rf_estimators": 100' not in baselines
    assert '"xgb_estimators": 50' not in baselines
    assert 'get("test_fraction"' not in baselines


def test_ci_has_one_torch_job_and_does_not_download_imagery() -> None:
    workflow_path = ROOT / ".github" / "workflows" / "ci.yml"
    text = workflow_path.read_text(encoding="utf-8")
    workflow = yaml.safe_load(text)
    assert set(workflow["jobs"]) == {"test", "torch"}
    assert sum(1 for line in text.splitlines() if line.startswith("  torch:")) == 1
    assert text.count('python-version: "3.11"') == 2
    torch_job = yaml.dump(workflow["jobs"]["torch"])
    assert "streamlit" in torch_job
    for name in (
        "tests/test_model.py",
        "tests/test_sequences.py",
        "tests/test_train.py",
        "tests/test_predict.py",
    ):
        assert name in text
    lowered = text.lower()
    for token in ("load_dataset", "azcopy", "import torchgeo", "from torchgeo"):
        assert token not in lowered
    for folder in ("src", "cloudie"):
        for path in (ROOT / folder).rglob("*.py"):
            source = path.read_text(encoding="utf-8").lower()
            for token in ("load_dataset", "azcopy", "import torchgeo", "from torchgeo"):
                assert token not in source, path
