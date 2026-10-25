"""Training loop. A smoke fit only shows that the software runs.

Hyperparameters come from the YAML config. ``pretrained: true`` downloads
ImageNet weights and is refused unless ``CLOUDIE_ALLOW_PRETRAINED=1``.
The wind loss is weighted Smooth L1. The class loss is auxiliary. A size
term is added only when ``r34_nmi`` and ``rmw_nmi`` exist as labels, and
those columns are targets, not inputs. ``model.name: convlstm`` uses four
frames from the same storm and the same split.
"""

from __future__ import annotations

import argparse
import copy
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

from cloudie.config import checkpoint_path, load_config, parse_image_size
from cloudie.data_loader import load_image, load_manifest, resolve_image
from cloudie.preprocessing import NormStats, fit_norm_stats, prepare_image, stack_prepared
from cloudie.schema import (
    SchemaError,
    parse_is_synthetic,
    recorded_wind_averaging,
    wind_to_imd5,
)
from cloudie.sequences import N_FRAMES, size_targets_present, stack_windows
from cloudie.splits import (
    LeakageError,
    add_previous_frame_id,
    assert_no_forbidden_inputs,
    assign_storms,
    null_cross_split_links,
    storm_ids_by_split,
)

SMOKE_BANNER = (
    "Checkpoint was fit on synthetic smoke data. "
    "Metrics from it are a software check. Do not report them as skill."
)
NOT_A_FORECAST = (
    "This checkpoint is not a forecast or a warning. "
    "Agreement with a best track is not ground truth."
)
class EarlyStopping:
    """Stop after ``patience`` validation checks without an improvement."""

    def __init__(self, patience: int) -> None:
        if patience < 1:
            raise ValueError("patience must be at least 1")
        self.patience = patience
        self.best = float("inf")
        self.waits = 0
        self.best_state: dict | None = None

    def update(self, metric: float, state: dict) -> bool:
        """Return True when training should stop. Lower metric is better."""
        if metric < self.best:
            self.best = float(metric)
            self.waits = 0
            self.best_state = copy.deepcopy(state)
            return False
        self.waits += 1
        return self.waits >= self.patience


def assert_model_inputs(names: list[str] | tuple[str, ...]) -> None:
    """Images are the model input. Wind, size, and class labels are not."""
    assert_no_forbidden_inputs(names, "forbidden model inputs")


def class_sample_weights(
    class_names: list[str | None], known: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-frequency weights fit on this list only.

    ``class_weight`` lines up with ``known``. A class that does not appear
    gets weight 0. A row with no project class gets sample weight 1.
    """
    counts = np.array(
        [sum(name == item for item in class_names) for name in known], dtype=np.float64
    )
    present = counts > 0
    class_weight = np.zeros(len(known), dtype=np.float64)
    if present.any():
        class_weight[present] = counts[present].sum() / (present.sum() * counts[present])
    index = {name: position for position, name in enumerate(known)}
    sample = np.ones(len(class_names), dtype=np.float64)
    for row, name in enumerate(class_names):
        if name in index and class_weight[index[name]] > 0:
            sample[row] = class_weight[index[name]]
    return sample, class_weight


def project_labels(frame: pd.DataFrame, config: dict) -> list[str | None]:
    """Project class names, only when one wind-averaging period is recorded.

    These names are the project's grouping, not official IMD classes.
    A missing or mixed period raises instead of leaving the names blank.
    """
    recorded_wind_averaging(frame)
    edges = tuple(float(edge) for edge in config["data"]["imd5_edges_kt"])
    names = tuple(str(name) for name in config["data"]["imd5_names"])
    return [
        wind_to_imd5(float(wind), edges, names) for wind in frame["wind_speed_kt"]
    ]


def manifest_is_synthetic(frame: pd.DataFrame) -> bool:
    """True when any row is explicitly synthetic.

    A missing column or a null flag raises. Null is not treated as real,
    and one synthetic row keeps the whole manifest marked synthetic.
    """
    if "is_synthetic" not in frame.columns:
        raise SchemaError(
            "is_synthetic is required so a synthetic row cannot be read as real"
        )
    flags = [parse_is_synthetic(value) for value in frame["is_synthetic"].tolist()]
    return any(flags)


_TRAIN_KEYS = (
    "batch_size",
    "epochs",
    "lr",
    "weight_decay",
    "patience",
    "class_loss_weight",
    "size_loss_weight",
)


def _require_train_block(config: dict) -> dict:
    train_cfg = config.get("train")
    if not isinstance(train_cfg, dict):
        raise ValueError("config train is required")
    missing = [key for key in _TRAIN_KEYS if key not in train_cfg]
    if missing:
        raise ValueError(f"config train missing {missing}")
    return train_cfg


def assert_normalize(config: dict) -> None:
    """The deep path standardizes with train-split mean and std."""
    mode = config["data"]["normalize"]
    if str(mode) != "zscore":
        raise ValueError("data.normalize must be zscore")


def recorded_wind_period(frame: pd.DataFrame) -> str | None:
    """The single ``wind_averaging`` value on a manifest, if the column exists."""
    if "wind_averaging" not in frame.columns:
        return None
    periods = sorted({str(value) for value in frame["wind_averaging"].dropna().unique()})
    if len(periods) != 1:
        return None
    return periods[0]


def assert_wind_averaging(config: dict, frame: pd.DataFrame) -> str | None:
    """Require ``data.wind_averaging`` and refuse a manifest that disagrees.

    A manifest with no period column does not inherit the config value.
    Project class names stay off until the column records one period.
    """
    declared = config["data"]["wind_averaging"]
    if declared is None or str(declared).strip() == "":
        raise ValueError("config data.wind_averaging is required")
    declared_text = str(declared)
    observed = recorded_wind_period(frame)
    if observed is not None and observed != declared_text:
        raise ValueError(
            f"manifest wind_averaging is {observed} but config data.wind_averaging "
            f"is {declared_text}. Do not mix averaging periods."
        )
    return observed


def split_frame(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    split_cfg = config["split"]
    holdout_years = split_cfg["holdout_years"] or []
    holdout_basin = split_cfg["holdout_basin"] or None
    assigned = assign_storms(
        frame,
        str(split_cfg["strategy"]),
        test_fraction=float(split_cfg["test_fraction"]),
        val_fraction=float(split_cfg["val_fraction"]),
        seed=int(split_cfg["seed"]),
        holdout_years=[int(year) for year in holdout_years],
        holdout_basin=None if holdout_basin in (None, "null", "") else str(holdout_basin),
    )
    if "prev_sample_id" not in assigned.columns:
        assigned = add_previous_frame_id(assigned)
    assigned, _dropped = null_cross_split_links(assigned)
    return assigned


def stack_split(
    frame: pd.DataFrame,
    manifest_path: Path,
    split_name: str,
    image_size: int | str | None,
    stats: NormStats | None,
    max_missing_fraction: float,
) -> tuple[np.ndarray, np.ndarray, list[str | None]]:
    rows = frame[frame["split"] == split_name].reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"split {split_name} has no rows")
    labels = [None if pd.isna(value) else str(value) for value in rows["_class"]]
    images = []
    winds = []
    for row in rows.itertuples(index=False):
        path = resolve_image(manifest_path, row.image_path)
        images.append(
            prepare_image(load_image(path), image_size, stats, max_missing_fraction)
        )
        winds.append(float(row.wind_speed_kt))
    return stack_prepared(images), np.asarray(winds, dtype=np.float32), labels


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch = _torch()
    torch.manual_seed(seed)


def _torch():
    import torch

    return torch


def uses_sequences(config: dict) -> bool:
    return str(config["model"]["name"]) == "convlstm"


def train_size_head(frame: pd.DataFrame) -> bool:
    """Turn the size head on only when the train split has size labels."""
    rows = frame[frame["split"] == "train"] if "split" in frame.columns else frame
    return size_targets_present(rows)


def model_n_frames(config: dict) -> int:
    if not uses_sequences(config):
        return 1
    requested = int(config["model"]["n_frames"])
    if requested != N_FRAMES:
        raise ValueError(f"ConvLSTM uses {N_FRAMES} frames")
    return requested


def build_backbone(config: dict):
    """Build the image backbone. Refuses an ImageNet download by default."""
    from cloudie.model import ConvLSTMRegressor, ResNetRegressor, SimpleCNN

    model_cfg = config["model"]
    pretrained = bool(model_cfg["pretrained"])
    if pretrained and os.environ.get("CLOUDIE_ALLOW_PRETRAINED") != "1":
        raise RuntimeError(
            "pretrained: true downloads ImageNet weights and must not run in CI. "
            "Set CLOUDIE_ALLOW_PRETRAINED=1 only for a local run that intends the download."
        )
    name = str(model_cfg["name"])
    dropout = float(model_cfg["dropout"])
    channels = int(config["data"]["channels"])
    if name == "simple_cnn":
        return SimpleCNN(in_channels=channels, dropout=dropout)
    if name == "resnet18":
        return ResNetRegressor(in_channels=channels, pretrained=pretrained, dropout=dropout)
    if name == "convlstm":
        return ConvLSTMRegressor(
            in_channels=channels,
            hidden=int(model_cfg["hidden"]),
            n_frames=model_n_frames(config),
            dropout=dropout,
        )
    raise ValueError(f"unknown model {name}")


def build_net(config: dict, size_head: bool = False):
    """Wind and class heads. The size head exists only when labels do.

    The size head reads image features. It does not take wind speed.
    """
    torch = _torch()
    from torch import nn

    backbone = build_backbone(config)
    n_classes = len(config["data"]["imd5_names"])

    class _Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.class_head = nn.Linear(int(backbone.feature_dim), n_classes)
            self.size_head = (
                nn.Linear(int(backbone.feature_dim), 2) if size_head else None
            )

        def forward(self, inputs: torch.Tensor):
            features = self.backbone.encode(inputs)
            wind = self.backbone.head(features)
            logits = self.class_head(features)
            if self.size_head is None:
                return wind, logits
            return wind, logits, self.size_head(features)

    return _Net()


def _wind_loss(prediction, target, sample_weight):
    import torch.nn.functional as F

    per_row = F.smooth_l1_loss(prediction.view(-1), target.view(-1), reduction="none")
    if sample_weight is None:
        return per_row.mean()
    weight = sample_weight.view(-1)
    return (per_row * weight).sum() / weight.sum().clamp_min(1e-8)


def _class_loss(logits, class_index, class_weight):
    import torch.nn.functional as F

    mask = class_index >= 0
    if int(mask.sum()) == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(logits[mask], class_index[mask], weight=class_weight)


def _size_loss(prediction, target):
    """Smooth L1 on finite R34 and RMW pairs. No term when none are finite."""
    import torch
    import torch.nn.functional as F

    finite = torch.isfinite(target).all(dim=1)
    if int(finite.sum()) == 0:
        return prediction.sum() * 0.0
    return F.smooth_l1_loss(prediction[finite], target[finite])


def _split_output(output):
    if isinstance(output, tuple) and len(output) == 3:
        return output[0], output[1], output[2]
    wind, logits = output
    return wind, logits, None


def _run_epoch(
    model,
    loader,
    optimizer,
    class_weight,
    class_loss_weight: float,
    train: bool,
    size_loss_weight: float = 0.0,
):
    torch = _torch()
    model.train(train)
    total = 0.0
    count = 0
    for batch in loader:
        images, winds, class_index, sample_weight = batch[0], batch[1], batch[2], batch[3]
        size_target = batch[4] if len(batch) > 4 else None
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            wind_hat, logits, size_hat = _split_output(model(images))
            monitor = _wind_loss(wind_hat, winds, None)
            loss = _wind_loss(wind_hat, winds, sample_weight) if train else monitor
            if train and class_loss_weight > 0:
                loss = loss + class_loss_weight * _class_loss(logits, class_index, class_weight)
            if (
                train
                and size_hat is not None
                and size_target is not None
                and size_loss_weight > 0
            ):
                loss = loss + size_loss_weight * _size_loss(size_hat, size_target)
            if train:
                loss.backward()
                optimizer.step()
        batch_rows = int(images.shape[0])
        total += float(monitor.detach()) * batch_rows
        count += batch_rows
    if count == 0:
        raise ValueError("loader produced no rows")
    return total / count


def _dataset(
    images,
    winds,
    labels,
    known: list[str],
    sample_weight: np.ndarray,
    size: np.ndarray | None = None,
):
    torch = _torch()
    from torch.utils.data import Dataset

    index = {name: position for position, name in enumerate(known)}
    class_index = [index.get(label, -1) if label is not None else -1 for label in labels]

    class _Frames(Dataset):
        def __init__(self) -> None:
            self.images = torch.from_numpy(np.ascontiguousarray(images))
            self.winds = torch.from_numpy(np.asarray(winds, dtype=np.float32)).view(-1, 1)
            self.class_index = torch.tensor(class_index, dtype=torch.long)
            self.sample_weight = torch.tensor(sample_weight, dtype=torch.float32)
            self.size = None if size is None else torch.tensor(np.asarray(size), dtype=torch.float32)

        def __len__(self) -> int:
            return int(self.images.shape[0])

        def __getitem__(self, item: int):
            row = (
                self.images[item],
                self.winds[item],
                self.class_index[item],
                self.sample_weight[item],
            )
            if self.size is None:
                return row
            return row + (self.size[item],)

    return _Frames()


def _aligned_size(frame: pd.DataFrame, split_name: str) -> np.ndarray:
    rows = frame[frame["split"] == split_name].reset_index(drop=True)
    values = rows[["r34_nmi", "rmw_nmi"]].apply(pd.to_numeric, errors="coerce")
    return values.to_numpy(dtype=np.float32)


def _load_examples(
    config: dict,
    frame: pd.DataFrame,
    manifest: Path,
    split_name: str,
    stats: NormStats | None,
    size_head: bool,
):
    image_size = parse_image_size(config["data"]["image_size"])
    max_missing = float(config["data"]["max_missing_fraction"])
    if uses_sequences(config):
        images, winds, labels, size = stack_windows(
            frame,
            manifest,
            split_name,
            image_size,
            stats,
            max_missing,
            model_n_frames(config),
        )
        return images, winds, labels, size if size_head else None
    images, winds, labels = stack_split(
        frame, manifest, split_name, image_size, stats, max_missing
    )
    size = _aligned_size(frame, split_name) if size_head else None
    return images, winds, labels, size


def fit_split_normalization(
    frame: pd.DataFrame, manifest: str | Path, config: dict
) -> NormStats:
    """Fit pixel mean and std on the train split only.

    The images are the train split. ``data.image_size`` comes from config.
    ``native`` keeps each chip's height and width.
    A later held-out coverage check reuses ``storm_ids_by_split`` and these
    stored stats.
    """
    groups = storm_ids_by_split(frame)
    if not groups["train"]:
        raise LeakageError("normalization stats may be fit on the train split only")
    raw_images, _winds, _labels, _size = _load_examples(
        config, frame, Path(manifest), "train", None, False
    )
    if uses_sequences(config):
        norm_frames = [
            raw_images[index, step]
            for index in range(raw_images.shape[0])
            for step in range(raw_images.shape[1])
        ]
    else:
        norm_frames = list(raw_images)
    return fit_norm_stats(norm_frames, "train")


def train(config: dict, manifest_path: str | Path | None = None) -> Path:
    """Fit on the train split and write the checkpoint named in ``paths``."""
    assert_model_inputs(("image",))
    train_cfg = _require_train_block(config)
    assert_normalize(config)
    manifest = Path(manifest_path or config["paths"]["manifest"])
    frame = load_manifest(manifest)
    wind_period = assert_wind_averaging(config, frame)
    torch = _torch()
    from torch.utils.data import DataLoader

    seed = int(config["project"]["seed"])
    _set_seed(seed)
    synthetic = manifest_is_synthetic(frame)
    assigned = split_frame(frame, config)
    assigned["_class"] = project_labels(assigned, config)
    image_size = parse_image_size(config["data"]["image_size"])
    max_missing = float(config["data"]["max_missing_fraction"])
    # Size labels are a training-fold decision, same as class weights.
    size_head = train_size_head(assigned)
    n_frames = model_n_frames(config)
    stats = fit_split_normalization(assigned, manifest, config)
    train_images, train_winds, train_labels, train_size = _load_examples(
        config, assigned, manifest, "train", stats, size_head
    )
    val_images, val_winds, val_labels, val_size = _load_examples(
        config, assigned, manifest, "val", stats, size_head
    )
    known = [str(name) for name in config["data"]["imd5_names"]]
    sample_weight, class_weight_np = class_sample_weights(train_labels, known)
    # Validation uses weight 1. The class weights stay the train fold's.
    val_weight = np.ones(len(val_labels), dtype=np.float64)
    batch_size = int(train_cfg["batch_size"])
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        _dataset(train_images, train_winds, train_labels, known, sample_weight, train_size),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        _dataset(val_images, val_winds, val_labels, known, val_weight, val_size),
        batch_size=batch_size,
        shuffle=False,
    )
    model = build_net(config, size_head=size_head)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    class_weight = torch.tensor(class_weight_np, dtype=torch.float32)
    class_loss_weight = float(train_cfg["class_loss_weight"])
    size_loss_weight = float(train_cfg["size_loss_weight"]) if size_head else 0.0
    stopper = EarlyStopping(int(train_cfg["patience"]))
    epochs = int(train_cfg["epochs"])
    last_epoch = 0
    for epoch in range(epochs):
        last_epoch = epoch
        _run_epoch(
            model,
            train_loader,
            optimizer,
            class_weight,
            class_loss_weight,
            True,
            size_loss_weight,
        )
        val_loss = _run_epoch(
            model,
            val_loader,
            optimizer,
            class_weight,
            class_loss_weight,
            False,
            size_loss_weight,
        )
        if not np.isfinite(val_loss):
            raise RuntimeError("validation loss is not finite")
        if stopper.update(val_loss, model.state_dict()):
            break
    if stopper.best_state is None:
        raise RuntimeError("training produced no validation score")
    model.load_state_dict(stopper.best_state)
    checkpoint = checkpoint_path(config)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "model_name": str(config["model"]["name"]),
        "dropout": float(config["model"]["dropout"]),
        "pretrained": False,
        "in_channels": int(config["data"]["channels"]),
        "n_classes": len(known),
        "class_names": known,
        "edges_kt": [float(edge) for edge in config["data"]["imd5_edges_kt"]],
        "norm_mean": stats.mean,
        "norm_std": stats.std,
        "norm_source_split": stats.source_split,
        "image_size": image_size,
        "max_missing_fraction": max_missing,
        "mean_baseline_kt": float(
            np.mean(
                assigned.loc[assigned["split"] == "train", "wind_speed_kt"].to_numpy(
                    dtype=np.float64
                )
            )
        ),
        "is_synthetic": synthetic,
        "wind_averaging": wind_period,
        "wind_averaging": wind_period,
        "split": {
            "strategy": str(config["split"]["strategy"]),
            "seed": int(config["split"]["seed"]),
            "test_fraction": float(config["split"]["test_fraction"]),
            "val_fraction": float(config["split"]["val_fraction"]),
        },
        "best_val_loss": stopper.best,
        "epoch": last_epoch,
        "present_as_forecast": False,
        "honesty": SMOKE_BANNER if synthetic else NOT_A_FORECAST,
        "n_frames": n_frames,
        "size_head": size_head,
        "hidden": int(config["model"]["hidden"]) if uses_sequences(config) else None,
    }
    torch.save(payload, checkpoint)
    if synthetic:
        print(SMOKE_BANNER)
    else:
        print(NOT_A_FORECAST)
    print(f"Wrote {checkpoint}")
    return checkpoint


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    return train(config, args.manifest)


if __name__ == "__main__":
    main()
