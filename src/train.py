"""Training loop. A smoke fit only shows that the software runs.

Hyperparameters come from the YAML config. ``pretrained: true`` downloads
ImageNet weights and is refused unless ``CLOUDIE_ALLOW_PRETRAINED=1``.
The wind loss is weighted Smooth L1. The class loss is auxiliary. A size
term is added only when ``r34_nmi`` and ``rmw_nmi`` exist as labels, and
those columns are targets, not inputs. ``model.name: convlstm`` uses four
frames from the same storm and the same split.

``python -m src.train`` picks xpu, then cuda, then cpu. XPU and CUDA both
need ``CLOUDIE_ALLOW_GPU=1``. Normalization is fit per dataset on train
storms only. ``--init-from`` loads weights for a later dataset and refits
those stats. ``--resume`` continues the last finished epoch.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import checkpoint_path, load_config, parse_image_size
from src.data_loader import apply_frame_cache, load_image, load_manifest, resolve_image
from src.device import (
    amp_policy_for,
    autocast_context,
    compare_model_parity,
    make_grad_scaler,
    peak_memory_bytes,
    reset_peak_memory,
    resolve_device,
    seeded_parity_batch,
    xpu_device_properties,
)
from src.preprocessing import NormStats, fit_norm_stats, norm_by_dataset, prepare_image, stack_prepared
from src.schema import (
    SchemaError,
    parse_is_synthetic,
    recorded_wind_averaging,
    split_wind_averaging,
    wind_averaging_seconds,
    wind_to_imd5,
)
from src.sequences import N_FRAMES, size_targets_present, stack_windows
from src.splits import (
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
    "grad_accum_steps",
    "num_workers",
    "device_parity_tol",
)

LAST_CHECKPOINT = "last.pt"


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


def build_loader(dataset, *, batch_size: int, shuffle: bool, num_workers: int, generator=None):
    """DataLoader with ``persistent_workers`` whenever workers are used.

    ``num_workers`` comes from ``train.num_workers``. Zero workers load in
    the parent process and do not set ``persistent_workers``.
    """
    from torch.utils.data import DataLoader

    workers = int(num_workers)
    if workers < 0:
        raise ValueError("train.num_workers must be >= 0")
    kwargs = {
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": workers,
    }
    if generator is not None:
        kwargs["generator"] = generator
    if workers > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(dataset, **kwargs)


class ArrayFrames:
    """In-memory frames. Module-level so Windows workers can unpickle it."""

    def __init__(self, images, winds, labels, known: list[str], sample_weight, size=None) -> None:
        torch = _torch()
        index = {name: position for position, name in enumerate(known)}
        class_index = [index.get(label, -1) if label is not None else -1 for label in labels]
        self.images = torch.from_numpy(np.ascontiguousarray(images))
        self.winds = torch.from_numpy(np.asarray(winds, dtype=np.float32)).view(-1, 1)
        self.class_index = torch.tensor(class_index, dtype=torch.long)
        self.sample_weight = torch.tensor(np.asarray(sample_weight), dtype=torch.float32)
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


class PathFrames:
    """Load one frame per item so DataLoader workers stay on small files.

    A cache path is an fp16 ``.npy``. A source path is whatever the
    manifest already pointed at. Normalization uses the train-split stats
    for this dataset only.
    """

    def __init__(
        self,
        records: list[dict],
        image_size: int | None,
        mean: float | None,
        std: float | None,
        max_missing: float,
        known: list[str],
    ) -> None:
        self.records = list(records)
        self.image_size = image_size
        self.mean = None if mean is None else float(mean)
        self.std = None if std is None else float(std)
        self.max_missing = float(max_missing)
        self.known = list(known)
        self._index = {name: position for position, name in enumerate(known)}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, item: int):
        torch = _torch()
        row = self.records[item]
        stats = None
        if self.mean is not None and self.std is not None:
            stats = NormStats(mean=self.mean, std=self.std, source_split="train")
        image = prepare_image(load_image(row["path"]), self.image_size, stats, self.max_missing)
        label = row["label"]
        class_index = self._index.get(label, -1) if label is not None else -1
        return (
            torch.from_numpy(np.ascontiguousarray(image)),
            torch.tensor([float(row["wind"])], dtype=torch.float32),
            torch.tensor(class_index, dtype=torch.long),
            torch.tensor(float(row["weight"]), dtype=torch.float32),
        )


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
    from src.model import ConvLSTMRegressor, ResNetRegressor, SimpleCNN

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
    grad_accum_steps: int = 1,
):
    torch = _torch()
    accum = int(grad_accum_steps)
    if accum < 1:
        raise ValueError("train.grad_accum_steps must be at least 1")
    model.train(train)
    total = 0.0
    count = 0
    pending = 0
    if train:
        optimizer.zero_grad(set_to_none=True)
    for batch in loader:
        images, winds, class_index, sample_weight = batch[0], batch[1], batch[2], batch[3]
        size_target = batch[4] if len(batch) > 4 else None
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
                (loss / accum).backward()
                pending += 1
                if pending == accum:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    pending = 0
        batch_rows = int(images.shape[0])
        total += float(monitor.detach()) * batch_rows
        count += batch_rows
    if train and pending:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
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
    return ArrayFrames(images, winds, labels, known, sample_weight, size)


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
    workers = int(train_cfg["num_workers"])
    grad_accum_steps = int(train_cfg["grad_accum_steps"])
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = build_loader(
        _dataset(train_images, train_winds, train_labels, known, sample_weight, train_size),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        generator=generator,
    )
    val_loader = build_loader(
        _dataset(val_images, val_winds, val_labels, known, val_weight, val_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
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
            grad_accum_steps,
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
    dataset_name = str(config["data"]["dataset"])
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
        "norm_by_dataset": norm_by_dataset(dataset_name, stats),
        "dataset": dataset_name,
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
        "wind_averaging_s": wind_averaging_seconds(wind_period),
        "wind_averaging_by_split": split_wind_averaging(assigned),
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


LOGGER = logging.getLogger("src.train")

# CLI names. simplecnn is the YAML name simple_cnn. ConvLSTM stays on train().
_CLI_MODELS = {"simplecnn": "simple_cnn", "simple_cnn": "simple_cnn", "resnet18": "resnet18"}
_DATASETS = ("nasa", "tcir", "smoke")
RUN_FILES = (
    "checkpoint",
    "resolved_config.yaml",
    "split_manifest.csv",
    "loss_curve.csv",
    "loss_curve.png",
    "metrics.json",
)


def canonical_model_name(name: str) -> str:
    """Map the CLI model name onto ``model.name`` in the YAML."""
    key = str(name).strip().lower()
    if key not in _CLI_MODELS:
        raise ValueError("model must be simplecnn or resnet18")
    return _CLI_MODELS[key]


def refuse_mixed_resolution(frame: pd.DataFrame, dataset: str) -> None:
    """One dataset, one source resolution. NASA and TCIR stay in separate runs."""
    if frame.empty:
        raise ValueError(f"{dataset} frame has no rows")
    sizes: set[int] = set()
    if "orig_size" in frame.columns:
        sizes = {int(value) for value in frame["orig_size"].tolist()}
    elif {"native_height", "native_width"} <= set(frame.columns):
        sizes = {
            int(height) if int(height) == int(width) else -1
            for height, width in zip(frame["native_height"], frame["native_width"], strict=True)
        }
    if len(sizes) > 1:
        raise ValueError(
            f"refusing a {dataset} batch that mixes resolutions {sorted(sizes)}. "
            "NASA and TCIR runs stay separate."
        )
    if dataset == "nasa" and sizes:
        from src.nasa_tc import WORKING_SIZE

        if sizes != {int(WORKING_SIZE)}:
            raise ValueError(
                f"NASA phase 1 keeps only {int(WORKING_SIZE)}x{int(WORKING_SIZE)} frames, got {sorted(sizes)}"
            )
    if dataset == "tcir" and sizes:
        from src.tcir import IMAGE_SIZE

        if sizes != {int(IMAGE_SIZE)}:
            raise ValueError(f"TCIR frames must stay {int(IMAGE_SIZE)}x{int(IMAGE_SIZE)}, got {sorted(sizes)}")


def _log_excluded(dataset: str, count: int, detail: str) -> None:
    message = f"{dataset} excluded {int(count)} frames ({detail})"
    LOGGER.info(message)
    print(message)


def _absolute_image_paths(frame: pd.DataFrame, base: Path) -> pd.DataFrame:
    out = frame.copy()
    paths: list[str] = []
    for image_path in out["image_path"].tolist():
        path = Path(str(image_path))
        if not path.is_absolute():
            path = base / path
        paths.append(path.resolve().as_posix())
    out["image_path"] = paths
    return out


def load_nasa_frame(config: dict) -> tuple[pd.DataFrame, int]:
    """NASA manifest for both official sides. ``source_split`` is not the split.

    Phase 1 keeps ``orig_size`` 366 only. Colour frames and any other size
    are excluded, and the excluded count is logged. Nothing is downloaded.
    """
    from src.nasa_tc import WORKING_SIZE, build_manifest, csv_tables, exclusions_for, split_exclusions
    from src.schema import require_valid

    root = Path(config["paths"]["nasa_root"])
    if not root.is_dir():
        raise FileNotFoundError(
            f"NASA root {root} is missing. Set paths.nasa_root in the YAML. "
            "This entrypoint does not download imagery."
        )
    raw = build_manifest(root, root, tables=csv_tables, keep_excluded=True)
    colour = exclusions_for(raw)
    kept, _dropped = split_exclusions(raw)
    if "orig_size" not in kept.columns:
        raise ValueError("NASA manifest is missing orig_size")
    if "split" in kept.columns:
        raise LeakageError("NASA manifest must not carry split; source_split holds the official side")
    size_ok = kept["orig_size"].astype(int) == int(WORKING_SIZE)
    size_excluded = int((~size_ok).sum())
    frame = require_valid(kept.loc[size_ok].reset_index(drop=True))
    excluded = int(len(colour)) + size_excluded
    _log_excluded(
        "nasa",
        excluded,
        f"{int(len(colour))} colour, {size_excluded} not {int(WORKING_SIZE)}x{int(WORKING_SIZE)}",
    )
    if "source_split" not in frame.columns:
        raise LeakageError("NASA manifest must keep the official side in source_split")
    refuse_mixed_resolution(frame, "nasa")
    return _absolute_image_paths(frame, root), excluded


def load_tcir_frame(config: dict) -> tuple[pd.DataFrame, int]:
    """TCIR IR1 manifest. There is no official split inside the HDF5 files.

    ``assign_storms`` writes ``split``. Excluded IR1 frames are logged.
    This entrypoint does not download TCIR.
    """
    from src.tcir import IMAGE_SIZE, build_manifest

    files = config["paths"]["tcir_files"]
    if not isinstance(files, list) or not files:
        raise ValueError(
            "paths.tcir_files must list TCIR HDF5 paths. This entrypoint does not download TCIR."
        )
    manifest_dir = Path(config["paths"]["tcir_manifest_dir"])
    frame, excluded, _pixel = build_manifest(
        list(files), manifest_dir, export=True, flip_southern=False
    )
    if "split" in frame.columns or "source_split" in frame.columns:
        raise LeakageError(
            "TCIR has no official split. Keep any official label in source_split only, "
            "and let assign_storms write split."
        )
    refuse_mixed_resolution(frame, "tcir")
    sizes = {int(value) for value in frame["orig_size"].tolist()}
    if sizes != {int(IMAGE_SIZE)}:
        raise ValueError(f"TCIR frames must stay {int(IMAGE_SIZE)}x{int(IMAGE_SIZE)}, got {sorted(sizes)}")
    _log_excluded("tcir", int(len(excluded)), "IR1 gate")
    return _absolute_image_paths(frame, manifest_dir), int(len(excluded))


def load_smoke_frame(config: dict) -> tuple[pd.DataFrame, int]:
    """Synthetic manifest named by ``paths.manifest``. Not a result."""
    path = Path(config["paths"]["manifest"])
    if not path.is_file():
        raise FileNotFoundError(
            f"smoke manifest {path} is missing. Run python -m src.make_smoke_data. "
            "This entrypoint does not download imagery."
        )
    frame = load_manifest(path)
    if "split" in frame.columns:
        frame = frame.drop(columns=["split"])
    if "orig_size" in frame.columns or "native_height" in frame.columns:
        refuse_mixed_resolution(frame, "smoke")
    _log_excluded("smoke", 0, "synthetic manifest is already one size")
    return _absolute_image_paths(frame, path.parent), 0


def load_training_frame(dataset: str, config: dict) -> tuple[pd.DataFrame, int]:
    """Load one dataset. NASA and TCIR are never concatenated."""
    name = str(dataset).strip().lower()
    if name not in _DATASETS:
        raise ValueError(f"dataset must be one of {', '.join(_DATASETS)}")
    if name == "nasa":
        return load_nasa_frame(config)
    if name == "tcir":
        return load_tcir_frame(config)
    return load_smoke_frame(config)


def assign_project_split(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Storm-wise split over the whole frame. ``source_split`` is not used.

    A pre-existing ``split`` column is dropped. Official NASA sides stay in
    ``source_split``. A storm that sits on both official sides gets one
    project split.
    """
    work = frame.copy()
    if "split" in work.columns:
        work = work.drop(columns=["split"])
    assigned = split_frame(work, config)
    if "source_split" in assigned.columns:
        per_storm = assigned.groupby(assigned["storm_id"].astype(str))["split"].nunique()
        if bool((per_storm > 1).any()):
            raise LeakageError("source_split leaked into the project split and cut a storm")
        both = assigned.groupby(assigned["storm_id"].astype(str))["source_split"].nunique()
        shared = both[both > 1]
        for storm_id in shared.index:
            rows = assigned.loc[assigned["storm_id"].astype(str) == str(storm_id), "split"]
            if int(rows.nunique()) != 1:
                raise LeakageError(f"source_split leaked into the project split for {storm_id}")
    groups = storm_ids_by_split(assigned)
    if groups["train"] & groups["val"] or groups["train"] & groups["test"] or groups["val"] & groups["test"]:
        raise LeakageError("storm overlap across splits")
    return assigned


def _move_batch(batch, device):
    images = batch[0].to(device)
    winds = batch[1].to(device)
    class_index = batch[2].to(device)
    sample_weight = batch[3].to(device)
    size_target = batch[4].to(device) if len(batch) > 4 else None
    return images, winds, class_index, sample_weight, size_target


def _split_mae(model, loader, device, policy) -> float:
    """Validation MAE in knots. Dropout is off. This is the early-stopping score."""
    torch = _torch()
    was_training = model.training
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            images, winds, _class_index, _sample_weight, _size = _move_batch(batch, device)
            with autocast_context(policy):
                wind_hat, _logits, _size_hat = _split_output(model(images))
                error = torch.abs(wind_hat.float().view(-1) - winds.float().view(-1))
            total += float(error.sum())
            count += int(images.shape[0])
    if was_training:
        model.train()
    if count == 0:
        raise ValueError("validation loader produced no rows")
    return total / count


def _collect_predictions(model, loader, device, policy) -> tuple[np.ndarray, np.ndarray]:
    """Point predictions with dropout off, aligned with the loader order."""
    torch = _torch()
    was_training = model.training
    model.eval()
    truths: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            images, winds, _class_index, _sample_weight, _size = _move_batch(batch, device)
            with autocast_context(policy):
                wind_hat, _logits, _size_hat = _split_output(model(images))
            truths.append(winds.detach().float().cpu().numpy().reshape(-1))
            preds.append(wind_hat.detach().float().cpu().numpy().reshape(-1))
    if was_training:
        model.train()
    if not truths:
        raise ValueError("loader produced no rows")
    return np.concatenate(truths), np.concatenate(preds)


def _apply_step(optimizer, scaler) -> None:
    if scaler is None:
        optimizer.step()
    else:
        scaler.step(optimizer)
        scaler.update()
    optimizer.zero_grad(set_to_none=True)


def _optimizer_cpu_state(optimizer) -> dict:
    torch = _torch()
    blob = optimizer.state_dict()
    for state in blob["state"].values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.detach().cpu()
    return blob


def _move_optimizer_to(optimizer, device) -> None:
    torch = _torch()
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _fit_run(
    model,
    train_loader,
    val_loader,
    train_cfg: dict,
    device,
    max_steps: int | None,
    *,
    start_epoch: int = 0,
    history: list | None = None,
    stopper: EarlyStopping | None = None,
    steps_done: int = 0,
    optimizer_state: dict | None = None,
    on_epoch_end=None,
    already_stopped: bool = False,
):
    """Smooth L1 on wind, AdamW, early stopping on validation MAE.

    Gradient accumulation steps come from ``train.grad_accum_steps``.
    CUDA fp16 uses GradScaler. XPU bf16 and CPU fp32 do not. A checkpoint
    callback runs after every finished epoch so a sleep can resume there.
    """
    torch = _torch()
    import torch.nn.functional as F

    policy = amp_policy_for(device)
    accum = int(train_cfg["grad_accum_steps"])
    if accum < 1:
        raise ValueError("train.grad_accum_steps must be at least 1")
    if max_steps is not None and int(max_steps) < 1:
        raise ValueError("max_steps must be at least 1")
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        _move_optimizer_to(optimizer, device)
    scaler = make_grad_scaler(policy)
    epochs = int(train_cfg["epochs"])
    if stopper is None:
        stopper = EarlyStopping(int(train_cfg["patience"]))
    history = list(history or [])
    steps = int(steps_done)
    stopped_early = bool(already_stopped)
    reset_peak_memory(device)
    end_epoch = start_epoch if already_stopped else epochs
    for epoch in range(int(start_epoch), int(end_epoch)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        count = 0
        pending = 0
        for batch in train_loader:
            if max_steps is not None and steps >= int(max_steps):
                break
            images, winds, _class_index, _sample_weight, _size = _move_batch(batch, device)
            with autocast_context(policy):
                wind_hat, _logits, _size_hat = _split_output(model(images))
                loss = F.smooth_l1_loss(wind_hat.float().view(-1), winds.float().view(-1))
            if scaler is None:
                (loss / accum).backward()
            else:
                scaler.scale(loss / accum).backward()
            pending += 1
            rows = int(images.shape[0])
            total += float(loss.detach()) * rows
            count += rows
            steps += 1
            if pending == accum:
                _apply_step(optimizer, scaler)
                pending = 0
        if pending:
            _apply_step(optimizer, scaler)
        if count == 0:
            break
        val_mae = _split_mae(model, val_loader, device, policy)
        if not np.isfinite(val_mae):
            raise RuntimeError("validation MAE is not finite")
        history.append(
            {
                "epoch": epoch,
                "step": steps,
                "train_loss": total / count,
                "val_mae": float(val_mae),
            }
        )
        state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        if stopper.update(float(val_mae), state):
            stopped_early = True
        if on_epoch_end is not None:
            on_epoch_end(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                history=history,
                stopper=stopper,
                steps=steps,
                stopped_early=stopped_early,
            )
        if stopped_early:
            break
        if max_steps is not None and steps >= int(max_steps):
            break
    if stopper.best_state is None:
        raise RuntimeError("training produced no validation score")
    model.load_state_dict(stopper.best_state)
    return model, history, stopped_early, float(stopper.best), peak_memory_bytes(device)


def _config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_loss_curve(history: list[dict], csv_path: Path, png_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    table = pd.DataFrame(history)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)
    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    axis.plot(table["epoch"], table["train_loss"], marker="o", label="train SmoothL1")
    axis.plot(table["epoch"], table["val_mae"], marker="o", label="val MAE (kt)")
    axis.set_xlabel("epoch")
    axis.set_ylabel("loss / MAE")
    axis.set_title("Training curve. Not a forecast.")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(png_path, dpi=100)
    plt.close(figure)


def _score_split(truth: np.ndarray, pred: np.ndarray, config: dict) -> dict:
    """MAE, RMSE, and bias from ``src.evaluate`` on one held-out split."""
    from src.evaluate import score_winds

    edges = tuple(float(edge) for edge in config["data"]["imd5_edges_kt"])
    names = tuple(str(name) for name in config["data"]["imd5_names"])
    overall = score_winds(truth, pred, edges, names)["overall"]
    return {
        "n": overall["n"],
        "mae_kt": overall["mae_kt"],
        "rmse_kt": overall["rmse_kt"],
        "bias_kt": overall["bias_kt"],
    }


def _run_id(dataset: str, model_name: str, config_digest: str) -> str:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return f"{dataset}-{model_name}-{stamp}-{time.time_ns()}-{config_digest[:8]}"


def git_commit() -> str | None:
    """HEAD of this checkout, or null when git is not available."""
    try:
        text = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return text or None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_torch_payload(path: Path) -> dict:
    torch = _torch()
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} is not a checkpoint dict")
    return payload


def _parent_periods(payload: dict) -> set[str]:
    periods: set[str] = set()
    grouped = payload.get("wind_averaging_by_split")
    if isinstance(grouped, dict):
        for entry in grouped.values():
            if isinstance(entry, dict):
                periods.update(str(value) for value in entry.get("wind_averaging", []))
    single = payload.get("wind_averaging")
    if single not in (None, ""):
        periods.add(str(single))
    return {value for value in periods if value and value.lower() != "none"}


def load_init_weights(model, path: str | Path, config: dict) -> tuple[object, str, set[str]]:
    """Load weights only. Optimizer state is left untouched.

    A different ``model_name`` or a state dict whose keys or shapes differ
    is refused. The returned hash is the file bytes, and the period set is
    whatever the parent run recorded.
    """
    checkpoint = Path(path)
    payload = _load_torch_payload(checkpoint)
    if "state_dict" not in payload:
        raise RuntimeError(f"Refusing init-from: {checkpoint} has no state_dict")
    expected = str(config["model"]["name"])
    loaded_name = payload.get("model_name")
    if loaded_name is not None and str(loaded_name) != expected:
        raise RuntimeError(
            "Refusing init-from: architecture does not match "
            f"(checkpoint model {loaded_name}, training model {expected})"
        )
    incoming = payload["state_dict"]
    current = model.state_dict()
    if set(incoming) != set(current):
        missing = sorted(set(current) - set(incoming))[:8]
        extra = sorted(set(incoming) - set(current))[:8]
        raise RuntimeError(
            "Refusing init-from: architecture does not match "
            f"(missing {missing}, extra {extra})"
        )
    for key, tensor in current.items():
        if tuple(tensor.shape) != tuple(incoming[key].shape):
            raise RuntimeError(
                "Refusing init-from: architecture does not match at "
                f"{key}: {tuple(tensor.shape)} vs {tuple(incoming[key].shape)}"
            )
    model.load_state_dict({key: value.detach().cpu() for key, value in incoming.items()})
    run_id = payload.get("run_id")
    return run_id, _file_sha256(checkpoint), _parent_periods(payload)


def _warn_wind_mismatch(parent: set[str], current: set[str]) -> bool:
    """Loud warning when a fine-tune crosses averaging periods. No conversion."""
    if not parent or not current or parent == current:
        return False
    message = (
        "WARNING: wind_averaging mismatch. "
        f"Parent checkpoint records {sorted(parent)} and this dataset records {sorted(current)}. "
        "No conversion factor is applied. A sourced factor will be added later."
    )
    LOGGER.warning(message)
    print(message)
    return True


def _path_records(frame: pd.DataFrame, manifest: Path, split_name: str) -> list[dict]:
    rows = frame[frame["split"] == split_name].reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"split {split_name} has no rows")
    labels = [None if pd.isna(value) else str(value) for value in rows["_class"]]
    paths = [str(resolve_image(manifest, path)) for path in rows["image_path"].tolist()]
    winds = rows["wind_speed_kt"].to_numpy(dtype=np.float64)
    return [
        {"path": paths[index], "wind": float(winds[index]), "label": labels[index], "weight": 1.0}
        for index in range(len(rows))
    ]


def _parity_side(config: dict, frame: pd.DataFrame) -> int:
    side = parse_image_size(config["data"]["image_size"])
    if side is not None:
        return int(side)
    if "orig_size" in frame.columns and len(frame):
        return int(frame["orig_size"].iloc[0])
    if "native_height" in frame.columns and len(frame):
        return int(frame["native_height"].iloc[0])
    raise ValueError("cannot infer the frame size for the device parity check")


def _current_periods(report: dict) -> set[str]:
    periods: set[str] = set()
    for entry in report.values():
        periods.update(str(value) for value in entry["wind_averaging"])
    return periods


def run_training(
    config: dict,
    *,
    dataset: str | None = None,
    model: str | None = None,
    max_steps: int | None = None,
    device: str | None = None,
    resume: str | Path | None = None,
    init_from: str | Path | None = None,
) -> Path:
    """Fit one dataset and write the run record under ``paths.runs``.

    ``resume`` continues the last finished epoch in that run directory.
    ``init_from`` copies weights from another run, refits normalization on
    this dataset's train storms, and does not load the optimizer.
    """
    import yaml

    if resume is not None and init_from is not None:
        raise ValueError("pass either --resume or --init-from, not both")
    assert_model_inputs(("image",))
    train_cfg = _require_train_block(config)
    assert_normalize(config)
    resolved = copy.deepcopy(config)
    saved = None
    if resume is not None:
        run_dir = Path(resume)
        saved = _load_torch_payload(run_dir / LAST_CHECKPOINT)
        if "state_dict" not in saved or "epoch" not in saved:
            raise RuntimeError(f"{run_dir / LAST_CHECKPOINT} is not a resume checkpoint")
    if dataset is not None:
        dataset_name = str(dataset).strip().lower()
    elif saved is not None:
        dataset_name = str(saved["dataset"]).strip().lower()
    else:
        dataset_name = str(resolved["data"]["dataset"]).strip().lower()
    if dataset_name not in _DATASETS:
        raise ValueError(f"dataset must be one of {', '.join(_DATASETS)}")
    if saved is not None and str(saved["dataset"]) != dataset_name:
        raise RuntimeError(
            f"resume dataset is {saved['dataset']} but this command asked for {dataset_name}"
        )
    if model is not None:
        resolved["model"]["name"] = canonical_model_name(model)
    elif saved is not None:
        resolved["model"]["name"] = str(saved["model_name"])
    else:
        resolved["model"]["name"] = canonical_model_name(str(resolved["model"]["name"]))
    if resolved["model"]["name"] not in {"simple_cnn", "resnet18"}:
        raise ValueError("this entrypoint trains simplecnn or resnet18")
    if saved is not None and str(saved["model_name"]) != str(resolved["model"]["name"]):
        raise RuntimeError(
            "Refusing resume: architecture does not match "
            f"(checkpoint model {saved['model_name']}, training model {resolved['model']['name']})"
        )
    device_name = str(device if device is not None else resolved["train"]["device"])
    torch_device = resolve_device(device_name)
    policy = amp_policy_for(torch_device)
    if max_steps is None:
        max_steps = resolved["train"]["max_steps"]
    if max_steps is not None:
        max_steps = int(max_steps)
    resolved["data"]["dataset"] = dataset_name
    resolved["train"]["device"] = torch_device.type
    resolved["train"]["device_requested"] = device_name
    resolved["train"]["precision"] = policy.precision
    resolved["train"]["max_steps"] = max_steps
    digest = _config_hash(resolved)
    resolved["config_hash"] = digest
    print(f"device={torch_device.type} precision={policy.precision}")
    LOGGER.info("device=%s precision=%s", torch_device.type, policy.precision)

    if saved is None:
        frame, n_excluded = load_training_frame(dataset_name, resolved)
        frame = apply_frame_cache(frame, dataset_name, resolved["paths"].get("frame_cache"))
        refuse_mixed_resolution(frame, dataset_name)
        wind_period = assert_wind_averaging(resolved, frame)
        synthetic = dataset_name == "smoke" or manifest_is_synthetic(frame)
        seed = int(resolved["project"]["seed"])
        _set_seed(seed)
        assigned = assign_project_split(frame, resolved)
        assigned["_class"] = project_labels(assigned, resolved)
        groups = storm_ids_by_split(assigned)
        if not groups["train"] or not groups["val"] or not groups["test"]:
            raise LeakageError("train, val, and test storms are all required")
        run_id = _run_id(dataset_name, str(resolved["model"]["name"]), digest)
        run_dir = Path(resolved["paths"]["runs"]) / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        manifest_path = run_dir / "split_manifest.csv"
        assigned.drop(columns=["_class"], errors="ignore").to_csv(manifest_path, index=False)
        labeled = assigned.copy()
        labeled["_class"] = project_labels(labeled, resolved)
        start_epoch = 0
        prior_history: list = []
        restored_stopper = None
        steps_done = 0
        optimizer_state = None
        already_stopped = False
    else:
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
        manifest_path = run_dir / "split_manifest.csv"
        labeled = load_manifest(manifest_path)
        wind_period = assert_wind_averaging(resolved, labeled)
        synthetic = dataset_name == "smoke" or manifest_is_synthetic(labeled)
        seed = int(saved.get("seed", resolved["project"]["seed"]))
        _set_seed(seed)
        labeled["_class"] = project_labels(labeled, resolved)
        groups = storm_ids_by_split(labeled)
        if not groups["train"] or not groups["val"] or not groups["test"]:
            raise LeakageError("train, val, and test storms are all required")
        run_id = str(saved["run_id"])
        n_excluded = int(saved.get("n_excluded", 0))
        start_epoch = int(saved["epoch"]) + 1
        prior_history = list(saved.get("history") or [])
        restored_stopper = EarlyStopping(int(train_cfg["patience"]))
        restored_stopper.best = float(saved["best_metric"])
        restored_stopper.waits = int(saved["waits"])
        restored_stopper.best_state = saved["best_state"]
        steps_done = int(saved.get("steps", 0))
        optimizer_state = saved.get("optimizer_state_dict")
        already_stopped = bool(saved.get("stopped_early"))

    size_head = False
    stats = fit_split_normalization(labeled, manifest_path, resolved)
    if stats.source_split != "train":
        raise LeakageError("normalization stats may be fit on the train split only")
    dataset_stats = norm_by_dataset(dataset_name, stats)
    averaging = split_wind_averaging(labeled)
    known = [str(name) for name in resolved["data"]["imd5_names"]]
    image_size = parse_image_size(resolved["data"]["image_size"])
    max_missing = float(resolved["data"]["max_missing_fraction"])
    batch_size = int(train_cfg["batch_size"])
    workers = int(train_cfg["num_workers"])
    torch = _torch()
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = build_loader(
        PathFrames(
            _path_records(labeled, manifest_path, "train"),
            image_size,
            stats.mean,
            stats.std,
            max_missing,
            known,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        generator=generator,
    )
    val_loader = build_loader(
        PathFrames(
            _path_records(labeled, manifest_path, "val"),
            image_size,
            stats.mean,
            stats.std,
            max_missing,
            known,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )
    test_loader = build_loader(
        PathFrames(
            _path_records(labeled, manifest_path, "test"),
            image_size,
            stats.mean,
            stats.std,
            max_missing,
            known,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )
    model = build_net(resolved, size_head=size_head)
    if not any(layer.__class__.__name__ == "Dropout" for layer in model.modules()):
        raise RuntimeError("MC-dropout needs a Dropout layer left in the network")
    parent_run_id = None
    parent_sha = None
    parent_periods: set[str] = set()
    if init_from is not None:
        parent_run_id, parent_sha, parent_periods = load_init_weights(model, init_from, resolved)
    if saved is not None:
        model.load_state_dict(saved["state_dict"])
    mismatch = _warn_wind_mismatch(parent_periods, _current_periods(averaging))
    parity = None
    if torch_device.type != "cpu":
        images, winds = seeded_parity_batch(
            min(2, batch_size),
            int(resolved["data"]["channels"]),
            _parity_side(resolved, labeled),
            seed=0,
        )
        parity = compare_model_parity(
            model,
            images,
            winds,
            torch_device,
            float(train_cfg["device_parity_tol"]),
            policy=policy,
        )

    def _on_epoch_end(*, epoch, model, optimizer, history, stopper, steps, stopped_early) -> None:
        blob = {
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "optimizer_state_dict": _optimizer_cpu_state(optimizer),
            "epoch": int(epoch),
            "history": list(history),
            "best_state": stopper.best_state,
            "best_metric": float(stopper.best),
            "waits": int(stopper.waits),
            "steps": int(steps),
            "stopped_early": bool(stopped_early),
            "run_id": run_id,
            "dataset": dataset_name,
            "model_name": str(resolved["model"]["name"]),
            "seed": seed,
            "n_excluded": int(n_excluded),
        }
        torch.save(blob, run_dir / LAST_CHECKPOINT)

    model, history, stopped_early, best_val_mae, peak_bytes = _fit_run(
        model,
        train_loader,
        val_loader,
        train_cfg,
        torch_device,
        max_steps,
        start_epoch=start_epoch,
        history=prior_history,
        stopper=restored_stopper,
        steps_done=steps_done,
        optimizer_state=optimizer_state,
        on_epoch_end=_on_epoch_end,
        already_stopped=already_stopped,
    )
    if not history:
        raise RuntimeError("training produced no loss curve")
    val_truth, val_pred = _collect_predictions(model, val_loader, torch_device, policy)
    test_truth, test_pred = _collect_predictions(model, test_loader, torch_device, policy)
    checkpoint_name = str(resolved["paths"]["checkpoint_name"])
    if not checkpoint_name or checkpoint_name != Path(checkpoint_name).name:
        raise ValueError("paths.checkpoint_name must be a file name, not a path")
    checkpoint = run_dir / checkpoint_name
    split_ids = {name: sorted(groups[name]) for name in ("train", "val", "test")}
    train_winds = labeled.loc[labeled["split"] == "train", "wind_speed_kt"].to_numpy(dtype=np.float64)
    properties = xpu_device_properties(torch_device)
    commit = git_commit()
    payload = {
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "model_name": str(resolved["model"]["name"]),
        "dropout": float(resolved["model"]["dropout"]),
        "pretrained": False,
        "in_channels": int(resolved["data"]["channels"]),
        "n_classes": len(known),
        "class_names": known,
        "edges_kt": [float(edge) for edge in resolved["data"]["imd5_edges_kt"]],
        "norm_mean": stats.mean,
        "norm_std": stats.std,
        "norm_source_split": stats.source_split,
        "norm_by_dataset": dataset_stats,
        "image_size": image_size,
        "max_missing_fraction": max_missing,
        "mean_baseline_kt": float(np.mean(train_winds)),
        "is_synthetic": bool(synthetic),
        "wind_averaging": wind_period,
        "wind_averaging_s": wind_averaging_seconds(wind_period),
        "wind_averaging_by_split": averaging,
        "wind_averaging_mismatch": bool(mismatch),
        "split": {
            "strategy": str(resolved["split"]["strategy"]),
            "seed": int(resolved["split"]["seed"]),
            "test_fraction": float(resolved["split"]["test_fraction"]),
            "val_fraction": float(resolved["split"]["val_fraction"]),
        },
        "split_storm_ids": split_ids,
        "best_val_loss": best_val_mae,
        "best_val_mae": best_val_mae,
        "epoch": int(history[-1]["epoch"]),
        "present_as_forecast": False,
        "honesty": SMOKE_BANNER if synthetic else NOT_A_FORECAST,
        "n_frames": 1,
        "size_head": False,
        "hidden": None,
        "config_hash": digest,
        "seed": seed,
        "dataset": dataset_name,
        "run_id": run_id,
        "result": False if synthetic else True,
        "device": torch_device.type,
        "device_requested": device_name,
        "precision": policy.precision,
        "use_grad_scaler": bool(policy.use_grad_scaler),
        "grad_accum_steps": int(train_cfg["grad_accum_steps"]),
        "peak_memory_bytes": peak_bytes,
        "git_commit": commit,
        "parent_run_id": None if parent_run_id is None else str(parent_run_id),
        "parent_checkpoint_sha256": parent_sha,
        "xpu_device_properties": properties,
        "device_parity_ok": None if parity is None else bool(parity["ok"]),
        "device_parity_max_abs_diff": None if parity is None else parity["max_abs_diff"],
        "device_parity_output_max_abs": None if parity is None else parity["output_max_abs"],
        "device_parity_loss_max_abs": None if parity is None else parity["loss_max_abs"],
        "device_parity_tol": float(train_cfg["device_parity_tol"]),
    }
    torch.save(payload, checkpoint)
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    _write_loss_curve(history, run_dir / "loss_curve.csv", run_dir / "loss_curve.png")
    metrics = {
        "result": False if synthetic else True,
        "is_synthetic": bool(synthetic),
        "quote_as_result": "no",
        "present_as_forecast": False,
        "dataset": dataset_name,
        "model": str(resolved["model"]["name"]),
        "seed": seed,
        "config_hash": digest,
        "git_commit": commit,
        "resolved_config": "resolved_config.yaml",
        "run_id": run_id,
        "wind_averaging": wind_period,
        "wind_averaging_s": wind_averaging_seconds(wind_period),
        "wind_averaging_by_split": averaging,
        "wind_averaging_mismatch": bool(mismatch),
        "parent_run_id": None if parent_run_id is None else str(parent_run_id),
        "parent_checkpoint_sha256": parent_sha,
        "n_excluded": int(n_excluded),
        "stopped_early": bool(stopped_early),
        "best_val_mae": best_val_mae,
        "epochs_ran": int(len(history)),
        "split_storm_ids": split_ids,
        "scored_by": "src.evaluate.score_winds",
        "val": _score_split(val_truth, val_pred, resolved),
        "test": _score_split(test_truth, test_pred, resolved),
        "device": torch_device.type,
        "device_requested": device_name,
        "precision": policy.precision,
        "use_grad_scaler": bool(policy.use_grad_scaler),
        "grad_accum_steps": int(train_cfg["grad_accum_steps"]),
        "peak_memory_bytes": peak_bytes,
        "xpu_device_properties": properties,
        "device_parity_ok": None if parity is None else bool(parity["ok"]),
        "device_parity_max_abs_diff": None if parity is None else parity["max_abs_diff"],
        "device_parity_output_max_abs": None if parity is None else parity["output_max_abs"],
        "device_parity_loss_max_abs": None if parity is None else parity["loss_max_abs"],
        "device_parity_tol": float(train_cfg["device_parity_tol"]),
        "norm_by_dataset": dataset_stats,
        "note": SMOKE_BANNER if synthetic else NOT_A_FORECAST,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    if synthetic:
        print(SMOKE_BANNER)
    else:
        print(NOT_A_FORECAST)
    print(f"Wrote run record {run_dir}")
    return run_dir


def main(argv: list[str] | None = None) -> Path:
    """``python -m src.train --config config/config.yaml``."""
    parser = argparse.ArgumentParser(
        description="Train a wind-speed model. Hyperparameters, paths, and seeds come from the YAML."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", choices=_DATASETS, default=None)
    parser.add_argument("--model", choices=("simplecnn", "resnet18"), default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "xpu", "cuda", "cpu"), default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--init-from", type=Path, default=None)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    return run_training(
        config,
        dataset=args.dataset,
        model=args.model,
        max_steps=args.max_steps,
        device=args.device,
        resume=args.resume,
        init_from=args.init_from,
    )


if __name__ == "__main__":
    main()
