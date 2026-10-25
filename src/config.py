"""YAML config loader. Training and baselines read hyperparameters from here."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")
    return raw


def parse_image_size(value: object) -> int | None:
    """Square side to resize to, or None to keep the chip's height and width.

    ``None`` and the string ``native`` do not resize. An integer resizes to
    that square. NASA TorchGeo chips are 366 by 366 and TCIR chips are 201
    by 201; this function does not assume either size and does not download
    imagery.
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"native", "null", "none"}:
            return None
        if not text:
            raise ValueError("data.image_size is blank")
        try:
            value = int(text)
        except ValueError as exc:
            raise ValueError(
                "data.image_size must be native or a positive integer"
            ) from exc
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"data.image_size must be native or a positive integer, got {value!r}"
        )
    size = int(value)
    if size < 1 or float(value) != float(size):
        raise ValueError("data.image_size must be native or a positive integer")
    return size


def image_sizes_match(left: object, right: object) -> bool:
    """True when both sides resize to the same square, or both stay native."""
    return parse_image_size(left) == parse_image_size(right)


def checkpoint_path(config: dict, root: str | Path | None = None) -> Path:
    """Path of the deep-model checkpoint named in ``paths``.

    ``paths.artifacts`` is the directory. ``paths.checkpoint_name`` is a
    file name, not a directory. ``root`` prefixes a relative path.
    """
    artifacts = Path(config["paths"]["artifacts"])
    name = str(config["paths"]["checkpoint_name"])
    if not name or name != Path(name).name:
        raise ValueError("paths.checkpoint_name must be a file name, not a path")
    path = artifacts / name
    if root is not None and not path.is_absolute():
        path = Path(root) / path
    return path
