"""Deep-model preprocessing accepts a square resize and a native chip."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

PREPROCESSORS = ("src.preprocessing",)
CONFIGS = ("src.config",)
CHIP_SIZES = [128, 201, 366]


@pytest.mark.parametrize("module_name", CONFIGS)
@pytest.mark.parametrize("value", [None, "native", "Native", " null "])
def test_native_token_keeps_the_chip_size(module_name: str, value: object) -> None:
    parse = importlib.import_module(module_name).parse_image_size
    assert parse(value) is None


@pytest.mark.parametrize("module_name", CONFIGS)
@pytest.mark.parametrize("image_size", CHIP_SIZES)
def test_square_sizes_parse(module_name: str, image_size: int) -> None:
    module = importlib.import_module(module_name)
    assert module.parse_image_size(image_size) == image_size
    assert module.parse_image_size(str(image_size)) == image_size
    assert module.image_sizes_match(image_size, str(image_size))
    assert module.image_sizes_match("native", None)
    assert not module.image_sizes_match("native", image_size)


@pytest.mark.parametrize("module_name", CONFIGS)
@pytest.mark.parametrize("value", [0, -4, "nope", "", True])
def test_image_size_rejects_junk(module_name: str, value: object) -> None:
    parse = importlib.import_module(module_name).parse_image_size
    with pytest.raises(ValueError):
        parse(value)


@pytest.mark.parametrize("module_name", PREPROCESSORS)
@pytest.mark.parametrize("image_size", CHIP_SIZES)
def test_prepare_image_resizes_to_the_requested_square(module_name: str, image_size: int) -> None:
    prepare = importlib.import_module(module_name).prepare_image
    image = np.zeros((40, 70), dtype=np.float32)
    image[0, 0] = 4.0
    prepared = prepare(image, image_size, None)
    assert prepared.shape == (1, image_size, image_size)
    assert np.isfinite(prepared).all()


@pytest.mark.parametrize("module_name", PREPROCESSORS)
@pytest.mark.parametrize("image_size", CHIP_SIZES)
@pytest.mark.parametrize("setting", [None, "native"])
def test_prepare_image_keeps_native_square_chips(
    module_name: str, image_size: int, setting: object
) -> None:
    prepare = importlib.import_module(module_name).prepare_image
    image = np.full((image_size, image_size), 3.0, dtype=np.float32)
    prepared = prepare(image, setting, None)
    assert prepared.shape == (1, image_size, image_size)
    assert prepared[0, 0, 0] == pytest.approx(3.0)


@pytest.mark.parametrize("module_name", PREPROCESSORS)
def test_native_size_does_not_force_a_square(module_name: str) -> None:
    prepare = importlib.import_module(module_name).prepare_image
    image = np.full((30, 18), 2.0, dtype=np.float32)
    prepared = prepare(image, "native", None)
    assert prepared.shape == (1, 30, 18)


@pytest.mark.parametrize("module_name", PREPROCESSORS)
def test_native_stack_refuses_mixed_shapes(module_name: str) -> None:
    stack = importlib.import_module(module_name).stack_prepared
    left = np.zeros((1, 8, 8), dtype=np.float32)
    right = np.zeros((1, 9, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="height and width"):
        stack([left, right])
    stacked = stack([left, left.copy()])
    assert stacked.shape == (2, 1, 8, 8)
