"""cloudie re-exports the canonical objects defined in src."""

from __future__ import annotations

import importlib

import pytest

# Checkpoint scoring stays in cloudie.evaluate. Package markers stay separate.
# The model shim is checked in test_model.py, where torch is installed.
SHIMMED = (
    "config",
    "data_loader",
    "inference",
    "make_smoke_data",
    "metrics",
    "preprocessing",
    "schema",
    "splits",
)


def assert_shim_reexports(name: str) -> None:
    """Every public name on the src module is the same object on the shim."""
    canonical = importlib.import_module(f"src.{name}")
    shim = importlib.import_module(f"cloudie.{name}")
    compared = 0
    for attr, value in vars(canonical).items():
        if attr.startswith("_"):
            continue
        assert hasattr(shim, attr), f"cloudie.{name}.{attr} is missing"
        assert getattr(shim, attr) is value, f"cloudie.{name}.{attr} is not src.{name}.{attr}"
        compared += 1
    assert compared > 0


@pytest.mark.parametrize("name", SHIMMED)
def test_shims_reexport_the_same_objects(name: str) -> None:
    assert_shim_reexports(name)
