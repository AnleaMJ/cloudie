"""Centralized seed helper for reproducible runs.

Sets Python, NumPy, and (when available) PyTorch random seeds in one call.
Training, baselines, and splits should call ``seed_everything`` before any
stochastic operation.  This module does not import torch at module level so
that CPU-only installs still work.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int = 42) -> int:
    """Set Python, NumPy, and optional PyTorch seeds.

    Returns the seed so callers can log it.

    Parameters
    ----------
    seed : int
        Non-negative integer seed.  The same value is forwarded to every
        generator so a single config entry controls all randomness.
    """
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.manual_seed(seed)

    return seed
