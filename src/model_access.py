"""The only import point for the network and for MC-dropout.

``src.model.MCDropout`` is the class from the merged regressor work.
``src.train.build_net`` wraps that module and ``cloudie.train`` re-exports
it. This file does not copy either implementation.
"""

from __future__ import annotations


class ModelAccess:
    """Lazy imports so a gate check does not load torch."""

    @staticmethod
    def mc_dropout():
        """Return ``src.model.MCDropout``."""
        from src.model import MCDropout

        return MCDropout

    @staticmethod
    def build_net(config: dict, size_head: bool = False):
        """Return the training wrapper around the configured regressor."""
        from src.train import build_net

        return build_net(config, size_head=size_head)
