"""The only import point for the network and for MC-dropout.

``cloudie.model.MCDropout`` is the class from the merged regressor work.
``cloudie.train.build_net`` wraps that module. This file does not copy
either implementation. If the two trees are consolidated, change the
imports here and leave the callers alone.
"""

from __future__ import annotations


class ModelAccess:
    """Lazy imports so a gate check does not load torch."""

    @staticmethod
    def mc_dropout():
        """Return ``cloudie.model.MCDropout``."""
        from cloudie.model import MCDropout

        return MCDropout

    @staticmethod
    def build_net(config: dict, size_head: bool = False):
        """Return the training wrapper around the configured regressor."""
        from cloudie.train import build_net

        return build_net(config, size_head=size_head)
