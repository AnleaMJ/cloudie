"""Model interfaces. Importing this module requires torch."""

from __future__ import annotations

import torch
from torch import nn


class SimpleCNN(nn.Module):
    """Small regressor for smoke tests and a first real baseline."""

    def __init__(self, in_channels: int = 1, dropout: float = 0.3) -> None:
        super().__init__()
        self.feature_dim = 64
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(64, 1))

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.features(inputs).flatten(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(inputs))


class ResNetRegressor(nn.Module):
    """ResNet-18 regressor. One-channel IR is repeated to three channels.

    Repeating keeps the pretrained first convolution. Replacing it would
    discard those weights. ImageNet pretraining is not a physical prior.
    ``pretrained=True`` downloads weights and must not run in CI.
    """

    def __init__(
        self, in_channels: int = 1, pretrained: bool = False, dropout: float = 0.2
    ) -> None:
        super().__init__()
        from torchvision.models import resnet18

        weights = None
        if pretrained:
            from torchvision.models import ResNet18_Weights

            weights = ResNet18_Weights.DEFAULT
        backbone = resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.feature_dim = in_features
        self.repeat_to_rgb = in_channels == 1
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.repeat_to_rgb and inputs.shape[1] == 1:
            inputs = inputs.repeat(1, 3, 1, 1)
        return self.backbone(inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(inputs))


def enable_mc_dropout(module: nn.Module) -> None:
    """Eval mode for BatchNorm, train mode for Dropout."""
    module.eval()
    for layer in module.modules():
        if isinstance(layer, nn.Dropout):
            layer.train()


@torch.no_grad()
def mc_dropout_predict(
    module: nn.Module, inputs: torch.Tensor, n_samples: int = 20
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean and std. This is not a calibrated prediction interval."""
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    enable_mc_dropout(module)
    draws = torch.stack([module(inputs) for _ in range(n_samples)], dim=0)
    return draws.mean(dim=0), draws.std(dim=0, unbiased=True)
