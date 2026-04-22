

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class CarPoseModel(nn.Module):
    """EfficientNet-B0 fine-tuned for N-way vehicle orientation (default 8)."""

    def __init__(self, num_classes: int = 8, dropout: float = 0.2, pretrained: bool = True):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class SiameseNetwork(nn.Module):
    """ResNet50 backbone with a small MLP projection head."""

    def __init__(self, embed_dim: int = 128, hidden_dim: int = 512, pretrained: bool = True):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        self.encoder = models.resnet50(weights=weights)
        self.encoder.fc = nn.Sequential(
            nn.Linear(2048, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    @staticmethod
    def similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cosine_similarity(a, b, dim=-1)
