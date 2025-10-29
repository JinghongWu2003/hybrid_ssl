"""ResNet encoder definitions."""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torchvision import models


class ResNetEncoder(nn.Module):
    def __init__(self, arch: str) -> None:
        super().__init__()
        if arch == "resnet18":
            backbone = models.resnet18(weights=None)
            out_dim = 512
        elif arch == "resnet50":
            backbone = models.resnet50(weights=None)
            out_dim = 2048
        else:
            raise ValueError(f"Unsupported ResNet architecture: {arch}")

        modules = list(backbone.children())[:-2]
        self.feature_extractor = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor, return_tokens: bool = False) -> Tuple[torch.Tensor, torch.Tensor | None]:
        feats = self.feature_extractor(x)
        pooled = torch.flatten(self.pool(feats), 1)
        tokens = feats.flatten(2).transpose(1, 2) if return_tokens else None
        return pooled, tokens


def build_resnet_encoder(name: str, img_size: int, patch_size: int) -> ResNetEncoder:  # pylint: disable=unused-argument
    return ResNetEncoder(name.lower())


__all__ = ["build_resnet_encoder", "ResNetEncoder"]
