"""MAE reconstruction loss."""
from __future__ import annotations

import torch
from torch import nn

from utils.mask import patchify


class MAELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, images: torch.Tensor, preds: torch.Tensor, mask: torch.Tensor, patch_size: int) -> torch.Tensor:
        target = patchify(images, patch_size)
        masked_target = target[mask]
        masked_preds = preds[mask]
        return self.mse(masked_preds, masked_target)


__all__ = ["MAELoss"]
