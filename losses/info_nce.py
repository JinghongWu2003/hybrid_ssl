"""InfoNCE / NT-Xent loss."""
from __future__ import annotations

import torch
from torch import nn

from utils.dist import concat_all_gather, get_world_size


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.2) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)
        features = torch.cat([z1, z2], dim=0)
        if get_world_size() > 1:  # pragma: no cover - requires DDP
            features = concat_all_gather(features)
        logits = torch.matmul(features, features.T) / self.temperature
        logits = logits - torch.max(logits, dim=1, keepdim=True).values.detach()
        mask = torch.eye(logits.size(0), dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(mask, float("-inf"))
        batch_size = z1.size(0)
        targets = torch.arange(batch_size, device=z1.device)
        targets = torch.cat([targets + batch_size, targets])
        loss = nn.functional.cross_entropy(logits, targets)
        return loss


__all__ = ["InfoNCELoss"]
