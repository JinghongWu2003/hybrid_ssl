"""Masking utilities for MAE."""
from __future__ import annotations

import torch


def num_patches(img_size: int, patch_size: int) -> int:
    if img_size % patch_size != 0:
        raise ValueError(f"Image size {img_size} must be divisible by patch size {patch_size}.")
    grid = img_size // patch_size
    return grid * grid


def sample_mask(batch: int, num_patches_: int, mask_ratio: float, device: torch.device) -> torch.Tensor:
    keep = max(1, int(round(num_patches_ * (1 - mask_ratio))))
    noise = torch.rand(batch, num_patches_, device=device)
    ids_shuffle = torch.argsort(noise, dim=1)
    mask = torch.ones(batch, num_patches_, device=device, dtype=torch.bool)
    mask.scatter_(1, ids_shuffle[:, :keep], False)
    return mask


def random_masking(x: torch.Tensor, mask_ratio: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomly mask a portion of tokens following the MAE strategy."""

    if x.dim() != 3:
        raise ValueError(f"Expected token tensor of shape (B, N, C), got {tuple(x.shape)}")
    B, N, C = x.shape
    if not 0.0 <= mask_ratio < 1.0:
        raise ValueError(f"mask_ratio must be in [0, 1), got {mask_ratio}")

    len_keep = max(1, int(N * (1 - mask_ratio)))

    noise = torch.rand(B, N, device=x.device)  # noise in [0, 1)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :len_keep]
    ids_keep_expanded = ids_keep.unsqueeze(-1).expand(-1, -1, C)
    x_masked = torch.gather(x, dim=1, index=ids_keep_expanded)

    mask = torch.ones(B, N, device=x.device, dtype=x.dtype)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore


__all__ = ["num_patches", "sample_mask", "random_masking"]
