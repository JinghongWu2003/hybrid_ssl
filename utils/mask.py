"""Masking utilities for MAE."""
from __future__ import annotations

import math
from typing import Tuple

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


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Convert images to patch sequences."""
    B, C, H, W = images.shape
    assert H == W and H % patch_size == 0
    h = w = H // patch_size
    x = images.reshape(B, C, h, patch_size, w, patch_size)
    x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, h * w, patch_size * patch_size * C)
    return x


def unpatchify(patches: torch.Tensor, patch_size: int, channels: int) -> torch.Tensor:
    B, N, L = patches.shape
    grid = int(math.sqrt(N))
    x = patches.reshape(B, grid, grid, patch_size, patch_size, channels)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, channels, grid * patch_size, grid * patch_size)
    return x


__all__ = ["num_patches", "sample_mask", "patchify", "unpatchify"]
