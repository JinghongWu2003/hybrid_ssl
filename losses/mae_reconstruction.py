"""MAE reconstruction utilities and loss."""
from __future__ import annotations

import math

import torch
from torch import nn


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Split images into non-overlapping patches."""

    if images.dim() != 4:
        raise ValueError(f"Expected images of shape (B, C, H, W), got {tuple(images.shape)}")
    B, C, H, W = images.shape
    if H != W:
        raise ValueError("Only square images are supported for patchify.")
    if H % patch_size != 0:
        raise ValueError(f"Image size {H} must be divisible by patch size {patch_size}.")

    grid = H // patch_size
    patches = images.reshape(B, C, grid, patch_size, grid, patch_size)
    patches = patches.permute(0, 2, 4, 3, 5, 1).reshape(B, grid * grid, patch_size * patch_size * C)
    return patches


def unpatchify(patches: torch.Tensor, patch_size: int, channels: int) -> torch.Tensor:
    """Reconstruct images from patch sequences."""

    if patches.dim() != 3:
        raise ValueError(f"Expected patches of shape (B, N, L), got {tuple(patches.shape)}")
    B, N, L = patches.shape
    grid = int(math.sqrt(N))
    if grid * grid != N:
        raise ValueError("Number of patches must form a square grid.")
    expected_length = patch_size * patch_size * channels
    if L != expected_length:
        raise ValueError(f"Patch embedding length {L} does not match expected {expected_length}.")

    patches = patches.reshape(B, grid, grid, patch_size, patch_size, channels)
    patches = patches.permute(0, 5, 1, 3, 2, 4)
    images = patches.reshape(B, channels, grid * patch_size, grid * patch_size)
    return images


def mae_loss(
    preds: torch.Tensor,
    images: torch.Tensor,
    mask: torch.Tensor,
    patch_size: int,
    *,
    norm_pix_loss: bool = True,
) -> torch.Tensor:
    """Compute masked MSE loss between predictions and image patches."""

    target = patchify(images, patch_size)
    if target.shape != preds.shape:
        raise ValueError(
            "Predictions and targets must have the same shape: "
            f"got {tuple(preds.shape)} vs {tuple(target.shape)}"
        )

    if mask.dtype != preds.dtype:
        mask = mask.to(preds.dtype)

    if norm_pix_loss:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, unbiased=False, keepdim=True)
        target = (target - mean) / torch.sqrt(var + 1e-6)

    loss = (preds - target) ** 2
    loss = loss.mean(dim=-1)  # (B, N)
    masked_loss = (loss * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    loss_per_sample = masked_loss / denom
    return loss_per_sample.mean()


class MAELoss(nn.Module):
    """Module wrapper around :func:`mae_loss`."""

    def __init__(self, norm_pix_loss: bool = True) -> None:
        super().__init__()
        self.norm_pix_loss = norm_pix_loss

    def forward(
        self,
        images: torch.Tensor,
        preds: torch.Tensor,
        mask: torch.Tensor,
        patch_size: int,
    ) -> torch.Tensor:
        return mae_loss(
            preds=preds,
            images=images,
            mask=mask,
            patch_size=patch_size,
            norm_pix_loss=self.norm_pix_loss,
        )


__all__ = ["MAELoss", "patchify", "unpatchify", "mae_loss"]
