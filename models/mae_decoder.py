"""Lightweight MAE decoder."""
from __future__ import annotations

import torch
from torch import nn


class MAEDecoder(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        patch_size: int,
        decoder_dim: int = 512,
        img_channels: int = 3,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.img_channels = img_channels
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder = nn.Sequential(
            nn.LayerNorm(decoder_dim),
            nn.Linear(decoder_dim, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, decoder_dim),
            nn.GELU(),
            nn.LayerNorm(decoder_dim),
        )
        self.output = nn.Linear(decoder_dim, patch_size * patch_size * img_channels)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Reconstruct patches; mask selects which predictions are used for loss."""
        # tokens expected shape: (B, N_patches, encoder_dim)
        x = self.decoder_embed(tokens)
        mask_token = self.mask_token.expand(x.size(0), x.size(1), -1)
        x = torch.where(mask.unsqueeze(-1), mask_token, x)
        x = self.decoder(x)
        preds = self.output(x)
        return preds


__all__ = ["MAEDecoder"]
