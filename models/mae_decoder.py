"""Transformer-based MAE decoder."""
from __future__ import annotations

import torch
from torch import nn

from utils.pos_embed import build_2d_sincos_pos_embed


class DecoderBlock(nn.Module):
    """Transformer block with pre-layer normalization."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + y
        y = self.mlp(self.norm2(x))
        x = x + y
        return x


class MAEDecoder(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        patch_size: int,
        img_size: int,
        img_channels: int = 3,
        decoder_dim: int = 512,
        depth: int = 4,
        num_heads: int = 8,
    ) -> None:
        super().__init__()

        if img_size % patch_size != 0:
            raise ValueError("Image size must be divisible by patch size for MAE decoder.")

        self.patch_size = patch_size
        self.img_channels = img_channels
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        pos_embed = torch.zeros(1, self.num_patches, decoder_dim)
        self.pos_embed = nn.Parameter(pos_embed, requires_grad=False)
        self._pos_initialized = False

        self.blocks = nn.ModuleList([DecoderBlock(decoder_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(decoder_dim)
        self.head = nn.Linear(decoder_dim, patch_size * patch_size * img_channels)

        nn.init.trunc_normal_(self.mask_token, std=0.02)

    @torch.no_grad()
    def build_pos_embed(self, device: torch.device) -> None:
        grid = int(self.num_patches ** 0.5)
        pos_embed = build_2d_sincos_pos_embed(self.pos_embed.shape[-1], grid)
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).to(device)
        self.pos_embed.data.copy_(pos_embed)
        self._pos_initialized = True

    def _add_pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        if not self._pos_initialized:
            self.build_pos_embed(x.device)
        return x + self.pos_embed.to(x.dtype)

    def forward(
        self,
        enc_tokens: torch.Tensor,
        mask: torch.Tensor,
        ids_restore: torch.Tensor,
    ) -> torch.Tensor:
        """Decode masked tokens to pixel patches."""

        if enc_tokens.dim() != 3:
            raise ValueError("enc_tokens must have shape (B, N_visible, C)")

        B, _, _ = enc_tokens.shape
        x = self.proj(enc_tokens)

        # Re-insert mask tokens
        len_total = ids_restore.size(1)
        num_mask = len_total - x.size(1)
        if num_mask < 0:
            raise ValueError("ids_restore indicates more visible tokens than provided.")
        mask_tokens = self.mask_token.expand(B, num_mask, -1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        index = ids_restore.unsqueeze(-1).expand(-1, -1, x_.size(-1))
        x_ = torch.gather(x_, dim=1, index=index)

        x_ = self._add_pos_embed(x_)

        for blk in self.blocks:
            x_ = blk(x_)
        x_ = self.norm(x_)

        preds = self.head(x_)
        return preds


__all__ = ["MAEDecoder"]
