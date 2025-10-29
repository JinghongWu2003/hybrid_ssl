"""Vision Transformer encoder definitions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn


@dataclass
class ViTConfig:
    embed_dim: int
    depth: int
    num_heads: int
    mlp_ratio: float = 4.0


VIT_PRESETS: Dict[str, ViTConfig] = {
    "vit_small": ViTConfig(embed_dim=384, depth=8, num_heads=6),
    "vit_base": ViTConfig(embed_dim=768, depth=12, num_heads=12),
}


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    def __init__(self, img_size: int, patch_size: int, cfg: ViTConfig, in_chans: int = 3) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, cfg.embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, cfg.embed_dim))
        self.pos_drop = nn.Dropout(p=0.1)
        self.blocks = nn.ModuleList([Block(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio) for _ in range(cfg.depth)])
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.embed_dim = cfg.embed_dim
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor, return_tokens: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B = x.size(0)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, : x.size(1)]
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls = x[:, 0]
        tokens = x[:, 1:] if return_tokens else None
        return cls, tokens


def build_vit_encoder(name: str, img_size: int, patch_size: int) -> ViTEncoder:
    name = name.lower()
    if name not in VIT_PRESETS:
        raise ValueError(f"Unknown ViT encoder: {name}. Available: {list(VIT_PRESETS)}")
    if img_size % patch_size != 0:
        raise ValueError(f"Image size {img_size} must be divisible by patch size {patch_size}.")
    return ViTEncoder(img_size=img_size, patch_size=patch_size, cfg=VIT_PRESETS[name])


__all__ = ["build_vit_encoder", "ViTEncoder"]
