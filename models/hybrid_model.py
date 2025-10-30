"""Hybrid MAE + SimCLR model."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn

from losses.info_nce import InfoNCELoss
from losses.mae_reconstruction import MAELoss
from losses.schedulers import gradient_balance
from utils.mask import sample_mask

from .mae_decoder import MAEDecoder
from .projector import Projector
from .resnet_encoder import build_resnet_encoder
from .vit_encoder import build_vit_encoder


@dataclass
class HybridConfig:
    encoder: str
    img_size: int
    patch_size: int
    mask_ratio: float
    projector_dim: int
    projector_layers: int
    temp: float
    grad_balance: bool = False


class HybridModel(nn.Module):
    def __init__(self, cfg: HybridConfig) -> None:
        super().__init__()
        encoder_name = cfg.encoder.lower()
        if encoder_name.startswith("vit"):
            self.encoder = build_vit_encoder(encoder_name, cfg.img_size, cfg.patch_size)
        elif encoder_name.startswith("resnet"):
            self.encoder = build_resnet_encoder(encoder_name, cfg.img_size, cfg.patch_size)
        else:
            raise ValueError(f"Unsupported encoder: {cfg.encoder}")
        self.encoder_name = encoder_name
        self.mask_ratio = cfg.mask_ratio
        self.patch_size = cfg.patch_size
        encoder_dim = getattr(self.encoder, "embed_dim", None)
        if encoder_dim is None:
            encoder_dim = getattr(self.encoder, "out_dim")

        self.projector = Projector(
            in_dim=encoder_dim,
            hidden_dim=max(cfg.projector_dim, 128),
            out_dim=cfg.projector_dim,
            num_layers=cfg.projector_layers,
        )
        self.decoder = MAEDecoder(
            encoder_dim=encoder_dim,
            patch_size=self.patch_size,
        )
        self.info_nce = InfoNCELoss(cfg.temp)
        self.mae_loss = MAELoss()
        self.grad_balance = cfg.grad_balance

    def forward(self, batch: Dict[str, torch.Tensor], alpha: float) -> Dict[str, torch.Tensor]:
        x1, x2, mae_image = batch["view1"], batch["view2"], batch["image"]
        z1, _ = self.encoder(x1, return_tokens=False)
        z2, _ = self.encoder(x2, return_tokens=False)
        h1 = self.projector(z1)
        h2 = self.projector(z2)
        loss_contrast = self.info_nce(h1, h2)

        _, tokens = self.encoder(mae_image, return_tokens=True)
        if tokens is None:
            raise RuntimeError("Encoder must return patch tokens for MAE branch.")
        B, N, _ = tokens.shape
        grid = int(math.sqrt(N))
        inferred_patch = self.patch_size if grid == 0 else self.encoder_input_patch_size(mae_image.size(-1), grid)
        if inferred_patch != self.patch_size:
            # Adjust decoder patch size for non-square tokenization (e.g., ResNet)
            self.patch_size = inferred_patch
            self.decoder.set_patch_size(inferred_patch)
        mask = sample_mask(B, N, self.mask_ratio, mae_image.device)
        preds = self.decoder(tokens, mask)
        loss_rec = self.mae_loss(mae_image, preds, mask, self.decoder.patch_size)

        losses = {"rec": loss_rec, "contrast": loss_contrast}
        if self.grad_balance:
            balanced = gradient_balance({k: v for k, v in losses.items()}, self.parameters())
            loss_rec = balanced["rec"]
            loss_contrast = balanced["contrast"]
        loss_total = alpha * loss_rec + (1 - alpha) * loss_contrast

        return {
            "z1": z1,
            "z2": z2,
            "h1": h1,
            "h2": h2,
            "recon": preds,
            "mask": mask,
            "loss_rec": loss_rec.detach(),
            "loss_contrast": loss_contrast.detach(),
            "loss_total": loss_total,
        }

    @staticmethod
    def encoder_input_patch_size(img_size: int, grid: int) -> int:
        if grid == 0:
            return img_size
        if img_size % grid != 0:
            raise ValueError("Image size must be divisible by token grid size")
        return img_size // grid


__all__ = ["HybridModel", "HybridConfig"]
