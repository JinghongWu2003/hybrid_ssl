"""Hybrid MAE + SimCLR model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn

from losses.info_nce import InfoNCELoss
from losses.mae_reconstruction import MAELoss
from losses.schedulers import gradient_balance
from utils.mask import random_masking

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
    decoder_dim: int = 512
    decoder_depth: int = 4
    decoder_heads: int = 8
    norm_pix_loss: bool = True
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
            img_size=cfg.img_size,
            decoder_dim=cfg.decoder_dim,
            depth=cfg.decoder_depth,
            num_heads=cfg.decoder_heads,
        )
        self.info_nce = InfoNCELoss(cfg.temp)
        self.mae_loss = MAELoss(norm_pix_loss=cfg.norm_pix_loss)
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

        visible_tokens, mask, ids_restore = random_masking(tokens, self.mask_ratio)
        preds = self.decoder(visible_tokens, mask, ids_restore)

        loss_rec = self.mae_loss(mae_image, preds, mask, self.decoder.patch_size)

        losses = {"rec": loss_rec, "contrast": loss_contrast}
        loss_rec_for_total = loss_rec
        loss_contrast_for_total = loss_contrast
        if self.grad_balance:
            balanced = gradient_balance({k: v for k, v in losses.items()}, self.parameters())
            loss_rec_for_total = balanced["rec"]
            loss_contrast_for_total = balanced["contrast"]
        loss_total = alpha * loss_rec_for_total + (1 - alpha) * loss_contrast_for_total

        return {
            "z1": z1,
            "z2": z2,
            "h1": h1,
            "h2": h2,
            "recon": preds,
            "mask": (mask > 0.5) if mask.dtype != torch.bool else mask,
            "ids_restore": ids_restore,
            "loss_rec": loss_rec.detach(),
            "loss_contrast": loss_contrast.detach(),
            "loss_total": loss_total,
        }

__all__ = ["HybridModel", "HybridConfig"]
