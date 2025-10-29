"""Fine-tuning script."""
from __future__ import annotations

import argparse

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from data.datamodules import DataConfig, HybridDataModule
from models.hybrid_model import HybridConfig, HybridModel
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.common import ensure_dir, load_config
from utils.logging import Logger
from utils.metrics import accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tuning for downstream tasks")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--wandb", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    dataset_cfg = cfg["dataset"]
    finetune_cfg = cfg["finetune"]
    ckpt_path = cfg["checkpoint"]["encoder_ckpt"]
    log_cfg = cfg.get("logging", {"out_dir": "runs/finetune", "tensorboard": True})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dm = HybridDataModule(
        DataConfig(
            name=dataset_cfg["name"],
            root=dataset_cfg["root"],
            img_size=dataset_cfg["img_size"],
            batch_size=finetune_cfg["batch_size"],
            num_workers=finetune_cfg.get("num_workers", 8),
            is_pretrain=False,
        )
    )
    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    if val_loader is None:
        raise RuntimeError("Fine-tuning requires validation split")

    checkpoint = load_checkpoint(ckpt_path, map_location="cpu")
    pretrain_cfg = checkpoint.get("config", {})
    model_cfg = pretrain_cfg.get("model", {})
    dataset_pretrain_cfg = pretrain_cfg.get("dataset", {})

    hybrid_cfg = HybridConfig(
        encoder=model_cfg["encoder"],
        img_size=dataset_pretrain_cfg.get("img_size", dataset_cfg["img_size"]),
        patch_size=model_cfg.get("patch_size", 16),
        mask_ratio=model_cfg.get("mask_ratio", 0.75),
        projector_dim=model_cfg.get("projector_dim", 256),
        projector_layers=model_cfg.get("projector_layers", 2),
        temp=model_cfg.get("temp", 0.2),
        grad_balance=False,
    )
    model = HybridModel(hybrid_cfg)
    model.load_state_dict(checkpoint["model"], strict=False)
    encoder = model.encoder.to(device)

    for name, param in encoder.named_parameters():
        param.requires_grad = True
        if finetune_cfg.get("freeze_layers") and any(layer in name for layer in finetune_cfg["freeze_layers"]):
            param.requires_grad = False

    head = nn.Linear(finetune_cfg.get("head_dim", hybrid_cfg.projector_dim if hasattr(hybrid_cfg, "projector_dim") else 256), dataset_cfg["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        [
            {"params": [p for p in encoder.parameters() if p.requires_grad], "lr": finetune_cfg["lr_backbone"]},
            {"params": head.parameters(), "lr": finetune_cfg["lr_head"]},
        ],
        weight_decay=finetune_cfg.get("weight_decay", 1e-4),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=finetune_cfg["epochs"])

    log_dir = ensure_dir(log_cfg.get("out_dir", "runs/finetune"))
    logger = Logger(log_dir, use_tensorboard=log_cfg.get("tensorboard", True), use_wandb=args.wandb or log_cfg.get("wandb", False))

    best_acc = 0.0
    for epoch in range(finetune_cfg["epochs"]):
        encoder.train()
        head.train()
        epoch_loss = 0.0
        for images, targets in tqdm(train_loader, desc=f"Finetune Epoch {epoch+1}", leave=False):
            images = images.to(device)
            targets = targets.to(device)
            feats, _ = encoder(images, return_tokens=False)
            outputs = head(feats)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg_loss = epoch_loss / max(1, len(train_loader))

        encoder.eval()
        head.eval()
        correct1 = 0.0
        total = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                feats, _ = encoder(images, return_tokens=False)
                outputs = head(feats)
                acc1, _ = accuracy(outputs, targets, topk=(1, 1))
                correct1 += acc1.item() * targets.size(0) / 100
                total += targets.size(0)
        top1 = 100.0 * correct1 / max(1, total)
        logger.log_scalars({"finetune/loss": avg_loss, "finetune/top1": top1}, epoch)
        if top1 > best_acc:
            best_acc = top1
            save_checkpoint({"encoder": encoder.state_dict(), "head": head.state_dict(), "top1": top1, "epoch": epoch}, log_dir / "best_finetune.pt")

    logger.close()


if __name__ == "__main__":
    main()
