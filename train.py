"""Joint pretraining entry script."""
from __future__ import annotations

import argparse
from typing import Dict

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from tqdm import tqdm

from data.datamodules import DataConfig, HybridDataModule
from losses.schedulers import AlphaScheduler, AlphaSchedulerConfig, cosine_scheduler
from models.hybrid_model import HybridConfig, HybridModel
from utils.checkpoint import save_checkpoint
from utils.common import ensure_dir, load_config
from utils.logging import Logger
from utils.seed import SeedConfig, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid MAE + SimCLR pretraining")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic mode")
    return parser.parse_args()


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    seed_cfg = SeedConfig(seed=cfg.get("seed", 42), deterministic=args.deterministic)
    seed_everything(seed_cfg)

    dataset_cfg = cfg["dataset"]
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    optim_cfg = cfg.get("optim", {"lr": 1e-4, "weight_decay": 0.05})
    loss_cfg = cfg.get("loss", {})
    log_cfg = cfg.get("logging", {"out_dir": "runs/default", "tensorboard": True})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dm = HybridDataModule(
        DataConfig(
            name=dataset_cfg["name"],
            root=dataset_cfg["root"],
            img_size=dataset_cfg["img_size"],
            batch_size=train_cfg["batch_size"],
            num_workers=train_cfg.get("num_workers", 8),
            is_pretrain=True,
        )
    )
    dm.setup()
    train_loader = dm.train_dataloader()

    hybrid_cfg = HybridConfig(
        encoder=model_cfg["encoder"],
        img_size=dataset_cfg["img_size"],
        patch_size=model_cfg["patch_size"],
        mask_ratio=model_cfg.get("mask_ratio", 0.75),
        projector_dim=model_cfg.get("projector_dim", 256),
        projector_layers=model_cfg.get("projector_layers", 2),
        temp=model_cfg.get("temp", 0.2),
        grad_balance=train_cfg.get("grad_balance", False),
    )
    model = HybridModel(hybrid_cfg).to(device)

    def _as_float(value: float | str, *, name: str) -> float:
        try:
            return float(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - configuration error
            raise TypeError(f"Optimizer hyperparameter '{name}' must be a real number, got {value!r}.") from exc

    raw_betas = optim_cfg.get("betas", (0.9, 0.95))
    if not isinstance(raw_betas, (list, tuple)):
        raise TypeError(
            "Optimizer hyperparameter 'betas' must be provided as a sequence of two numbers."
        )
    betas = tuple(_as_float(beta, name="betas") for beta in raw_betas)
    if len(betas) != 2:
        raise ValueError("Optimizer hyperparameter 'betas' must contain exactly two values.")

    lr = _as_float(optim_cfg.get("lr", 1e-4), name="lr")
    final_lr = _as_float(optim_cfg.get("final_lr", 1e-6), name="final_lr")
    weight_decay = _as_float(optim_cfg.get("weight_decay", 0.05), name="weight_decay")

    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
    )
    scaler = GradScaler(enabled=train_cfg.get("amp", True) and device.type == "cuda")

    epochs = train_cfg["epochs"]
    steps_per_epoch = len(train_loader)
    lr_schedule = cosine_scheduler(
        base_value=lr,
        final_value=final_lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
    )
    alpha_scheduler = AlphaScheduler(
        AlphaSchedulerConfig(
            warmup_epochs=loss_cfg.get("alpha_warmup_epochs", 10),
            final_alpha=loss_cfg.get("alpha_final", 0.5),
            total_epochs=epochs,
        )
    )
    log_dir = ensure_dir(log_cfg.get("out_dir", "runs/default"))
    logger = Logger(log_dir, use_tensorboard=log_cfg.get("tensorboard", True), use_wandb=args.wandb or log_cfg.get("wandb", False))

    alpha_values = []
    best_loss = float("inf")
    global_step = 0
    grad_clip_value = None
    if train_cfg.get("grad_clip") is not None:
        grad_clip_value = _as_float(train_cfg.get("grad_clip"), name="grad_clip")

    for epoch in range(epochs):
        model.train()
        alpha = alpha_scheduler.value(epoch)
        epoch_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        current_lr = lr_schedule[min(global_step, len(lr_schedule) - 1)] if lr_schedule else lr
        for step, batch in enumerate(progress):
            batch = to_device(batch, device)
            lr = lr_schedule[global_step]
            current_lr = lr
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            with autocast(enabled=train_cfg.get("amp", True) and device.type == "cuda"):
                outputs = model(batch, alpha)
                loss = outputs["loss_total"]
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if grad_clip_value is not None:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), grad_clip_value)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            logger.log_scalars(
                {
                    "train/loss_total": loss.item(),
                    "train/loss_rec": outputs["loss_rec"].item(),
                    "train/loss_contrast": outputs["loss_contrast"].item(),
                    "train/lr": lr,
                    "train/alpha": alpha,
                },
                step=global_step,
            )
            global_step += 1
        avg_loss = epoch_loss / max(1, steps_per_epoch)
        alpha_values.append(alpha)
        logger.log_scalars(
            {
                "train/epoch_loss": avg_loss,
                "train/epoch_alpha": alpha,
            },
            step=epoch,
        )
        print(
            f"[Epoch {epoch + 1}/{epochs}] loss_total={avg_loss:.4f} "
            f"alpha={alpha:.4f} lr={current_lr:.4e}",
            flush=True,
        )
        checkpoint_path = log_dir / f"epoch_{epoch:03d}.pt"
        save_checkpoint(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "config": cfg,
            },
            checkpoint_path,
        )
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "config": cfg,
                },
                log_dir / "best.pt",
            )

    # Save alpha schedule plot
    try:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(range(epochs), alpha_values)
        plt.xlabel("Epoch")
        plt.ylabel("Alpha")
        plt.title("Alpha Schedule")
        plt.tight_layout()
        plt.savefig(log_dir / "alpha_schedule.png")
        plt.close()
    except Exception as exc:  # pragma: no cover - matplotlib optional in tests
        print(f"Failed to save alpha plot: {exc}")

    logger.close()


if __name__ == "__main__":
    main()
