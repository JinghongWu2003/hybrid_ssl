"""Visualization utilities for reconstructions and embeddings."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from torchvision.utils import make_grid, save_image

from data.datamodules import DataConfig, HybridDataModule
from models.hybrid_model import HybridConfig, HybridModel
from utils.checkpoint import load_checkpoint
from utils.common import ensure_dir, load_config
from utils.mask import patchify, unpatchify


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualization for Hybrid SSL")
    parser.add_argument("--config", type=str, required=True, help="Training config used for pretraining")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint")
    parser.add_argument("--out", type=str, default=None, help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    dataset_cfg = cfg["dataset"]
    model_cfg = cfg["model"]
    loss_cfg = cfg.get("loss", {})
    train_cfg = cfg["train"]

    out_dir = ensure_dir(args.out or Path(cfg.get("logging", {}).get("out_dir", "runs/visualize")) / "figs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dm = HybridDataModule(
        DataConfig(
            name=dataset_cfg["name"],
            root=dataset_cfg["root"],
            img_size=dataset_cfg["img_size"],
            batch_size=min(32, train_cfg["batch_size"]),
            num_workers=train_cfg.get("num_workers", 4),
            is_pretrain=True,
        )
    )
    dm.setup()
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

    hybrid_cfg = HybridConfig(
        encoder=model_cfg["encoder"],
        img_size=dataset_cfg["img_size"],
        patch_size=model_cfg["patch_size"],
        mask_ratio=model_cfg.get("mask_ratio", 0.75),
        projector_dim=model_cfg.get("projector_dim", 256),
        projector_layers=model_cfg.get("projector_layers", 2),
        temp=model_cfg.get("temp", 0.2),
        grad_balance=False,
    )
    model = HybridModel(hybrid_cfg).to(device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()

    alpha = loss_cfg.get("alpha_final", 0.5)
    with torch.no_grad():
        outputs = model(batch, alpha)
    recon_patches = outputs["recon"]
    mask = outputs["mask"].cpu()
    images = batch["image"].cpu()
    patch_size = model.decoder.patch_size
    preds = unpatchify(recon_patches.cpu(), patch_size, images.size(1))
    grid = make_grid(torch.cat([images, preds], dim=0), nrow=images.size(0))
    save_image(grid, out_dir / "reconstructions.png")

    # Embedding visualization via t-SNE
    z = outputs["z1"].cpu().numpy()
    h = outputs["h1"].cpu().numpy()
    tsne_z = TSNE(n_components=2, perplexity=min(30, z.shape[0] - 1)).fit_transform(z)
    tsne_h = TSNE(n_components=2, perplexity=min(30, h.shape[0] - 1)).fit_transform(h)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(tsne_z[:, 0], tsne_z[:, 1], s=10, c="tab:blue")
    axes[0].set_title("Encoder features")
    axes[1].scatter(tsne_h[:, 0], tsne_h[:, 1], s=10, c="tab:orange")
    axes[1].set_title("Projector features")
    plt.tight_layout()
    fig.savefig(out_dir / "tsne.png")
    plt.close(fig)

    # Attention map proxy using token magnitude
    _, tokens = model.encoder(batch["image"], return_tokens=True)
    if tokens is not None:
        attn = tokens.norm(dim=-1).view(images.size(0), int(tokens.size(1) ** 0.5), -1)
        attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-6)
        fig, axes = plt.subplots(1, min(4, images.size(0)), figsize=(12, 3))
        if len(axes.shape) == 0:
            axes = [axes]
        for idx, ax in enumerate(axes):
            heatmap = attn[idx].cpu().numpy()
            ax.imshow(heatmap, cmap="magma")
            ax.axis("off")
        plt.tight_layout()
        fig.savefig(out_dir / "attention_maps.png")
        plt.close(fig)


if __name__ == "__main__":
    main()
