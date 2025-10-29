"""Logging utilities."""
from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Dict, Optional

from torch.utils.tensorboard import SummaryWriter

try:  # optional
    import wandb
except Exception:  # pragma: no cover - optional dependency
    wandb = None  # type: ignore


class Logger:
    def __init__(self, log_dir: Path, use_tensorboard: bool = True, use_wandb: bool = False, project: str = "hybrid-ssl") -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer: Optional[SummaryWriter] = None
        if use_tensorboard:
            self.writer = SummaryWriter(log_dir=str(log_dir))
        self.use_wandb = use_wandb and wandb is not None
        if self.use_wandb:
            wandb.init(project=project, dir=str(log_dir), config={})

    def log_scalars(self, scalars: Dict[str, float], step: int) -> None:
        if self.writer is not None:
            for key, value in scalars.items():
                self.writer.add_scalar(key, value, step)
        if self.use_wandb:
            wandb.log({**scalars, "step": step})

    def log_images(self, tag: str, images, step: int) -> None:
        if self.writer is not None:
            self.writer.add_images(tag, images, step)
        if self.use_wandb:
            wandb.log({tag: [wandb.Image(img) for img in images], "step": step})

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
        if self.use_wandb:
            wandb.finish()

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = ["Logger"]
