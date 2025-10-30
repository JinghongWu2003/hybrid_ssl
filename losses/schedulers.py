"""Schedulers for loss weighting and learning rate."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch
from torch import nn


@dataclass
class AlphaSchedulerConfig:
    warmup_epochs: int
    final_alpha: float
    total_epochs: int
    start_alpha: float = 1.0


class AlphaScheduler:
    def __init__(self, cfg: AlphaSchedulerConfig) -> None:
        self.cfg = cfg

    def value(self, epoch: int) -> float:
        warmup_epochs = max(0, self.cfg.warmup_epochs)
        start = getattr(self.cfg, "start_alpha", 1.0)
        end = self.cfg.final_alpha

        if warmup_epochs > 0 and epoch < warmup_epochs:
            progress = epoch / warmup_epochs
            return start + (end - start) * progress

        remaining_epochs = max(0, self.cfg.total_epochs - warmup_epochs)
        if remaining_epochs == 0:
            return end

        progress = (epoch - warmup_epochs) / remaining_epochs
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        start_cosine = end if warmup_epochs > 0 else start
        return end + (start_cosine - end) * cosine


def cosine_scheduler(
    base_value: float,
    final_value: float,
    epochs: int,
    steps_per_epoch: int,
    warmup_epochs: int = 0,
    start_warmup_value: float = 0.0,
) -> List[float]:
    total_steps = max(0, epochs * steps_per_epoch)
    warmup_steps = max(0, min(warmup_epochs * steps_per_epoch, total_steps))
    values: List[float] = []

    if warmup_steps > 0:
        for i in range(warmup_steps):
            progress = i / max(1, warmup_steps - 1)
            values.append(start_warmup_value + progress * (base_value - start_warmup_value))

    remaining_steps = total_steps - warmup_steps
    if remaining_steps > 0:
        for i in range(remaining_steps):
            progress = i / max(1, remaining_steps - 1)
            value = final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * progress))
            values.append(value)

    return values


def gradient_balance(losses: Dict[str, torch.Tensor], parameters: Iterable[nn.Parameter]) -> Dict[str, torch.Tensor]:
    params = [p for p in parameters if p.requires_grad]
    if not params:
        return losses
    grads = {}
    for name, loss in losses.items():
        grad = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        sq_sum = 0.0
        for g in grad:
            if g is not None:
                sq_sum += float(g.pow(2).sum().item())
        grads[name] = math.sqrt(sq_sum) if sq_sum > 0 else 1.0
    norm = sum(grads.values())
    if norm == 0:
        return losses
    balanced = {name: loss / (grads[name] / norm) for name, loss in losses.items()}
    return balanced


__all__ = ["AlphaScheduler", "AlphaSchedulerConfig", "cosine_scheduler", "gradient_balance"]
