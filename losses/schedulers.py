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


class AlphaScheduler:
    def __init__(self, cfg: AlphaSchedulerConfig) -> None:
        self.cfg = cfg

    def value(self, epoch: int) -> float:
        if epoch < self.cfg.warmup_epochs:
            start = 1.0
            end = max(self.cfg.final_alpha, 0.0)
            progress = epoch / max(1, self.cfg.warmup_epochs)
            return start + (0.6 - start) * progress  # linear to 0.6 during warmup
        progress = (epoch - self.cfg.warmup_epochs) / max(1, self.cfg.total_epochs - self.cfg.warmup_epochs)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return self.cfg.final_alpha + (0.6 - self.cfg.final_alpha) * cosine


def cosine_scheduler(base_value: float, final_value: float, epochs: int, steps_per_epoch: int) -> List[float]:
    values = []
    for i in range(epochs * steps_per_epoch):
        progress = i / max(1, epochs * steps_per_epoch - 1)
        values.append(final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * progress)))
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
