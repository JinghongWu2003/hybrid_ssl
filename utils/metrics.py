"""Metric utilities."""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


def accuracy(output: Tensor, target: Tensor, topk: Tuple[int, ...] = (1,)) -> list[Tensor]:
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


__all__ = ["accuracy"]
