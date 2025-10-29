"""Augmentation utilities for hybrid MAE + SimCLR training."""
from __future__ import annotations

from typing import Callable, Tuple

import torch
from torchvision import transforms


def get_normalization(img_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if img_size >= 96:
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
    else:
        mean = torch.tensor([0.4914, 0.4822, 0.4465])
        std = torch.tensor([0.2470, 0.2435, 0.2616])
    return mean, std


def get_simclr_transform(img_size: int) -> Callable:
    mean, std = get_normalization(img_size)
    augment = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=img_size // 10 * 2 + 1, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean.tolist(), std.tolist()),
        ]
    )
    return augment


def get_mae_transform(img_size: int) -> Callable:
    mean, std = get_normalization(img_size)
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean.tolist(), std.tolist()),
        ]
    )


def get_eval_transform(img_size: int) -> Callable:
    mean, std = get_normalization(img_size)
    return transforms.Compose(
        [
            transforms.Resize(int(img_size * 256 / 224)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean.tolist(), std.tolist()),
        ]
    )
