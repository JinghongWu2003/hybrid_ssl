"""Dataset modules for hybrid self-supervised learning."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from .transforms import get_eval_transform, get_mae_transform, get_simclr_transform

try:  # optional dependency
    from datasets import load_dataset  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    load_dataset = None  # type: ignore

LOGGER = logging.getLogger(__name__)


@dataclass
class DataConfig:
    name: str
    root: str
    img_size: int
    batch_size: int
    num_workers: int
    is_pretrain: bool = True
    download: bool = True
    eval_split: str = "val"


class HybridPretrainDataset(Dataset):
    """Wraps a vision dataset to provide two SimCLR views and one MAE view."""

    def __init__(
        self,
        base_dataset: Dataset,
        simclr_transform: Callable,
        mae_transform: Callable,
    ) -> None:
        self.base_dataset = base_dataset
        self.simclr_transform = simclr_transform
        self.mae_transform = mae_transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        img, target = self.base_dataset[index]
        view1 = self.simclr_transform(img)
        view2 = self.simclr_transform(img)
        mae_view = self.mae_transform(img)
        return {
            "view1": view1,
            "view2": view2,
            "image": mae_view,
            "target": torch.tensor(target) if isinstance(target, (int, float)) else target,
        }


class HybridEvalDataset(Dataset):
    """Dataset wrapper returning a single transformed image and label."""

    def __init__(self, base_dataset: Dataset, transform: Callable) -> None:
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        img, target = self.base_dataset[index]
        return self.transform(img), target


class HybridDataModule:
    """Lightweight datamodule mimicking PyTorch Lightning-style API."""

    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        self.root = Path(cfg.root)
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    def setup(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if self.cfg.is_pretrain:
            base_dataset = self._create_pretrain_dataset(self.cfg.name, train=True)
            simclr_transform = get_simclr_transform(self.cfg.img_size)
            mae_transform = get_mae_transform(self.cfg.img_size)
            self.train_dataset = HybridPretrainDataset(
                base_dataset=base_dataset,
                simclr_transform=simclr_transform,
                mae_transform=mae_transform,
            )
            self.val_dataset = None
        else:
            train_base = self._create_eval_dataset(self.cfg.name, train=True)
            val_base = self._create_eval_dataset(self.cfg.name, train=False)
            transform = get_eval_transform(self.cfg.img_size)
            self.train_dataset = HybridEvalDataset(train_base, transform)
            self.val_dataset = HybridEvalDataset(val_base, transform)

    # ------------------------------------------------------------------
    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None, "Call setup() before requesting dataloaders."
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=self.cfg.is_pretrain,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

    # ------------------------------------------------------------------
    def _create_pretrain_dataset(self, name: str, train: bool) -> Dataset:
        name = name.lower()
        if name == "imagenet100" or name == "tiny_imagenet":
            split = "train" if train else "val"
            dataset = datasets.ImageFolder(str(self.root / split))
            return dataset
        if name == "stl10":
            split = "unlabeled" if train else "test"
            return datasets.STL10(
                root=str(self.root),
                split=split,
                download=self.cfg.download,
                transform=None,
            )
        raise ValueError(f"Unsupported pretrain dataset: {name}")

    def _create_eval_dataset(self, name: str, train: bool) -> Dataset:
        name = name.lower()
        if name == "cifar10":
            return datasets.CIFAR10(
                root=str(self.root),
                train=train,
                download=self.cfg.download,
            )
        if name == "cifar100":
            return datasets.CIFAR100(
                root=str(self.root),
                train=train,
                download=self.cfg.download,
            )
        if name == "stl10":
            split = "train" if train else "test"
            return datasets.STL10(
                root=str(self.root),
                split=split,
                download=self.cfg.download,
            )
        if name == "flowers102":
            split = "train" if train else "val"
            return datasets.Flowers102(
                root=str(self.root),
                split=split,
                download=self.cfg.download,
            )
        if name == "caltech101":
            return datasets.Caltech101(
                root=str(self.root),
                download=self.cfg.download,
            )
        if name == "galaxy10-decals":
            return self._load_galaxy10(train)
        if name in {"imagenet100", "tiny_imagenet"}:
            split = "train" if train else "val"
            return datasets.ImageFolder(str(self.root / split))
        raise ValueError(f"Unsupported evaluation dataset: {name}")

    def _load_galaxy10(self, train: bool) -> Dataset:
        if load_dataset is None:
            raise RuntimeError(
                "HuggingFace datasets is not installed. Install via `pip install datasets` or provide local data."
            )
        split = "train" if train else "test"
        hf_dataset = load_dataset("galaxy10", "decals", split=split)  # type: ignore[arg-type]
        transform = get_eval_transform(self.cfg.img_size)

        class _Galaxy10Dataset(Dataset):
            def __len__(self) -> int:
                return len(hf_dataset)

            def __getitem__(self, index: int):
                sample = hf_dataset[index]
                image = sample["image"].convert("RGB")
                label = int(sample["label"])
                return transform(image), label

        return _Galaxy10Dataset()


__all__ = ["HybridDataModule", "DataConfig", "HybridPretrainDataset", "HybridEvalDataset"]
