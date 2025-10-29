"""Checkpoint utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

from .common import ensure_dir


def save_checkpoint(state: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(state, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)


__all__ = ["save_checkpoint", "load_checkpoint"]
