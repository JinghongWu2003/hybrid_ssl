"""Positional embedding utilities."""
from __future__ import annotations

from typing import Iterable

import numpy as np


def _get_1d_sincos_pos_embed(embed_dim: int, positions: np.ndarray) -> np.ndarray:
    if embed_dim % 2 != 0:
        raise ValueError("Embedding dimension must be even for sin-cos positional encoding.")
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000 ** omega)
    positions = positions.reshape(-1)
    out = np.einsum("m,d->md", positions, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def _get_2d_sincos_pos_embed(embed_dim: int, grid: Iterable[np.ndarray]) -> np.ndarray:
    half_dim = embed_dim // 2
    return np.concatenate(
        [_get_1d_sincos_pos_embed(half_dim, g) for g in grid],
        axis=1,
    )


def build_2d_sincos_pos_embed(embed_dim: int, grid_size: int | tuple[int, int]) -> np.ndarray:
    """Build 2D sine-cosine positional embeddings as in MAE."""

    if embed_dim % 2 != 0:
        raise ValueError("Embedding dimension must be even for 2D sin-cos embeddings.")

    if isinstance(grid_size, int):
        grid_h = grid_w = grid_size
    else:
        if len(grid_size) != 2:
            raise ValueError("grid_size tuple must have length 2")
        grid_h, grid_w = grid_size

    grid_w_values = np.arange(grid_w, dtype=np.float32)
    grid_h_values = np.arange(grid_h, dtype=np.float32)
    grid = np.meshgrid(grid_w_values, grid_h_values)
    grid = [g.reshape(-1) for g in grid]

    pos_embed = _get_2d_sincos_pos_embed(embed_dim, grid)
    return pos_embed


__all__ = ["build_2d_sincos_pos_embed"]
