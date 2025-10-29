"""Minimal distributed utilities."""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterable

import torch
import torch.distributed as dist


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def barrier() -> None:
    if is_dist_avail_and_initialized():  # pragma: no cover - depends on DDP
        dist.barrier()


def setup_distributed(port: int | None = None) -> None:
    if "RANK" not in os.environ:
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    init_method = f"tcp://127.0.0.1:{port or 29500}"
    dist.init_process_group(backend=backend, init_method=init_method)


@contextmanager
def distributed_zero_first(local_rank: int):
    if local_rank not in (0, -1):
        barrier()
    yield
    if local_rank == 0:
        barrier()


def concat_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    if get_world_size() == 1:
        return tensor
    tensors_gather = [torch.zeros_like(tensor) for _ in range(get_world_size())]
    dist.all_gather(tensors_gather, tensor)
    return torch.cat(tensors_gather, dim=0)


__all__ = [
    "is_dist_avail_and_initialized",
    "get_world_size",
    "get_rank",
    "barrier",
    "setup_distributed",
    "distributed_zero_first",
    "concat_all_gather",
]
