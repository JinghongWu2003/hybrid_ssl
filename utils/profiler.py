"""Simple profiling helpers."""
from __future__ import annotations

import contextlib
import time
from typing import Dict


@contextlib.contextmanager
def profile(section: str, stats: Dict[str, float]) -> None:
    start = time.time()
    yield
    elapsed = time.time() - start
    stats[section] = stats.get(section, 0.0) + elapsed


__all__ = ["profile"]
