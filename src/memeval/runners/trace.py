"""Shared trace-runner utilities independent of dataset and backend details."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def retry_call(fn: Callable[[], T], *, retries: int = 3, delay_seconds: float = 2.0) -> T:
    """Retry a backend operation while preserving the final exception."""
    if retries < 1:
        raise ValueError("retries must be at least 1")
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - backend-specific exceptions are normalized at the boundary.
            last_error = exc
            if attempt < retries - 1:
                time.sleep(delay_seconds)
    assert last_error is not None
    raise last_error


def validate_range(start: int, end: int, total: int) -> None:
    if start < 0 or end < start or end > total:
        raise ValueError(f"Invalid range start={start}, end={end}, dataset size={total}")


def part_location(global_index: int, part_size: int, prefix: str) -> tuple[int, str, str]:
    if part_size <= 0:
        raise ValueError("part_size must be positive")
    part_id = global_index // part_size + 1
    return part_id, str(global_index % part_size), f"{prefix}_part{part_id}.json"
