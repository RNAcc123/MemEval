"""File-level matching facade used during legacy analysis migration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from memeval.storage import load_records

from .metrics import compare_records


def compare_files(expected_path: Path, actual_path: Path) -> dict[str, Any]:
    """Compare legacy JSON arrays or JSONL records and return structured coverage."""
    return compare_records(load_records(expected_path), load_records(actual_path))
