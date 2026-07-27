"""Deterministic coverage and matching metrics."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


def record_key(record: dict[str, Any]) -> str | None:
    value = record.get("record_id") or record.get("conv_id_question_id") or record.get("question_id")
    return str(value) if value is not None and str(value) else None


def compare_records(expected: Iterable[dict[str, Any]], actual: Iterable[dict[str, Any]]) -> dict[str, Any]:
    expected_map = {key: item for item in expected if (key := record_key(item)) is not None}
    actual_records = [item for item in actual if record_key(item) is not None]
    actual_keys = [record_key(item) for item in actual_records]
    counts = Counter(actual_keys)
    matched = {key for key in expected_map if counts[key] == 1}
    duplicate = sorted(key for key, count in counts.items() if count > 1)
    missing = sorted(set(expected_map) - set(actual_keys))
    unexpected = sorted(set(actual_keys) - set(expected_map))
    invalid = sum(1 for item in actual_records if item.get("status") == "error")
    return {
        "expected_records": len(expected_map),
        "actual_records": len(actual_records),
        "matched_records": len(matched),
        "missing_records": len(missing),
        "duplicate_records": len(duplicate),
        "unexpected_records": len(unexpected),
        "invalid_records": invalid,
        "coverage": round(len(matched) / len(expected_map), 6) if expected_map else 0.0,
        "missing_ids": missing,
        "duplicate_ids": duplicate,
        "unexpected_ids": unexpected,
    }
