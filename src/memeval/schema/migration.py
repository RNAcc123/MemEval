"""Non-destructive migration helpers for historical diagnosis records."""

from __future__ import annotations

from typing import Any

from memeval.schema.diagnosis import DIAGNOSIS_SCHEMA_VERSION


def migrate_diagnosis_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return a current-schema copy of a legacy diagnosis record."""
    migrated = dict(record)
    migrated.setdefault("schema_version", DIAGNOSIS_SCHEMA_VERSION)
    migrated.setdefault("status", "completed")
    migrated.setdefault(
        "answer_correct",
        migrated.get("status") == "completed"
        and migrated.get("stage") == "0_consistency_check"
        and migrated.get("label") is None,
    )
    return migrated
