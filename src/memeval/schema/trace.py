"""Validation for legacy and versioned MemEval trace datasets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


TRACE_SCHEMA_VERSION = "1.0"
REQUIRED_TRACE_FIELDS = {"qa_question", "qa_answer", "qa_response"}


class TraceValidationError(ValueError):
    """Raised when trace input does not satisfy the supported contract."""


def _validate_record(record: object, location: str) -> None:
    if not isinstance(record, Mapping):
        raise TraceValidationError(f"{location} must be an object")
    missing = sorted(REQUIRED_TRACE_FIELDS - record.keys())
    if missing:
        raise TraceValidationError(f"{location} missing required fields: {', '.join(missing)}")
    if "subjects" in record:
        subjects = record["subjects"]
        if not isinstance(subjects, list):
            raise TraceValidationError(f"{location}.subjects must be a list")
        for index, subject in enumerate(subjects):
            subject_location = f"{location}.subjects[{index}]"
            if not isinstance(subject, Mapping):
                raise TraceValidationError(f"{subject_location} must be an object")
            for list_field in ("memories", "retrieval"):
                value = subject.get(list_field, [])
                if not isinstance(value, list):
                    raise TraceValidationError(f"{subject_location}.{list_field} must be a list")
        return
    # Legacy shape: fixed person1/person2 + speaker_1_memories/speaker_2_memories keys.
    for person_key in ("person1", "person2"):
        person = record.get(person_key, {})
        if person is not None and not isinstance(person, Mapping):
            raise TraceValidationError(f"{location}.{person_key} must be an object")
        memories = (person or {}).get("memories", [])
        if not isinstance(memories, list):
            raise TraceValidationError(f"{location}.{person_key}.memories must be a list")
    for retrieval_key in ("speaker_1_memories", "speaker_2_memories"):
        if retrieval_key in record and not isinstance(record[retrieval_key], list):
            raise TraceValidationError(f"{location}.{retrieval_key} must be a list")


def validate_trace_dataset(data: object) -> Mapping[str, Sequence[Mapping[str, Any]]]:
    """Validate a legacy conversation mapping or a versioned trace envelope."""
    if not isinstance(data, Mapping):
        raise TraceValidationError("Trace dataset must be an object")

    if "schema_version" in data:
        if data.get("schema_version") != TRACE_SCHEMA_VERSION:
            raise TraceValidationError(
                f"Unsupported trace schema_version: {data.get('schema_version')!r}; expected {TRACE_SCHEMA_VERSION!r}"
            )
        records = data.get("records")
        if not isinstance(records, list):
            raise TraceValidationError("Versioned trace dataset requires a records list")
        normalized: dict[str, list[Mapping[str, Any]]] = {"records": records}
    else:
        normalized = dict(data)

    for conversation_id, records in normalized.items():
        if not isinstance(records, list):
            raise TraceValidationError(f"Conversation {conversation_id!r} must contain a list of QA records")
        for index, record in enumerate(records):
            _validate_record(record, f"{conversation_id}[{index}]")
    return normalized
