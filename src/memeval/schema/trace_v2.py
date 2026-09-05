"""Validation for the event-based trace v2 envelope."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from memeval.trace.events import TRACE_V2_SCHEMA_VERSION


class TraceV2ValidationError(ValueError):
    """Raised when an event trace violates the v2 contract."""


def validate_trace_v2(data: object) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise TraceV2ValidationError("Trace v2 must be an object")
    if data.get("schema_version") != TRACE_V2_SCHEMA_VERSION:
        raise TraceV2ValidationError("Unsupported or missing schema_version")
    for field in ("trace_id", "sample_id", "subject_id", "framework"):
        if not isinstance(data.get(field), str) or not data[field]:
            raise TraceV2ValidationError(f"{field} must be a non-empty string")
    events = data.get("events")
    if not isinstance(events, list):
        raise TraceV2ValidationError("events must be a list")
    ids: set[str] = set()
    sequences: list[int] = []
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise TraceV2ValidationError(f"events[{index}] must be an object")
        for field in ("event_id", "trace_id", "stage", "operation", "status"):
            if not isinstance(event.get(field), str) or not event[field]:
                raise TraceV2ValidationError(f"events[{index}].{field} is required")
        event_id = event["event_id"]
        if event_id in ids:
            raise TraceV2ValidationError(f"duplicate event_id: {event_id}")
        ids.add(event_id)
        sequence = event.get("sequence")
        if not isinstance(sequence, int):
            raise TraceV2ValidationError(f"events[{index}].sequence must be an integer")
        sequences.append(sequence)
        if event["trace_id"] != data["trace_id"]:
            raise TraceV2ValidationError(f"events[{index}] has a different trace_id")
    if sequences != sorted(sequences) or len(set(sequences)) != len(sequences):
        raise TraceV2ValidationError("event sequence must be strictly increasing")
    missing_parents = {
        event["parent_event_id"]
        for event in events
        if event.get("parent_event_id") and event["parent_event_id"] not in ids
    }
    if missing_parents:
        raise TraceV2ValidationError(f"Unknown parent event(s): {sorted(missing_parents)}")
    return data
