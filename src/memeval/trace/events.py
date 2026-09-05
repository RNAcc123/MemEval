"""Versioned, framework-neutral memory execution events."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


TRACE_V2_SCHEMA_VERSION = "2.0"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class TraceEvent:
    event_id: str
    trace_id: str
    stage: str
    operation: str
    sequence: int
    timestamp: str = field(default_factory=utc_now)
    turn_id: str | None = None
    parent_event_id: str | None = None
    status: str = "completed"
    input: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TraceEnvelope:
    trace_id: str
    sample_id: str
    subject_id: str
    framework: str
    qa: dict[str, Any] = field(default_factory=dict)
    run_id: str | None = None
    conversation_id: str | None = None
    adapter_version: str | None = None
    observability_level: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)
    events: list[TraceEvent] = field(default_factory=list)
    schema_version: str = TRACE_V2_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["events"] = [event.to_dict() for event in self.events]
        return value
