"""In-memory event collector; persistence is intentionally handled elsewhere."""

from __future__ import annotations

import itertools
from typing import Any

from memeval.trace.events import TraceEnvelope, TraceEvent, utc_now


class TraceCollector:
    """Collect causally-linked events for one sample execution."""

    def __init__(self, envelope: TraceEnvelope):
        self.envelope = envelope
        self._sequence = itertools.count(1)
        self._event_counter = itertools.count(1)

    def start_event(
        self,
        stage: str,
        operation: str,
        *,
        input: dict[str, Any] | None = None,
        turn_id: str | None = None,
        parent_event_id: str | None = None,
    ) -> str:
        event_id = f"{self.envelope.trace_id}:evt_{next(self._event_counter):04d}"
        self.envelope.events.append(
            TraceEvent(
                event_id=event_id,
                trace_id=self.envelope.trace_id,
                stage=stage,
                operation=operation,
                sequence=next(self._sequence),
                timestamp=utc_now(),
                turn_id=turn_id,
                parent_event_id=parent_event_id,
                status="running",
                input=dict(input or {}),
            )
        )
        return event_id

    def _get(self, event_id: str) -> TraceEvent:
        for event in self.envelope.events:
            if event.event_id == event_id:
                return event
        raise KeyError(f"Unknown event_id: {event_id}")

    def finish_event(
        self,
        event_id: str,
        *,
        output: dict[str, Any] | None = None,
        raw: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        event = self._get(event_id)
        if event.status != "running":
            raise ValueError(f"Event {event_id} is already {event.status}")
        event.status = "completed"
        event.output = dict(output or {})
        event.raw = dict(raw or {})
        event.metrics = dict(metrics or {})

    def fail_event(self, event_id: str, error: Exception | str) -> None:
        event = self._get(event_id)
        if event.status != "running":
            raise ValueError(f"Event {event_id} is already {event.status}")
        event.status = "error"
        event.error = {"type": type(error).__name__, "message": str(error)}

    def event(self, event_id: str) -> TraceEvent:
        return self._get(event_id)

    @property
    def latest_event_id(self) -> str | None:
        return self.envelope.events[-1].event_id if self.envelope.events else None

    def to_dict(self) -> dict[str, Any]:
        return self.envelope.to_dict()
