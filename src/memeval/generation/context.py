"""Select retrieved memories and record the exact generation context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from memeval.memory.base import RetrievedMemory
from memeval.trace.collector import TraceCollector


@dataclass(frozen=True)
class AssembledContext:
    event_id: str
    text: str
    items: list[dict]


def _default_formatter(memory: RetrievedMemory, position: int) -> str:
    return f"[{position}] {memory.memory}"


def assemble_context(
    query: str,
    memories: list[RetrievedMemory],
    collector: TraceCollector,
    *,
    limit: int | None = None,
    parent_event_id: str | None = None,
    turn_id: str | None = None,
    formatter: Callable[[RetrievedMemory, int], str] = _default_formatter,
) -> AssembledContext:
    """Select memories in retrieval order and record their rendered prompt text."""
    selected = memories if limit is None else memories[:limit]
    event_id = collector.start_event(
        "context_assembly",
        "assemble",
        parent_event_id=parent_event_id,
        turn_id=turn_id,
        input={"query": query, "candidate_count": len(memories), "limit": limit},
    )
    items = [
        {
            "memory_id": memory.memory_id,
            "memory": memory.memory,
            "position": position,
            "score": memory.score,
            "session_id": memory.session_id,
            "timestamp": memory.timestamp,
            "source": "memory_retrieval",
        }
        for position, memory in enumerate(selected, start=1)
    ]
    text = "\n".join(formatter(memory, position) for position, memory in enumerate(selected, start=1))
    collector.finish_event(
        event_id,
        output={"memory_ids": [item["memory_id"] for item in items], "context_items": items, "context_text": text},
        metrics={"candidate_count": len(memories), "selected_count": len(items)},
    )
    return AssembledContext(event_id=event_id, text=text, items=items)
