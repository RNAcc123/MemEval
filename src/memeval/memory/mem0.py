"""Mem0 client adapter."""

from __future__ import annotations

from typing import Any

from .base import MemoryEvent, RetrievedMemory
from memeval.trace.collector import TraceCollector


class Mem0Backend:
    name = "mem0"

    def __init__(self, client: Any):
        """Works with both the cloud MemoryClient and the self-hosted Memory.

        Both expose add(messages, user_id=..., metadata=...) and
        search(query, top_k=..., filters={"user_id": ...}), so no per-mode
        dispatch is needed. Note the self-hosted Memory rejects a top-level
        user_id on search and requires it inside filters.
        """
        self.client = client

    def reset(self, subject: str) -> None:
        self.client.delete_all(user_id=subject)

    def add_session(
        self,
        subject: str,
        messages: list[dict[str, str]],
        metadata: dict[str, Any],
        *,
        collector: TraceCollector | None = None,
        turn_id: str | None = None,
        parent_event_id: str | None = None,
    ) -> list[MemoryEvent]:
        event_id = collector.start_event(
            "memory_update", "add", turn_id=turn_id, parent_event_id=parent_event_id,
            input={"subject_id": subject, "messages": messages, "metadata": metadata},
        ) if collector is not None else None
        try:
            raw = self.client.add(messages, user_id=subject, metadata=metadata)
        except Exception as exc:
            if collector is not None and event_id is not None:
                collector.fail_event(event_id, exc)
            raise
        events = raw if isinstance(raw, list) else (raw or {}).get("results", [])
        normalized = [MemoryEvent(
            event=str(item.get("event", "ADD")), memory=item.get("memory", ""),
            memory_id=item.get("id"), session_id=metadata.get("session_id"),
            timestamp=metadata.get("timestamp"), metadata=dict(item),
        ) for item in events]
        if collector is not None and event_id is not None:
            collector.finish_event(
                event_id,
                output={"events": [
                    {"memory_id": item.memory_id, "event": item.event, "memory": item.memory,
                     "session_id": item.session_id, "timestamp": item.timestamp}
                    for item in normalized
                ]},
                raw={"response": raw} if isinstance(raw, dict) else {"response": {"results": raw}},
            )
        return normalized

    def search(
        self, subject: str, query: str, top_k: int, *, collector: TraceCollector | None = None,
        turn_id: str | None = None, parent_event_id: str | None = None,
    ) -> list[RetrievedMemory]:
        event_id = collector.start_event(
            "memory_retrieval", "search", turn_id=turn_id, parent_event_id=parent_event_id,
            input={"subject_id": subject, "query": query, "top_k": top_k},
        ) if collector is not None else None
        try:
            raw = self.client.search(query, filters={"user_id": subject}, top_k=top_k)
        except Exception as exc:
            if collector is not None and event_id is not None:
                collector.fail_event(event_id, exc)
            raise
        items = raw.get("results", []) if isinstance(raw, dict) else (raw or [])
        normalized = [RetrievedMemory(
            memory=item.get("memory", ""), score=round(item.get("score", 0) or 0, 2),
            memory_id=item.get("id"), session_id=(item.get("metadata") or {}).get("session_id"),
            timestamp=(item.get("metadata") or {}).get("timestamp"), metadata=dict(item),
        ) for item in items]
        if collector is not None and event_id is not None:
            collector.finish_event(
                event_id,
                output={"candidates": [
                    {"memory_id": item.memory_id, "memory": item.memory, "score": item.score,
                     "rank": rank, "selected": False, "session_id": item.session_id,
                     "timestamp": item.timestamp}
                    for rank, item in enumerate(normalized, start=1)
                ]},
                raw={"response": raw} if isinstance(raw, dict) else {"response": {"results": raw}},
            )
        return normalized
