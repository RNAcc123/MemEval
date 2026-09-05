"""Deterministic in-memory backend used for runner tests and dry runs."""

from __future__ import annotations

from memeval.memory.base import MemoryEvent, RetrievedMemory
from memeval.trace import TraceCollector


class FakeMemoryBackend:
    name = "fake"

    def __init__(self):
        self._memories: dict[str, list[MemoryEvent]] = {}

    def reset(self, subject: str) -> None:
        self._memories[subject] = []

    def add_session(self, subject, messages, metadata, *, collector: TraceCollector | None = None, turn_id=None):
        event_id = collector.start_event(
            "memory_update", "add", turn_id=turn_id, input={"messages": messages, "metadata": metadata},
        ) if collector is not None else None
        created = []
        for index, message in enumerate(messages):
            memory = MemoryEvent("ADD", message["content"], f"fake-{len(self._memories.get(subject, [])) + index}", metadata.get("session_id"), metadata.get("timestamp"), metadata)
            created.append(memory)
        self._memories.setdefault(subject, []).extend(created)
        if collector is not None and event_id is not None:
            collector.finish_event(event_id, output={"events": [
                {"memory_id": item.memory_id, "event": item.event, "memory": item.memory, "session_id": item.session_id, "timestamp": item.timestamp}
                for item in created
            ]})
        return created

    def search(self, subject, query, top_k, *, collector: TraceCollector | None = None, turn_id=None):
        event_id = collector.start_event(
            "memory_retrieval", "search", turn_id=turn_id, input={"query": query, "top_k": top_k},
        ) if collector is not None else None
        results = [RetrievedMemory(item.memory, 1.0 / (index + 1), item.memory_id, item.session_id, item.timestamp) for index, item in enumerate(self._memories.get(subject, []))][:top_k]
        if collector is not None and event_id is not None:
            collector.finish_event(event_id, output={"candidates": [
                {"memory_id": item.memory_id, "memory": item.memory, "score": item.score, "rank": index, "selected": False,
                 "session_id": item.session_id, "timestamp": item.timestamp}
                for index, item in enumerate(results, 1)
            ]})
        return results
