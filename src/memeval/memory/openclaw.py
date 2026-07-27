"""Adapter for OpenClaw implementations exposing callable operations."""

from __future__ import annotations

from typing import Any, Callable

from .base import MemoryEvent, RetrievedMemory


class OpenClawBackend:
    name = "openclaw"

    def __init__(self, *, reset_fn: Callable[[str], Any], add_fn: Callable[..., Any], search_fn: Callable[..., Any]):
        self._reset = reset_fn
        self._add = add_fn
        self._search = search_fn

    def reset(self, subject: str) -> None:
        self._reset(subject)

    def add_session(self, subject: str, messages: list[dict[str, str]], metadata: dict[str, Any]) -> list[MemoryEvent]:
        raw = self._add(subject, messages, metadata)
        items = raw if isinstance(raw, list) else []
        return [MemoryEvent(event=str(item.get("event", "UPDATE")), memory=item.get("memory", ""), memory_id=item.get("id"), session_id=metadata.get("session_id"), timestamp=metadata.get("timestamp"), metadata=dict(item)) for item in items]

    def search(self, subject: str, query: str, top_k: int) -> list[RetrievedMemory]:
        raw = self._search(subject, query, top_k)
        return [RetrievedMemory(memory=item.get("memory", ""), score=float(item.get("score", 0) or 0), memory_id=item.get("id"), timestamp=item.get("timestamp"), metadata=dict(item)) for item in (raw or [])]
