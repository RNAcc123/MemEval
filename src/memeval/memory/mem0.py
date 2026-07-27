"""Mem0 client adapter."""

from __future__ import annotations

from typing import Any

from .base import MemoryEvent, RetrievedMemory


class Mem0Backend:
    name = "mem0"

    def __init__(self, client: Any):
        self.client = client

    def reset(self, subject: str) -> None:
        self.client.delete_all(user_id=subject)

    def add_session(self, subject: str, messages: list[dict[str, str]], metadata: dict[str, Any]) -> list[MemoryEvent]:
        raw = self.client.add(messages, user_id=subject, metadata=metadata)
        events = raw if isinstance(raw, list) else (raw or {}).get("results", [])
        return [MemoryEvent(
            event=str(item.get("event", "ADD")), memory=item.get("memory", ""),
            memory_id=item.get("id"), session_id=metadata.get("session_id"),
            timestamp=metadata.get("timestamp"), metadata=dict(item),
        ) for item in events]

    def search(self, subject: str, query: str, top_k: int) -> list[RetrievedMemory]:
        raw = self.client.search(query, filters={"user_id": subject}, top_k=top_k)
        items = raw.get("results", []) if isinstance(raw, dict) else (raw or [])
        return [RetrievedMemory(
            memory=item.get("memory", ""), score=round(item.get("score", 0) or 0, 2),
            memory_id=item.get("id"), session_id=(item.get("metadata") or {}).get("session_id"),
            timestamp=(item.get("metadata") or {}).get("timestamp"), metadata=dict(item),
        ) for item in items]
