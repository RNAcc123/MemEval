"""Memory backend contracts and normalized records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class MemoryEvent:
    event: str
    memory: str = ""
    memory_id: str | None = None
    session_id: str | None = None
    timestamp: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedMemory:
    memory: str
    score: float = 0.0
    memory_id: str | None = None
    session_id: str | None = None
    timestamp: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryBackend(Protocol):
    name: str

    def reset(self, subject: str) -> None: ...
    def add_session(self, subject: str, messages: list[dict[str, str]], metadata: dict[str, Any]) -> list[MemoryEvent]: ...
    def search(self, subject: str, query: str, top_k: int) -> list[RetrievedMemory]: ...
