"""A-Mem (agentic_memory) backend.

A-Mem has no tenant scoping: the collection name is hardcoded and
``AgenticMemorySystem.__init__`` resets the Chroma client globally. We therefore
build one system per subject, each pinned to its own persist directory.

``add_note`` returns only a memory id, so memory-update events are recovered by
diffing the in-process note store around each call, in the same spirit as the
OpenCLAW workspace diff.

The evolution step (deciding whether a new note should update an existing
neighbor) calls an LLM through A-Mem's own ``LLMController``. ``api_key`` and
``base_url`` let that call go through a private key or an OpenAI-compatible
gateway instead of relying on ambient ``OPENAI_API_KEY``/``OPENAI_BASE_URL``
process environment variables; see ``_openai_base_url_override`` for why
``base_url`` still has to go through the environment.
"""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from memeval.memory.base import MemoryEvent, RetrievedMemory
from memeval.trace.collector import TraceCollector


_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

# e.g. "1:56 pm on 8 May, 2023"
_LOCOMO_TS = re.compile(
    r"(?P<hour>\d{1,2}):(?P<minute>\d{2})\s*(?P<ampm>am|pm)?\s*on\s*"
    r"(?P<day>\d{1,2})\s+(?P<month>[A-Za-z]+),?\s*(?P<year>\d{4})",
    re.IGNORECASE,
)


def to_amem_timestamp(value: str) -> str:
    """Convert a LoCoMo date string to A-Mem's YYYYMMDDHHmm, or '' if unparseable."""
    if not value:
        return ""
    match = _LOCOMO_TS.search(str(value))
    if not match:
        return ""
    hour = int(match.group("hour"))
    ampm = (match.group("ampm") or "").lower()
    if ampm == "pm" and hour != 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0
    month = _MONTHS.get(match.group("month")[:3].lower())
    if month is None:
        return ""
    return f"{int(match.group('year')):04d}{month:02d}{int(match.group('day')):02d}{hour:02d}{int(match.group('minute')):02d}"


@dataclass(frozen=True)
class _NoteSnapshot:
    """The mutable fields A-Mem's evolution step can rewrite."""

    content: str
    context: str
    tags: tuple[str, ...]
    links: tuple[str, ...]

    @classmethod
    def of(cls, note: Any) -> "_NoteSnapshot":
        return cls(
            content=str(getattr(note, "content", "") or ""),
            context=str(getattr(note, "context", "") or ""),
            tags=tuple(str(tag) for tag in (getattr(note, "tags", None) or [])),
            links=tuple(str(link) for link in (getattr(note, "links", None) or [])),
        )


def snapshot_notes(system: Any) -> dict[str, _NoteSnapshot]:
    return {str(key): _NoteSnapshot.of(note) for key, note in getattr(system, "memories", {}).items()}


def memory_events_from_diff(
    before: dict[str, _NoteSnapshot],
    after: dict[str, _NoteSnapshot],
    *,
    added_id: str | None,
    session_id: str,
    timestamp: str,
) -> list[dict[str, Any]]:
    """Recover ADD/UPDATE events from the note store around one add_note call.

    The newly added note is reported as ADD; any pre-existing note whose tags,
    context or links changed is reported as UPDATE, which is how A-Mem's
    ``strengthen`` and ``update_neighbor`` actions surface to an outside caller.
    """
    events: list[dict[str, Any]] = []
    for note_id in sorted(set(after) - set(before)):
        note = after[note_id]
        events.append({
            "id": note_id,
            "event": "ADD",
            "memory": note.content,
            "tags": list(note.tags),
            "context": note.context,
            "links": list(note.links),
            "timestamp": timestamp,
            "session_id": session_id,
        })
    for note_id in sorted(set(before) & set(after)):
        old, new = before[note_id], after[note_id]
        if old == new:
            continue
        changed = [
            field for field in ("content", "context", "tags", "links")
            if getattr(old, field) != getattr(new, field)
        ]
        events.append({
            "id": note_id,
            "event": "UPDATE",
            "memory": new.content,
            "tags": list(new.tags),
            "context": new.context,
            "links": list(new.links),
            "changed_fields": changed,
            "evolved_from": added_id,
            "timestamp": timestamp,
            "session_id": session_id,
        })
    return events


def normalize_search_results(raw: Any) -> list[dict[str, Any]]:
    """A-Mem's search_agentic returns a list of dicts; tolerate wrapped shapes."""
    if isinstance(raw, dict):
        candidates = raw.get("results") or raw.get("memories") or []
    elif isinstance(raw, list):
        candidates = raw
    else:
        candidates = []
    results = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        results.append({
            "memory": item.get("content") or item.get("memory") or "",
            "id": item.get("id"),
            "score": round(float(item.get("score", 0) or 0), 2),
            "timestamp": item.get("timestamp"),
            "tags": item.get("tags") or [],
            "context": item.get("context"),
        })
    return results


class AMemBackend:
    """One AgenticMemorySystem per subject, each with its own persist directory."""

    name = "amem"

    def __init__(
        self,
        *,
        persist_root: Path,
        model_name: str = "all-MiniLM-L6-v2",
        llm_backend: str = "openai",
        llm_model: str = "gpt-4o-mini",
        api_key: str = "",
        base_url: str = "",
        system_factory: Callable[..., Any] | None = None,
    ):
        self.persist_root = Path(persist_root)
        self.model_name = model_name
        self.llm_backend = llm_backend
        self.llm_model = llm_model
        self.api_key = api_key
        self.base_url = base_url
        self._system_factory = system_factory
        self._systems: dict[str, Any] = {}

    def _safe(self, subject: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(subject)).strip("-") or "unnamed"

    def persist_dir_for(self, subject: str) -> Path:
        return self.persist_root / self._safe(subject)

    def _build_system(self, subject: str) -> Any:
        persist_dir = self.persist_dir_for(subject)
        persist_dir.mkdir(parents=True, exist_ok=True)
        if self._system_factory is not None:
            return self._system_factory(
                subject=subject,
                persist_dir=persist_dir,
                model_name=self.model_name,
                llm_backend=self.llm_backend,
                llm_model=self.llm_model,
            )
        try:
            from agentic_memory.memory_system import AgenticMemorySystem
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            raise RuntimeError(
                "amem backend requires the agentic_memory package "
                "(pip install git+https://github.com/agiresearch/A-mem.git)"
            ) from exc
        with self._openai_base_url_override():
            return AgenticMemorySystem(
                model_name=self.model_name,
                llm_backend=self.llm_backend,
                llm_model=self.llm_model,
                api_key=self.api_key or None,
            )

    @contextmanager
    def _openai_base_url_override(self):
        """A-Mem's OpenAIController takes an api_key but has no base_url
        parameter; it builds ``OpenAI(api_key=...)`` and that client reads
        OPENAI_BASE_URL from the environment when base_url isn't passed. This
        is the only hook available to point amem at a custom gateway without
        forking the upstream controller, so we set the env var for the
        duration of system construction (the client bakes it in) and restore
        whatever was there before.
        """
        if not self.base_url:
            yield
            return
        previous = os.environ.get("OPENAI_BASE_URL")
        os.environ["OPENAI_BASE_URL"] = self.base_url
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("OPENAI_BASE_URL", None)
            else:
                os.environ["OPENAI_BASE_URL"] = previous

    def system_for(self, subject: str) -> Any:
        system = self._systems.get(subject)
        if system is None:
            system = self._build_system(subject)
            self._systems[subject] = system
        return system

    def reset(self, subject: str) -> None:
        # Dropping the instance is the reset: A-Mem has no delete_all, and a
        # fresh system rebuilds an empty note store for this subject.
        self._systems.pop(subject, None)
        self.system_for(subject)

    def add_session(
        self,
        subject: str,
        messages: list[dict[str, Any]],
        metadata: dict[str, Any],
        *,
        collector: TraceCollector | None = None,
        turn_id: str | None = None,
        parent_event_id: str | None = None,
    ) -> list[MemoryEvent]:
        session_id = str(metadata.get("session_id", ""))
        timestamp = str(metadata.get("timestamp", ""))
        amem_timestamp = to_amem_timestamp(timestamp)
        event_ref = collector.start_event(
            "memory_update", "add", turn_id=turn_id, parent_event_id=parent_event_id,
            input={"subject_id": subject, "messages": messages, "metadata": metadata},
        ) if collector is not None else None
        raw_events: list[dict[str, Any]] = []
        note_ids: list[str] = []
        try:
            system = self.system_for(subject)
            for message in messages:
                content = str(message.get("content", "")).strip()
                if not content:
                    continue
                speaker = message.get("speaker") or message.get("role") or subject
                before = snapshot_notes(system)
                kwargs: dict[str, Any] = {}
                if amem_timestamp:
                    kwargs["timestamp"] = amem_timestamp
                note_id = system.add_note(f"Speaker {speaker} says: {content}", **kwargs)
                after = snapshot_notes(system)
                note_ids.append(str(note_id))
                raw_events.extend(memory_events_from_diff(
                    before, after, added_id=str(note_id),
                    session_id=session_id, timestamp=timestamp,
                ))
        except Exception as exc:
            if collector is not None and event_ref is not None:
                collector.fail_event(event_ref, exc)
            raise
        normalized = [
            MemoryEvent(
                event=str(item.get("event", "ADD")), memory=item.get("memory", ""),
                memory_id=item.get("id"), session_id=session_id,
                timestamp=timestamp, metadata=dict(item),
            )
            for item in raw_events
        ]
        if collector is not None and event_ref is not None:
            collector.finish_event(
                event_ref,
                output={"events": [
                    {"memory_id": item.memory_id, "event": item.event, "memory": item.memory,
                     "session_id": item.session_id, "timestamp": item.timestamp}
                    for item in normalized
                ]},
                raw={"response": {"results": raw_events}, "note_ids": note_ids},
                metrics={"notes_added": len(note_ids), "events_observed": len(raw_events)},
            )
        return normalized

    def search(
        self,
        subject: str,
        query: str,
        top_k: int,
        *,
        collector: TraceCollector | None = None,
        turn_id: str | None = None,
        parent_event_id: str | None = None,
    ) -> list[RetrievedMemory]:
        event_ref = collector.start_event(
            "memory_retrieval", "search", turn_id=turn_id, parent_event_id=parent_event_id,
            input={"subject_id": subject, "query": query, "top_k": top_k},
        ) if collector is not None else None
        try:
            raw = self.system_for(subject).search_agentic(query, k=top_k)
            items = normalize_search_results(raw)
        except Exception as exc:
            if collector is not None and event_ref is not None:
                collector.fail_event(event_ref, exc)
            raise
        normalized = [
            RetrievedMemory(
                memory=item.get("memory", ""), score=round(item.get("score", 0) or 0, 2),
                memory_id=item.get("id"), session_id=None,
                timestamp=item.get("timestamp"), metadata=dict(item),
            )
            for item in items
        ]
        if collector is not None and event_ref is not None:
            collector.finish_event(
                event_ref,
                output={"candidates": [
                    {"memory_id": item.memory_id, "memory": item.memory, "score": item.score,
                     "rank": rank, "selected": False, "session_id": item.session_id,
                     "timestamp": item.timestamp}
                    for rank, item in enumerate(normalized, start=1)
                ]},
                raw={"response": raw if isinstance(raw, dict) else {"results": raw}},
            )
        return normalized
