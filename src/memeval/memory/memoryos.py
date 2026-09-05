"""MemoryOS backend.

MemoryOS differs from the other backends in three ways that shape this adapter:

* There is no public raw-retrieval method. ``Memoryos`` exposes only
  ``add_memory`` and ``get_response``; the latter returns generated text, not
  records. We therefore call the internal ``retriever.retrieve_context``, which
  is a deliberate coupling to MemoryOS internals (see ``RETRIEVAL_IS_INTERNAL``).
* Retrieval scores are computed internally but discarded before
  ``retrieve_context`` returns, so retrieved memories carry no score. We emit
  0.0 and flag ``score_available: False`` in metrics so Stage 3 analysis can
  distinguish "unscored" from "scored low".
* ``add_memory`` returns nothing, so update events are recovered by diffing
  memory state around the call, as with the A-Mem and OpenCLAW backends.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any, Callable

from memeval.memory.base import MemoryEvent, RetrievedMemory
from memeval.trace.collector import TraceCollector


# Flags the coupling above for anyone auditing cross-backend comparability.
RETRIEVAL_IS_INTERNAL = True
SCORE_AVAILABLE = False

MEMORY_TYPE_PAGE = "page"
MEMORY_TYPE_USER_KNOWLEDGE = "user_knowledge"
MEMORY_TYPE_ASSISTANT_KNOWLEDGE = "assistant_knowledge"


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-") or "unnamed"


def _patch_keyword_extraction_default(model: str) -> None:
    """Retarget memoryos's hardcoded keyword-extraction model default.

    ``updater.py``/``mid_term.py`` call ``llm_extract_keywords(text, client=...)``
    without a ``model=`` kwarg, so it falls back to the package's hardcoded
    default of "gpt-4o-mini". Gateways that don't allow that model then 401 on
    every keyword-extraction call. All three modules import the same function
    object, so overwriting its ``__defaults__`` fixes every call site at once.
    """
    from memoryos.utils import llm_extract_keywords

    if llm_extract_keywords.__defaults__ != (model,):
        llm_extract_keywords.__defaults__ = (model,)


def page_to_text(page: dict[str, Any]) -> str:
    """Flatten a mid-term conversation page into one memory string."""
    user_input = str(page.get("user_input", "") or "").strip()
    agent_response = str(page.get("agent_response", "") or "").strip()
    parts = []
    if user_input:
        parts.append(f"User: {user_input}")
    if agent_response:
        parts.append(f"Assistant: {agent_response}")
    return "\n".join(parts)


def normalize_retrieved(context: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the three retrieval channels into one comparable candidate list.

    Order is preserved within each channel (MemoryOS already sorted them) and
    channels are concatenated pages-first, matching how get_response prioritises
    conversational history over distilled knowledge.
    """
    items: list[dict[str, Any]] = []
    for page in context.get("retrieved_pages") or []:
        if not isinstance(page, dict):
            continue
        items.append({
            "memory": page_to_text(page),
            "id": page.get("page_id"),
            "timestamp": page.get("timestamp"),
            "memory_type": MEMORY_TYPE_PAGE,
            "meta_info": page.get("meta_info"),
        })
    for key, memory_type in (
        ("retrieved_user_knowledge", MEMORY_TYPE_USER_KNOWLEDGE),
        ("retrieved_assistant_knowledge", MEMORY_TYPE_ASSISTANT_KNOWLEDGE),
    ):
        for entry in context.get(key) or []:
            if not isinstance(entry, dict):
                continue
            items.append({
                "memory": str(entry.get("knowledge", "") or ""),
                "id": None,
                "timestamp": entry.get("timestamp"),
                "memory_type": memory_type,
                "meta_info": None,
            })
    return [item for item in items if item["memory"].strip()]


def pair_messages(messages: list[dict[str, Any]]) -> list[tuple[str, str, dict[str, Any]]]:
    """Pair consecutive messages into (user_input, agent_response) turns.

    MemoryOS stores dialogue turns, not documents. LoCoMo is speaker-to-speaker,
    so the first speaker of each pair is mapped to the user slot and the reply to
    the assistant slot; the original speakers are kept for the trace.
    """
    turns: list[tuple[str, str, dict[str, Any]]] = []
    pending: dict[str, Any] | None = None
    for message in messages:
        content = str(message.get("content", "") or "").strip()
        if not content:
            continue
        if pending is None:
            pending = message
            continue
        turns.append((
            str(pending.get("content", "")).strip(),
            content,
            {
                "user_speaker": pending.get("speaker") or pending.get("role"),
                "assistant_speaker": message.get("speaker") or message.get("role"),
            },
        ))
        pending = None
    if pending is not None:
        # Trailing unpaired message still needs to reach memory.
        turns.append((
            str(pending.get("content", "")).strip(),
            "",
            {"user_speaker": pending.get("speaker") or pending.get("role"), "assistant_speaker": None},
        ))
    return turns


def snapshot_state(system: Any) -> dict[str, Any]:
    """Capture counts across the memory tiers; used to infer what a write did."""
    try:
        stats = system.get_memory_stats()
    except Exception:
        stats = {}
    return dict(stats) if isinstance(stats, dict) else {}


class MemoryOSBackend:
    """One Memoryos instance per subject, each with its own storage directory."""

    name = "memoryos"

    def __init__(
        self,
        *,
        storage_root: Path,
        openai_api_key: str = "",
        openai_base_url: str | None = None,
        llm_model: str = "gpt-4o-mini",
        assistant_id: str = "memeval_assistant",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        short_term_capacity: int = 7,
        mid_term_heat_threshold: int = 5,
        retrieval_queue_capacity: int = 7,
        long_term_knowledge_capacity: int = 100,
        system_factory: Callable[..., Any] | None = None,
    ):
        self.storage_root = Path(storage_root)
        self.openai_api_key = openai_api_key
        self.openai_base_url = openai_base_url
        self.llm_model = llm_model
        self.assistant_id = assistant_id
        self.embedding_model_name = embedding_model_name
        self.short_term_capacity = short_term_capacity
        self.mid_term_heat_threshold = mid_term_heat_threshold
        self.retrieval_queue_capacity = retrieval_queue_capacity
        self.long_term_knowledge_capacity = long_term_knowledge_capacity
        self._system_factory = system_factory
        self._systems: dict[str, Any] = {}

    def storage_dir_for(self, subject: str) -> Path:
        return self.storage_root / _safe(subject)

    def _build_system(self, subject: str) -> Any:
        storage_dir = self.storage_dir_for(subject)
        storage_dir.mkdir(parents=True, exist_ok=True)
        if self._system_factory is not None:
            return self._system_factory(subject=subject, storage_dir=storage_dir)
        try:
            from memoryos import Memoryos
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            raise RuntimeError(
                "memoryos backend requires the memoryos package "
                "(pip install memoryos-pro, or install from the MemoryOS repo's memoryos-pypi/ directory)"
            ) from exc
        _patch_keyword_extraction_default(self.llm_model)
        kwargs: dict[str, Any] = dict(
            user_id=subject,
            openai_api_key=self.openai_api_key,
            openai_base_url=self.openai_base_url,
            data_storage_path=str(storage_dir),
            llm_model=self.llm_model,
            assistant_id=self.assistant_id,
            short_term_capacity=self.short_term_capacity,
            mid_term_heat_threshold=self.mid_term_heat_threshold,
            retrieval_queue_capacity=self.retrieval_queue_capacity,
            long_term_knowledge_capacity=self.long_term_knowledge_capacity,
        )
        # embedding_model_name was added after the memoryos-pro PyPI 0.1.0
        # release; older installs raise TypeError on the unknown kwarg.
        if "embedding_model_name" in inspect.signature(Memoryos.__init__).parameters:
            kwargs["embedding_model_name"] = self.embedding_model_name
        return Memoryos(**kwargs)

    def system_for(self, subject: str) -> Any:
        system = self._systems.get(subject)
        if system is None:
            system = self._build_system(subject)
            self._systems[subject] = system
        return system

    def reset(self, subject: str) -> None:
        # MemoryOS has no clear API; the storage directory is the unit of reset.
        import shutil

        self._systems.pop(subject, None)
        storage_dir = self.storage_dir_for(subject)
        if storage_dir.exists():
            shutil.rmtree(storage_dir)
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
        event_ref = collector.start_event(
            "memory_update", "add", turn_id=turn_id, parent_event_id=parent_event_id,
            input={"subject_id": subject, "messages": messages, "metadata": metadata},
        ) if collector is not None else None
        raw_events: list[dict[str, Any]] = []
        try:
            system = self.system_for(subject)
            before_state = snapshot_state(system)
            for index, (user_input, agent_response, speakers) in enumerate(pair_messages(messages)):
                system.add_memory(
                    user_input=user_input,
                    agent_response=agent_response,
                    timestamp=timestamp or None,
                    meta_data={"session_id": session_id, **speakers},
                )
                raw_events.append({
                    "id": f"{session_id}:turn{index + 1}" if session_id else f"turn{index + 1}",
                    "event": "ADD",
                    "memory": page_to_text({"user_input": user_input, "agent_response": agent_response}),
                    "user_input": user_input,
                    "agent_response": agent_response,
                    "timestamp": timestamp,
                    "session_id": session_id,
                    **speakers,
                })
            after_state = snapshot_state(system)
        except Exception as exc:
            if collector is not None and event_ref is not None:
                collector.fail_event(event_ref, exc)
            raise
        normalized = [
            MemoryEvent(
                event=str(item["event"]), memory=item["memory"], memory_id=item["id"],
                session_id=session_id, timestamp=timestamp, metadata=dict(item),
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
                raw={"response": {"results": raw_events},
                     "memory_stats_before": before_state, "memory_stats_after": after_state},
                metrics={"turns_added": len(raw_events)},
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
            system = self.system_for(subject)
            retriever = getattr(system, "retriever", None)
            if retriever is None:
                raise RuntimeError("MemoryOS instance exposes no retriever; cannot retrieve raw memories")
            context = retriever.retrieve_context(user_query=query, user_id=subject)
            items = normalize_retrieved(context if isinstance(context, dict) else {})[:top_k]
        except Exception as exc:
            if collector is not None and event_ref is not None:
                collector.fail_event(event_ref, exc)
            raise
        normalized = [
            RetrievedMemory(
                # MemoryOS discards retrieval scores before returning; see module docstring.
                memory=item["memory"], score=0.0, memory_id=item.get("id"),
                session_id=None, timestamp=item.get("timestamp"), metadata=dict(item),
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
                raw={"response": context if isinstance(context, dict) else {"results": context}},
                metrics={
                    "score_available": SCORE_AVAILABLE,
                    "retrieval_is_internal": RETRIEVAL_IS_INTERNAL,
                    "candidates_by_type": {
                        memory_type: sum(1 for item in items if item.get("memory_type") == memory_type)
                        for memory_type in (
                            MEMORY_TYPE_PAGE, MEMORY_TYPE_USER_KNOWLEDGE, MEMORY_TYPE_ASSISTANT_KNOWLEDGE,
                        )
                    },
                },
            )
        return normalized
