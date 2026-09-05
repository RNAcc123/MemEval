"""Build the legacy MemEval QA trace view from a v2 event envelope."""

from __future__ import annotations

from typing import Any

from memeval.trace.events import TraceEnvelope


def materialize_legacy_trace(envelope: TraceEnvelope | dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    data = envelope.to_dict() if isinstance(envelope, TraceEnvelope) else envelope
    qa = dict(data.get("qa", {}))
    memories: list[dict[str, Any]] = []
    retrieval: list[dict[str, Any]] = []
    for event in data.get("events", []):
        if event.get("stage") == "memory_update" and event.get("status") == "completed":
            output = event.get("output") or {}
            mutations = output.get("events") or [output]
            for mutation in mutations:
                memory_id = mutation.get("memory_id")
                content = mutation.get("content_after", mutation.get("memory", ""))
                if not content and not memory_id:
                    continue
                memories.append({
                    "id": memory_id,
                    "event": str(mutation.get("event", event.get("operation", "add"))).upper(),
                    "memory": content,
                    "event_id": event.get("event_id"),
                    "raw": event.get("raw", {}),
                })
        if event.get("stage") == "memory_retrieval" and event.get("status") == "completed":
            retrieval.extend((event.get("output") or {}).get("candidates", []))
    key = data.get("conversation_id") or data.get("sample_id") or data.get("trace_id")
    record = {
        "qa_question": qa.get("question", ""),
        "qa_answer": qa.get("answer", ""),
        "qa_response": qa.get("response", ""),
        "qa_category": qa.get("category", ""),
        "subjects": [{
            "subject_id": data.get("subject_id", ""),
            "memories": memories,
            "retrieval": retrieval,
        }],
        "trace_id": data.get("trace_id"),
    }
    return {str(key): [record]}
