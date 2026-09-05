"""Trace answer generation independently of any particular model SDK."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from memeval.generation.context import AssembledContext
from memeval.trace.collector import TraceCollector


@dataclass(frozen=True)
class GenerationResponse:
    text: str
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    event_id: str | None = None


class GenerationBackend(Protocol):
    name: str

    def complete(
        self,
        messages: list[dict[str, str]],
        model: str,
        parameters: dict[str, Any],
    ) -> GenerationResponse: ...


class TracedGenerator:
    def __init__(self, backend: GenerationBackend):
        self.backend = backend

    def generate(
        self,
        question: str,
        context: AssembledContext,
        collector: TraceCollector,
        *,
        model: str,
        system_prompt: str = "Answer the question using the supplied memory context.",
        parameters: dict[str, Any] | None = None,
        turn_id: str | None = None,
    ) -> GenerationResponse:
        parameters = dict(parameters or {})
        user_content = f"Memory context:\n{context.text}\n\nQuestion:\n{question}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        event_id = collector.start_event(
            "generation",
            "generate",
            parent_event_id=context.event_id,
            turn_id=turn_id,
            input={
                "question": question,
                "context_event_id": context.event_id,
                "messages": messages,
                "model": model,
                "parameters": parameters,
                "provider": self.backend.name,
            },
        )
        started = time.perf_counter()
        try:
            response = self.backend.complete(messages, model, parameters)
        except Exception as exc:
            collector.fail_event(event_id, exc)
            raise
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        collector.finish_event(
            event_id,
            output={"response": response.text, "finish_reason": response.finish_reason},
            raw=response.raw,
            metrics={"provider": self.backend.name, "model": model, "latency_ms": latency_ms, **response.usage},
        )
        collector.envelope.qa["response"] = response.text
        return replace(response, event_id=event_id)
