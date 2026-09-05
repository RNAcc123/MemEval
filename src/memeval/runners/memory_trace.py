"""Dataset-neutral orchestration for memory ingestion, retrieval and generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from memeval.datasets.base import EvaluationSample
from memeval.generation import TracedGenerator, assemble_context
from memeval.memory.base import MemoryBackend
from memeval.trace import TraceCollector, TraceEnvelope


class TraceMemoryBackend(MemoryBackend, Protocol):
    def add_session(self, subject: str, messages: list[dict[str, str]], metadata: dict[str, Any], *, collector: TraceCollector, turn_id: str | None = None) -> list[Any]: ...

    def search(self, subject: str, query: str, top_k: int, *, collector: TraceCollector, turn_id: str | None = None) -> list[Any]: ...


@dataclass
class MemoryTraceRunner:
    backend: TraceMemoryBackend
    generator: TracedGenerator
    top_k: int = 10
    context_limit: int | None = None
    model: str = ""
    generation_parameters: dict[str, Any] | None = None

    @staticmethod
    def normalize_messages(session: list[dict[str, Any]]) -> list[dict[str, str]]:
        normalized = []
        for message in session:
            raw_role = message.get("role") or message.get("from") or message.get("speaker")
            content = message.get("content", message.get("text", ""))
            role = raw_role
            if role not in {"user", "assistant", "system"}:
                role = "user" if str(raw_role).lower() in {"human", "user"} else "assistant"
            if str(content).strip():
                item = {"role": role, "content": str(content)}
                # Backends that write memory in first person need the original
                # speaker name, which the role mapping above collapses.
                speaker = message.get("speaker") or raw_role
                if speaker and str(speaker) != role:
                    item["speaker"] = str(speaker)
                for key in ("blip_caption", "query"):
                    if message.get(key):
                        item[key] = str(message[key])
                normalized.append(item)
        return normalized

    def run_sample(self, sample: EvaluationSample, question: dict[str, Any], *, subject_id: str | None = None) -> TraceEnvelope:
        subject = subject_id or sample.sample_id
        question_text = str(question.get("question", question.get("query", "")))
        envelope = TraceEnvelope(
            trace_id=f"{sample.sample_id}:{question.get('question_id', 'question')}",
            sample_id=sample.sample_id,
            subject_id=subject,
            framework=self.backend.name,
            conversation_id=sample.sample_id,
            qa={
                "question": question_text,
                "answer": str(question.get("answer", question.get("ground_truth", ""))),
                "category": str(question.get("category", question.get("question_type", ""))),
            },
            metadata={"dataset": sample.metadata.get("source", "unknown")},
        )
        collector = TraceCollector(envelope)
        self.backend.reset(subject)
        for index, session in enumerate(sample.sessions):
            session_id = sample.session_ids[index] if index < len(sample.session_ids) else f"session_{index + 1}"
            timestamp = sample.timestamps[index] if index < len(sample.timestamps) else ""
            self.backend.add_session(
                subject,
                self.normalize_messages(session),
                {"session_id": session_id, "timestamp": timestamp, "sample_id": sample.sample_id},
                collector=collector,
                turn_id=session_id,
            )
        memories = self.backend.search(subject, question_text, self.top_k, collector=collector, turn_id="question")
        retrieval_event_id = collector.latest_event_id
        context = assemble_context(
            question_text,
            memories,
            collector,
            limit=self.context_limit,
            parent_event_id=retrieval_event_id,
            turn_id="question",
        )
        self.generator.generate(
            question_text,
            context,
            collector,
            model=self.model,
            parameters=self.generation_parameters,
            turn_id="question",
        )
        envelope.qa["question_id"] = question.get("question_id")
        return envelope
