from types import SimpleNamespace

import pytest

from memeval.generation import (
    GenerationResponse,
    OpenAIChatGenerationBackend,
    TracedGenerator,
    assemble_context,
)
from memeval.memory.base import RetrievedMemory
from memeval.memory.mem0 import Mem0Backend
from memeval.schema import validate_trace_dataset, validate_trace_v2
from memeval.trace import TraceCollector, TraceEnvelope, materialize_legacy_trace


class FakeMem0Client:
    def search(self, query, *, filters, top_k):
        return {"results": [
            {"id": "m1", "memory": "The user studies jazz piano.", "score": 0.91, "metadata": {}},
            {"id": "m2", "memory": "The user likes hiking.", "score": 0.62, "metadata": {}},
        ][:top_k]}


class FakeGenerationBackend:
    name = "fake-llm"

    def complete(self, messages, model, parameters):
        assert "[1] The user studies jazz piano." in messages[1]["content"]
        assert parameters == {"temperature": 0}
        return GenerationResponse(
            "The user studies jazz piano.",
            usage={"prompt_tokens": 20, "completion_tokens": 6, "total_tokens": 26},
            finish_reason="stop",
            raw={"request_id": "req-1"},
        )


class FailingGenerationBackend:
    name = "failing-llm"

    def complete(self, messages, model, parameters):
        raise RuntimeError("generation failed")


def _collector():
    return TraceCollector(TraceEnvelope(
        trace_id="trace-e2e",
        sample_id="sample-e2e",
        subject_id="user-e2e",
        framework="mem0",
        qa={"question": "What does the user study?", "answer": "jazz piano"},
    ))


def test_retrieval_context_and_generation_form_complete_causal_chain():
    collector = _collector()
    memories = Mem0Backend(FakeMem0Client()).search(
        "user-e2e", "What does the user study?", 2, collector=collector, turn_id="turn-1",
    )
    retrieval_event_id = collector.latest_event_id
    context = assemble_context(
        "What does the user study?", memories, collector, limit=1,
        parent_event_id=retrieval_event_id, turn_id="turn-1",
    )
    response = TracedGenerator(FakeGenerationBackend()).generate(
        "What does the user study?", context, collector, model="fake-model",
        parameters={"temperature": 0}, turn_id="turn-1",
    )

    assert response.text == "The user studies jazz piano."
    data = collector.to_dict()
    validate_trace_v2(data)
    retrieval, assembly, generation = data["events"]
    assert assembly["parent_event_id"] == retrieval["event_id"]
    assert generation["parent_event_id"] == assembly["event_id"]
    assert assembly["output"]["memory_ids"] == ["m1"]
    assert generation["metrics"]["total_tokens"] == 26
    assert generation["raw"]["request_id"] == "req-1"
    legacy = materialize_legacy_trace(data)
    validate_trace_dataset(legacy)
    assert legacy["sample-e2e"][0]["qa_response"] == response.text


def test_generation_failure_is_recorded_and_reraised():
    collector = _collector()
    context = assemble_context(
        "question", [RetrievedMemory("memory", memory_id="m1")], collector,
    )
    with pytest.raises(RuntimeError, match="generation failed"):
        TracedGenerator(FailingGenerationBackend()).generate(
            "question", context, collector, model="fake-model",
        )
    assert collector.envelope.events[-1].status == "error"
    assert collector.envelope.events[-1].error["message"] == "generation failed"


def test_openai_compatible_backend_normalizes_response():
    response = SimpleNamespace(
        id="req-42",
        created=123,
        model="served-model",
        choices=[SimpleNamespace(message=SimpleNamespace(content="answer"), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=7, completion_tokens=2, total_tokens=9),
    )

    class Completions:
        def create(self, **kwargs):
            assert kwargs["model"] == "requested-model"
            assert kwargs["temperature"] == 0
            return response

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    result = OpenAIChatGenerationBackend(client).complete(
        [{"role": "user", "content": "question"}], "requested-model", {"temperature": 0},
    )
    assert result.text == "answer"
    assert result.usage["total_tokens"] == 9
    assert result.raw == {"request_id": "req-42", "created": 123, "model": "served-model"}
