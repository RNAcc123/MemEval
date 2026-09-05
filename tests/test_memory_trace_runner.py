import json

from memeval.datasets.base import EvaluationSample
from memeval.generation import TracedGenerator
from memeval.generation.fake import FakeGenerationBackend
from memeval.memory.fake import FakeMemoryBackend
from memeval.runners.memory_trace import MemoryTraceRunner
from memeval.schema import validate_trace_v2
from memeval.storage import TraceStore


def sample():
    return EvaluationSample(
        sample_id="sample-1",
        sessions=[
            [{"role": "user", "content": "I study jazz piano."}],
            [{"role": "assistant", "content": "That sounds great."}],
        ],
        session_ids=["s1", "s2"],
        timestamps=["2024-01-01", "2024-01-02"],
        questions=[{"question_id": "q1", "question": "What do I study?", "answer": "jazz piano"}],
        metadata={"source": "fixture"},
    )


def test_runner_and_trace_store_support_resume(tmp_path):
    runner = MemoryTraceRunner(FakeMemoryBackend(), TracedGenerator(FakeGenerationBackend()), top_k=3, context_limit=1, model="fake")
    envelope = runner.run_sample(sample(), sample().questions[0])
    validate_trace_v2(envelope.to_dict())
    store = TraceStore(tmp_path)
    store.append(envelope)
    assert envelope.trace_id in store.completed_ids()
    assert len(store.traces_path.read_text().splitlines()) == 1
    assert len(store.legacy_path.read_text().splitlines()) == 1
    assert json.loads(store.traces_path.read_text())["record_id"] == envelope.trace_id

