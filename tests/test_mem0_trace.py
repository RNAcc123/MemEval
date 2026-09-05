from memeval.memory.mem0 import Mem0Backend
from memeval.schema import validate_trace_v2
from memeval.trace import TraceCollector, TraceEnvelope, materialize_legacy_trace


class FakeMem0Client:
    def add(self, messages, *, user_id, metadata):
        return {"results": [{
            "id": "mem-1", "event": "ADD", "memory": messages[0]["content"],
            "metadata": metadata,
        }]}

    def search(self, query, *, filters, top_k):
        return {"results": [{
            "id": "mem-1", "memory": "User studies jazz piano", "score": 0.876,
            "metadata": {"session_id": "s1", "timestamp": "2026-07-27"},
        }][:top_k]}


def test_mem0_adapter_emits_update_and_retrieval_events_without_changing_return_values():
    envelope = TraceEnvelope("trace-mem0", "sample-1", "user-1", "mem0")
    collector = TraceCollector(envelope)
    backend = Mem0Backend(FakeMem0Client())

    events = backend.add_session(
        "user-1", [{"role": "user", "content": "User studies jazz piano"}],
        {"session_id": "s1", "timestamp": "2026-07-27"}, collector=collector,
    )
    retrieved = backend.search("user-1", "What does the user study?", 3, collector=collector)

    assert events[0].memory_id == "mem-1"
    assert retrieved[0].score == 0.88
    validate_trace_v2(collector.to_dict())
    trace_events = collector.to_dict()["events"]
    assert [event["stage"] for event in trace_events] == ["memory_update", "memory_retrieval"]
    assert trace_events[0]["raw"]["response"]["results"][0]["id"] == "mem-1"
    legacy = materialize_legacy_trace({**collector.to_dict(), "qa": {"question": "q", "answer": "a", "response": "r"}})
    assert legacy["sample-1"][0]["subjects"][0]["memories"][0]["memory"] == "User studies jazz piano"

