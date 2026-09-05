import pytest

from memeval.schema import TraceV2ValidationError, validate_trace_v2
from memeval.trace import TraceCollector, TraceEnvelope, materialize_legacy_trace


def _envelope():
    return TraceEnvelope(
        trace_id="trace-1",
        sample_id="sample-1",
        subject_id="user-1",
        framework="fake",
        qa={"question": "What?", "answer": "A", "response": "A", "category": "test"},
    )


def test_collector_builds_causal_event_chain_and_materializes_legacy_trace():
    collector = TraceCollector(_envelope())
    extraction = collector.start_event("memory_extraction", "extract", input={"message_ids": ["m1"]})
    collector.finish_event(extraction, output={"candidates": [{"candidate_id": "c1", "content": "A"}]})
    update = collector.start_event("memory_update", "add", parent_event_id=extraction)
    collector.finish_event(update, output={"memory_id": "mem-1", "content_after": "A"})
    retrieval = collector.start_event("memory_retrieval", "search", input={"query": "What?"})
    collector.finish_event(retrieval, output={"candidates": [{"memory_id": "mem-1", "rank": 1, "selected": True}]})
    context = collector.start_event("context_assembly", "assemble", parent_event_id=retrieval)
    collector.finish_event(context, output={"memory_ids": ["mem-1"]})
    generation = collector.start_event("generation", "generate", parent_event_id=context)
    collector.finish_event(generation, output={"response": "A"})

    data = collector.to_dict()
    validate_trace_v2(data)
    assert [event["sequence"] for event in data["events"]] == [1, 2, 3, 4, 5]
    legacy = materialize_legacy_trace(data)
    record = legacy["sample-1"][0]
    assert record["qa_response"] == "A"
    assert record["subjects"][0]["memories"][0]["id"] == "mem-1"
    assert record["subjects"][0]["retrieval"][0]["selected"] is True


def test_collector_records_errors_and_rejects_invalid_updates():
    collector = TraceCollector(_envelope())
    event_id = collector.start_event("generation", "generate")
    collector.fail_event(event_id, RuntimeError("provider timeout"))
    assert collector.event(event_id).status == "error"
    with pytest.raises(ValueError):
        collector.finish_event(event_id)


def test_v2_validation_rejects_unknown_parent():
    data = TraceCollector(_envelope()).to_dict()
    data["events"] = [{
        "event_id": "e1", "trace_id": "trace-1", "stage": "generation",
        "operation": "generate", "sequence": 1, "status": "completed",
        "parent_event_id": "missing",
    }]
    with pytest.raises(TraceV2ValidationError, match="Unknown parent"):
        validate_trace_v2(data)
