from memeval.memory.memoryos import (
    MemoryOSBackend,
    normalize_retrieved,
    page_to_text,
    pair_messages,
)
from memeval.schema import validate_trace_v2
from memeval.trace import TraceCollector, TraceEnvelope


class FakeRetriever:
    def __init__(self, system):
        self.system = system

    def retrieve_context(self, user_query, user_id):
        return {
            "retrieved_pages": [
                {"page_id": "p1", "user_input": "I went to a support group",
                 "agent_response": "How was it?", "timestamp": "2023-05-08", "meta_info": "chain"},
            ],
            "retrieved_user_knowledge": [
                {"knowledge": "Caroline attends LGBTQ support groups", "timestamp": "2023-05-08"},
            ],
            "retrieved_assistant_knowledge": [
                {"knowledge": "Assistant offers empathetic replies", "timestamp": "2023-05-08"},
            ],
            "retrieved_at": "2023-05-08",
        }


class FakeMemoryos:
    """Mimics MemoryOS: add_memory returns nothing, retrieval is via .retriever."""

    def __init__(self):
        self.turns = []
        self.retriever = FakeRetriever(self)

    def add_memory(self, user_input, agent_response, timestamp=None, meta_data=None):
        self.turns.append((user_input, agent_response, timestamp, meta_data))
        return None

    def get_memory_stats(self):
        return {"turns": len(self.turns)}


def build_backend(tmp_path, factory=None):
    return MemoryOSBackend(
        storage_root=tmp_path / "storage",
        system_factory=factory or (lambda **kwargs: FakeMemoryos()),
    )


def test_memoryos_emits_update_and_retrieval_events(tmp_path):
    backend = build_backend(tmp_path)
    collector = TraceCollector(TraceEnvelope("t-mos", "sample-1", "Caroline", "memoryos"))

    backend.reset("Caroline")
    events = backend.add_session(
        "Caroline",
        [
            {"content": "I went to a support group", "speaker": "Caroline"},
            {"content": "How was it?", "speaker": "Melanie"},
        ],
        {"session_id": "s1", "timestamp": "2023-05-08"},
        collector=collector,
    )
    retrieved = backend.search("Caroline", "support group?", 10, collector=collector)

    assert len(events) == 1, "two messages form one dialogue turn"
    assert "support group" in events[0].memory
    assert retrieved
    validate_trace_v2(collector.to_dict())
    assert [e["stage"] for e in collector.to_dict()["events"]] == ["memory_update", "memory_retrieval"]


def test_memoryos_scores_are_absent_and_flagged(tmp_path):
    """MemoryOS discards scores; Stage 3 must be able to tell that apart from low relevance."""
    backend = build_backend(tmp_path)
    collector = TraceCollector(TraceEnvelope("t", "s", "Caroline", "memoryos"))
    backend.reset("Caroline")
    retrieved = backend.search("Caroline", "q", 10, collector=collector)

    assert all(item.score == 0.0 for item in retrieved)
    metrics = collector.to_dict()["events"][-1]["metrics"]
    assert metrics["score_available"] is False
    assert metrics["retrieval_is_internal"] is True


def test_memoryos_retrieval_merges_three_channels_with_type_tags(tmp_path):
    backend = build_backend(tmp_path)
    backend.reset("Caroline")
    retrieved = backend.search("Caroline", "q", 10)

    types = [item.metadata["memory_type"] for item in retrieved]
    assert types == ["page", "user_knowledge", "assistant_knowledge"]
    assert "User: I went to a support group" in retrieved[0].memory


def test_memoryos_search_respects_top_k(tmp_path):
    backend = build_backend(tmp_path)
    backend.reset("Caroline")
    assert len(backend.search("Caroline", "q", 2)) == 2


def test_memoryos_subjects_use_separate_storage(tmp_path):
    backend = build_backend(tmp_path)
    assert backend.storage_dir_for("Caroline") != backend.storage_dir_for("Melanie")


def test_pair_messages_pairs_and_keeps_trailing_message():
    turns = pair_messages([
        {"content": "a", "speaker": "Caroline"},
        {"content": "b", "speaker": "Melanie"},
        {"content": "c", "speaker": "Caroline"},
    ])
    assert [(t[0], t[1]) for t in turns] == [("a", "b"), ("c", "")]
    assert turns[0][2]["user_speaker"] == "Caroline"
    assert turns[0][2]["assistant_speaker"] == "Melanie"


def test_page_to_text_and_normalize_skip_empty():
    assert page_to_text({"user_input": "", "agent_response": ""}) == ""
    assert normalize_retrieved({"retrieved_pages": [{"user_input": "", "agent_response": ""}]}) == []
    assert normalize_retrieved({}) == []
