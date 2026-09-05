from memeval.memory.amem import AMemBackend, to_amem_timestamp
from memeval.schema import validate_trace_v2
from memeval.trace import TraceCollector, TraceEnvelope


class FakeNote:
    def __init__(self, note_id, content, tags=None, context="", links=None):
        self.id = note_id
        self.content = content
        self.tags = list(tags or [])
        self.context = context
        self.links = list(links or [])


class FakeAgenticMemorySystem:
    """Mimics A-Mem: add_note returns only an id, and may mutate neighbours."""

    def __init__(self, *, evolve=False):
        self.memories = {}
        self._counter = 0
        self._evolve = evolve

    def add_note(self, content, **kwargs):
        self._counter += 1
        note_id = f"n{self._counter}"
        # Evolution rewrites an existing neighbour in place, as A-Mem does.
        if self._evolve and self.memories:
            first = next(iter(self.memories.values()))
            first.tags = [*first.tags, "evolved"]
            first.context = "rewritten by evolution"
        self.memories[note_id] = FakeNote(note_id, content, tags=["t"], context="ctx")
        return note_id

    def search_agentic(self, query, k=5):
        return [
            {"id": note.id, "content": note.content, "tags": note.tags, "score": 0.876, "context": note.context}
            for note in list(self.memories.values())[:k]
        ]


def build_backend(tmp_path, *, evolve=False):
    return AMemBackend(
        persist_root=tmp_path / "persist",
        system_factory=lambda **kwargs: FakeAgenticMemorySystem(evolve=evolve),
    )


def test_amem_emits_update_and_retrieval_events(tmp_path):
    backend = build_backend(tmp_path)
    collector = TraceCollector(TraceEnvelope("t-amem", "sample-1", "Caroline", "amem"))

    backend.reset("Caroline")
    events = backend.add_session(
        "Caroline",
        [{"role": "user", "content": "I went to a support group", "speaker": "Caroline"}],
        {"session_id": "s1", "timestamp": "1:56 pm on 8 May, 2023"},
        collector=collector,
    )
    retrieved = backend.search("Caroline", "support group?", 3, collector=collector)

    assert events[0].event == "ADD"
    assert "support group" in events[0].memory
    assert retrieved[0].score == 0.88
    validate_trace_v2(collector.to_dict())
    assert [e["stage"] for e in collector.to_dict()["events"]] == ["memory_update", "memory_retrieval"]


def test_amem_diff_recovers_evolution_as_update_event(tmp_path):
    """A-Mem's add_note returns only an id; neighbour rewrites must come from the diff."""
    backend = build_backend(tmp_path, evolve=True)
    backend.reset("Caroline")
    backend.add_session(
        "Caroline", [{"content": "first", "speaker": "Caroline"}],
        {"session_id": "s1", "timestamp": ""},
    )
    events = backend.add_session(
        "Caroline", [{"content": "second", "speaker": "Caroline"}],
        {"session_id": "s2", "timestamp": ""},
    )

    kinds = [event.event for event in events]
    assert "ADD" in kinds
    assert "UPDATE" in kinds, "neighbour evolution should surface as an UPDATE event"
    update = next(e for e in events if e.event == "UPDATE")
    assert "tags" in update.metadata["changed_fields"]


def test_amem_subjects_are_isolated(tmp_path):
    backend = build_backend(tmp_path)
    backend.reset("Caroline")
    backend.reset("Melanie")
    backend.add_session("Caroline", [{"content": "caroline fact"}], {"session_id": "s1", "timestamp": ""})

    assert backend.search("Melanie", "anything", 5) == []
    assert backend.persist_dir_for("Caroline") != backend.persist_dir_for("Melanie")


def test_locomo_timestamp_converts_to_amem_format():
    assert to_amem_timestamp("1:56 pm on 8 May, 2023") == "202305081356"
    assert to_amem_timestamp("12:05 am on 1 January, 2024") == "202401010005"
    assert to_amem_timestamp("12:30 pm on 3 Dec, 2022") == "202212031230"
    assert to_amem_timestamp("") == ""
    assert to_amem_timestamp("not a date") == ""
