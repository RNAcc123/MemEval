from pathlib import Path

import pytest

from memeval.datasets import LoCoMoAdapter, LongMemEvalAdapter
from memeval.memory import Mem0Backend
from memeval.runners import part_location, retry_call


def test_locomo_adapter_normalizes_sessions(tmp_path: Path):
    path = tmp_path / "locomo.json"
    path.write_text(
        '[{"sample_id":"s1","conversation":{"speaker_a":"A","speaker_b":"B","session_2":[],"session_1":[],"session_1_date_time":"d1"},"qa":[]}]',
        encoding="utf-8",
    )
    sample = LoCoMoAdapter().load(path)[0]
    assert sample.session_ids == ["session_1", "session_2"]
    assert sample.subjects == ["A", "B"]
    assert sample.timestamps == ["d1", ""]


def test_longmemeval_adapter_rejects_misaligned_haystack(tmp_path: Path):
    path = tmp_path / "long.json"
    path.write_text('[{"haystack_dates":[],"haystack_session_ids":["s1"],"haystack_sessions":[]}]', encoding="utf-8")
    with pytest.raises(ValueError, match="different lengths"):
        LongMemEvalAdapter().load(path)


def test_mem0_backend_normalizes_add_and_search():
    class Client:
        def delete_all(self, **kwargs):
            return None

        def add(self, messages, **kwargs):
            return {"results": [{"id": "m1", "event": "ADD", "memory": "fact"}]}

        def search(self, query, **kwargs):
            return {"results": [{"id": "m1", "memory": "fact", "score": 0.876, "metadata": {"session_id": "s1"}}]}

    backend = Mem0Backend(Client())
    backend.reset("u1")
    assert backend.add_session("u1", [], {"session_id": "s1"})[0].memory_id == "m1"
    assert backend.search("u1", "q", 3)[0].score == 0.88


def test_trace_runner_helpers():
    assert part_location(11, 10, "trace") == (2, "1", "trace_part2.json")
    assert retry_call(lambda: "ok") == "ok"
