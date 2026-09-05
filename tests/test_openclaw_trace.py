import subprocess
from pathlib import Path

from memeval.memory.openclaw import OpenClawBackend
from memeval.runners.memory_trace import MemoryTraceRunner
from memeval.schema import validate_trace_v2
from memeval.trace import TraceCollector, TraceEnvelope


SEARCH_JSON = '{"results": [{"id": "m-1", "memory": "Caroline studies jazz piano", "score": 0.876, "timestamp": "2026-07-27"}]}'


class FakeCompleted:
    def __init__(self, stdout="", returncode=0, stderr=""):
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr


def fake_cli(monkeypatch, *, search_stdout=SEARCH_JSON):
    """Stub the openclaw CLI: agent turns write a memory file, search returns JSON."""
    def run(cmd, **kwargs):
        if "setup" in cmd:
            return FakeCompleted("ok")
        if "agent" in cmd:
            path = Path(kwargs["cwd"]) / "MEMORY.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("Caroline studies jazz piano\n", encoding="utf-8")
            return FakeCompleted("saved 1 memory")
        if "search" in cmd:
            return FakeCompleted(search_stdout)
        return FakeCompleted("")

    monkeypatch.setattr(subprocess, "run", run)


def build_backend(tmp_path):
    return OpenClawBackend(workspace_root=tmp_path / "ws", openclaw_bin="openclaw", agent="main")


def test_openclaw_adapter_emits_update_and_retrieval_events(tmp_path, monkeypatch):
    fake_cli(monkeypatch)
    backend = build_backend(tmp_path)
    envelope = TraceEnvelope("trace-oc", "sample-1", "Caroline", "openclaw")
    collector = TraceCollector(envelope)

    backend.reset("Caroline")
    events = backend.add_session(
        "Caroline",
        [{"role": "user", "content": "I started jazz piano", "speaker": "Caroline"}],
        {"session_id": "s1", "timestamp": "2026-07-27"},
        collector=collector,
    )
    retrieved = backend.search("Caroline", "What does she study?", 3, collector=collector)

    assert events[0].event == "ADD"
    assert "jazz piano" in events[0].memory
    assert retrieved[0].score == 0.88
    validate_trace_v2(collector.to_dict())
    assert [event["stage"] for event in collector.to_dict()["events"]] == ["memory_update", "memory_retrieval"]


def test_normalize_messages_preserves_speaker_for_memory_prompt():
    session = [{"speaker": "Caroline", "text": "I started jazz piano"}]
    normalized = MemoryTraceRunner.normalize_messages(session)
    assert normalized[0]["speaker"] == "Caroline"
    assert normalized[0]["role"] in {"user", "assistant"}
