"""Cross-backend contract tests for MemoryBackend implementations.

These assert properties Stage 3 diagnosis depends on being identical across
backends: the candidate/event field shapes, and failure handling. Backend
fakes and constructors are reused from each backend's own test module so
there is exactly one definition of "how do I stand up a fake mem0/openclaw/
amem/memoryos backend" in the whole suite.

Behaviour that is genuinely backend-specific (timestamp formats, 3-channel
merge, subprocess plumbing, subject isolation, ...) stays in each backend's
own test file rather than being forced in here.
"""

from __future__ import annotations

import subprocess

import pytest

from memeval.memory.amem import AMemBackend
from memeval.memory.fake import FakeMemoryBackend
from memeval.memory.mem0 import Mem0Backend
from memeval.trace import TraceCollector, TraceEnvelope

from tests.test_amem_trace import FakeAgenticMemorySystem
from tests.test_amem_trace import build_backend as build_amem
from tests.test_memoryos_trace import build_backend as build_memoryos
from tests.test_openclaw_trace import FakeCompleted
from tests.test_openclaw_trace import build_backend as build_openclaw
from tests.test_openclaw_trace import fake_cli

SUBJECT = "Caroline"

REFERENCE_CANDIDATE_KEYS = {
    "memory_id", "memory", "score", "rank", "selected", "session_id", "timestamp",
}


class RaisingAgenticMemorySystem(FakeAgenticMemorySystem):
    def search_agentic(self, query, k=5):
        raise RuntimeError("boom")


class NoRetrieverMemoryos:
    retriever = None

    def add_memory(self, **kwargs):
        return None


def _seeded_fake(tmp_path, monkeypatch):
    backend = FakeMemoryBackend()
    backend.reset(SUBJECT)
    backend.add_session(SUBJECT, [{"content": "fact"}], {"session_id": "s1", "timestamp": ""})
    return backend


def _seeded_mem0(tmp_path, monkeypatch):
    class FakeMem0Client:
        def search(self, query, *, filters, top_k):
            return {"results": [{"id": "m-1", "memory": "x", "score": 0.5,
                                  "metadata": {"session_id": "s1", "timestamp": ""}}]}

    return Mem0Backend(FakeMem0Client())


def _seeded_openclaw(tmp_path, monkeypatch):
    fake_cli(monkeypatch)
    backend = build_openclaw(tmp_path)
    backend.reset(SUBJECT)
    return backend


def _seeded_amem(tmp_path, monkeypatch):
    backend = build_amem(tmp_path)
    backend.reset(SUBJECT)
    backend.add_session(SUBJECT, [{"content": "fact"}], {"session_id": "s1", "timestamp": ""})
    return backend


def _seeded_memoryos(tmp_path, monkeypatch):
    backend = build_memoryos(tmp_path)
    backend.reset(SUBJECT)
    return backend


SEEDED_BUILDERS = {
    "fake": _seeded_fake,
    "mem0": _seeded_mem0,
    "openclaw": _seeded_openclaw,
    "amem": _seeded_amem,
    "memoryos": _seeded_memoryos,
}


@pytest.mark.parametrize("backend_id", sorted(SEEDED_BUILDERS))
def test_candidate_field_shape_matches_reference(backend_id, tmp_path, monkeypatch):
    """Stage 3 cross-backend comparison requires identical candidate fields."""
    backend = SEEDED_BUILDERS[backend_id](tmp_path, monkeypatch)
    collector = TraceCollector(TraceEnvelope("t", "s", SUBJECT, backend.name))
    backend.search(SUBJECT, "q", 3, collector=collector)

    candidates = collector.to_dict()["events"][-1]["output"]["candidates"]
    assert candidates, f"{backend_id} produced no candidates to compare shape against"
    assert set(candidates[0]) == REFERENCE_CANDIDATE_KEYS


def _failing_mem0(tmp_path, monkeypatch):
    class RaisingMem0Client:
        def search(self, query, *, filters, top_k):
            raise RuntimeError("boom")

    return Mem0Backend(RaisingMem0Client())


def _failing_openclaw(tmp_path, monkeypatch):
    def failing(cmd, **kwargs):
        if "setup" in cmd:
            return FakeCompleted("ok")
        return FakeCompleted("", returncode=1, stderr="boom")

    monkeypatch.setattr(subprocess, "run", failing)
    return build_openclaw(tmp_path)


def _failing_amem(tmp_path, monkeypatch):
    return AMemBackend(persist_root=tmp_path / "persist", system_factory=lambda **kwargs: RaisingAgenticMemorySystem())


def _failing_memoryos(tmp_path, monkeypatch):
    return build_memoryos(tmp_path, factory=lambda **kwargs: NoRetrieverMemoryos())


# FakeMemoryBackend has no failure path to inject: it never wraps its calls in
# try/except, so it cannot participate in this contract. That asymmetry is a
# known limitation of the fake, not something this test should paper over.
FAILING_BUILDERS = {
    "mem0": _failing_mem0,
    "openclaw": _failing_openclaw,
    "amem": _failing_amem,
    "memoryos": _failing_memoryos,
}


@pytest.mark.parametrize("backend_id", sorted(FAILING_BUILDERS))
def test_failure_marks_event_as_error(backend_id, tmp_path, monkeypatch):
    """Whatever raises, the collector must record it consistently as an error event."""
    backend = FAILING_BUILDERS[backend_id](tmp_path, monkeypatch)
    collector = TraceCollector(TraceEnvelope("t", "s", SUBJECT, backend.name))

    with pytest.raises(RuntimeError):
        backend.search(SUBJECT, "q", 3, collector=collector)

    event = collector.to_dict()["events"][-1]
    assert event["status"] == "error"
    assert event["error"]["type"] == "RuntimeError"
    assert event["error"]["message"]


# mem0, openclaw, and memoryos are excluded: mem0's reset delegates to the
# client's delete_all, which this contract can't observe without a stateful
# fake client; openclaw's reset only affects on-disk memory files that the
# stubbed CLI's search never actually reads back from; and FakeMemoryos's
# retriever is a static stub disconnected from add/reset state entirely.
RESETTABLE_BUILDERS = {
    "fake": _seeded_fake,
    "amem": _seeded_amem,
}


@pytest.mark.parametrize("backend_id", sorted(RESETTABLE_BUILDERS))
def test_reset_clears_prior_memories(backend_id, tmp_path, monkeypatch):
    backend = RESETTABLE_BUILDERS[backend_id](tmp_path, monkeypatch)
    assert backend.search(SUBJECT, "q", 5)

    backend.reset(SUBJECT)
    assert backend.search(SUBJECT, "q", 5) == []
