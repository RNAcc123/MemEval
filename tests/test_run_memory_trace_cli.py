import json
import subprocess
import sys
from pathlib import Path

from memeval.schema import validate_trace_dataset, validate_trace_v2
from memeval.storage import load_records


ROOT = Path(__file__).resolve().parents[1]


def test_cli_exports_legacy_json_and_resumes_without_duplicates(tmp_path):
    dataset = tmp_path / "longmemeval.json"
    dataset.write_text(json.dumps([{
        "question_id": "q1",
        "question": "What do I study?",
        "answer": "jazz piano",
        "question_type": "single-session-user",
        "haystack_dates": ["2024-01-01"],
        "haystack_session_ids": ["s1"],
        "haystack_sessions": [[{"role": "user", "content": "I study jazz piano."}]],
    }]), encoding="utf-8")
    output = tmp_path / "run"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_memory_trace.py"),
        "--dataset", str(dataset),
        "--dataset-type", "longmemeval",
        "--backend", "fake",
        "--generation-backend", "fake",
        "--output-dir", str(output),
    ]

    first = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    assert first.returncode == 0, first.stderr
    traces = load_records(output / "traces.jsonl")
    assert len(traces) == 1
    validate_trace_v2(traces[0])
    legacy = json.loads((output / "legacy_trace.json").read_text(encoding="utf-8"))
    validate_trace_dataset(legacy)
    assert legacy["q1"][0]["qa_response"].startswith("[fake]")

    second = subprocess.run(command + ["--resume"], cwd=ROOT, capture_output=True, text=True, check=False)
    assert second.returncode == 0, second.stderr
    assert len(load_records(output / "traces.jsonl")) == 1
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["completed"] == 1
    assert summary["added"] == 0
    assert summary["skipped"] == 1
    assert summary["failed"] == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["trace_schema_version"] == "2.0"
    assert "api_key" not in manifest
