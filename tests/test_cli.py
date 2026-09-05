import json

from typer.testing import CliRunner

from memeval.cli import app
from memeval.runners import BackendSettings, backend_manifest, build_backend

runner = CliRunner()


LOCOMO_SAMPLE = [{
    "sample_id": "s1",
    "conversation": {
        "speaker_a": "Caroline",
        "speaker_b": "Melanie",
        "session_1": [
            {"speaker": "Caroline", "text": "I went to a support group yesterday."},
            {"speaker": "Melanie", "text": "That sounds powerful."},
        ],
        "session_1_date_time": "1:56 pm on 8 May, 2023",
    },
    "qa": [{"question": "When did Caroline go?", "answer": "7 May 2023", "category": 2}],
}]


def write_dataset(tmp_path):
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps(LOCOMO_SAMPLE), encoding="utf-8")
    return path


def test_version_reports_schema_versions():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "trace schema" in result.stdout
    assert "diagnosis schema" in result.stdout


def test_backends_lists_every_backend():
    result = runner.invoke(app, ["backends"])
    assert result.exit_code == 0
    for name in ("fake", "mem0", "openclaw", "amem", "memoryos"):
        assert name in result.stdout


def test_trace_run_writes_records_and_manifest(tmp_path):
    dataset = write_dataset(tmp_path)
    out = tmp_path / "run"
    result = runner.invoke(app, [
        "trace", "run", "--dataset", str(dataset), "--dataset-type", "locomo",
        "--backend", "fake", "--output-dir", str(out),
    ])
    assert result.exit_code == 0, result.stdout
    assert "Completed traces: 1" in result.stdout
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["memory_backend"] == "fake"
    assert manifest["dataset_type"] == "locomo"
    assert manifest["trace_schema_version"]


def test_trace_run_resume_skips_completed(tmp_path):
    dataset = write_dataset(tmp_path)
    out = tmp_path / "run"
    args = [
        "trace", "run", "--dataset", str(dataset), "--dataset-type", "locomo",
        "--backend", "fake", "--output-dir", str(out),
    ]
    assert runner.invoke(app, args).exit_code == 0
    second = runner.invoke(app, [*args, "--resume"])
    assert second.exit_code == 0
    assert "Completed traces: 0" in second.stdout


def test_trace_run_dry_run_writes_nothing(tmp_path):
    dataset = write_dataset(tmp_path)
    out = tmp_path / "run"
    result = runner.invoke(app, [
        "trace", "run", "--dataset", str(dataset), "--dataset-type", "locomo",
        "--output-dir", str(out), "--dry-run",
    ])
    assert result.exit_code == 0
    assert "Dry run complete" in result.stdout
    assert not (out / "manifest.json").exists()


def test_trace_run_rejects_bad_dataset_type(tmp_path):
    dataset = write_dataset(tmp_path)
    result = runner.invoke(app, [
        "trace", "run", "--dataset", str(dataset), "--dataset-type", "nope",
        "--output-dir", str(tmp_path / "run"),
    ])
    assert result.exit_code != 0


def test_trace_run_rejects_inverted_range(tmp_path):
    dataset = write_dataset(tmp_path)
    result = runner.invoke(app, [
        "trace", "run", "--dataset", str(dataset), "--dataset-type", "locomo",
        "--output-dir", str(tmp_path / "run"), "--start", "5", "--end", "1",
    ])
    assert result.exit_code != 0


def test_validate_trace_accepts_generated_trace(tmp_path):
    dataset = write_dataset(tmp_path)
    out = tmp_path / "run"
    runner.invoke(app, [
        "trace", "run", "--dataset", str(dataset), "--dataset-type", "locomo",
        "--backend", "fake", "--output-dir", str(out),
    ])
    result = runner.invoke(app, ["validate-trace", str(out / "legacy_trace.json"), "--schema", "v1"])
    assert result.exit_code == 0
    assert "valid" in result.stdout


def test_validate_trace_rejects_malformed_file(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"not": "a trace"}', encoding="utf-8")
    result = runner.invoke(app, ["validate-trace", str(bad), "--schema", "v2"])
    assert result.exit_code == 1


def test_build_backend_returns_fake_without_optional_deps():
    backend = build_backend(BackendSettings(name="fake"))
    assert backend.name == "fake"


def test_build_backend_rejects_unknown_name():
    try:
        build_backend(BackendSettings(name="nope"))
    except ValueError as exc:
        assert "nope" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_backend_manifest_strips_secrets():
    settings = BackendSettings(
        name="memoryos",
        memoryos={"storage_root": "/x", "api_key": "SECRET", "llm_model": "m"},
    )
    manifest = backend_manifest(settings)
    assert "SECRET" not in json.dumps(manifest)
    assert manifest["memoryos"]["score_available"] is False


def test_backend_manifest_records_mem0_backbone():
    settings = BackendSettings(
        name="mem0",
        mem0={"mode": "local", "llm_model": "deepseek-chat", "embedding_model": "bge-m3"},
    )
    manifest = backend_manifest(settings)["mem0"]
    assert manifest["llm_model"] == "deepseek-chat"
    assert manifest["embedding_model"] == "bge-m3"
