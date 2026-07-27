import json

from memeval.analysis import compare_records
from memeval.storage import JsonlRunStore, load_records, stable_run_id


def test_jsonl_store_appends_and_resumes(tmp_path):
    store = JsonlRunStore(tmp_path / "run", run_id=stable_run_id("input", "model"), manifest={"model": "fake"})
    store.append_result({"record_id": "a", "status": "completed"})
    store.append_error({"record_id": "b", "message": "timeout"})
    assert store.completed_ids() == {"a"}
    assert load_records(store.results_path) == [{"record_id": "a", "status": "completed"}]
    assert json.loads(store.manifest_path.read_text(encoding="utf-8"))["model"] == "fake"


def test_metrics_reports_coverage_duplicates_and_invalid_records():
    metrics = compare_records(
        [{"record_id": "a"}, {"record_id": "b"}],
        [{"record_id": "a"}, {"record_id": "a"}, {"record_id": "c", "status": "error"}],
    )
    assert metrics["matched_records"] == 0
    assert metrics["missing_records"] == 1
    assert metrics["duplicate_records"] == 1
    assert metrics["invalid_records"] == 1
