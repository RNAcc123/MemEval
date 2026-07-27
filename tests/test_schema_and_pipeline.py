from __future__ import annotations

import pytest

from memeval.diagnosis import StageHandlers, run_staged_diagnosis
from memeval.schema import (
    DiagnosisStage,
    DiagnosisStatus,
    MemoryData,
    QAData,
    StageResult,
    TraceValidationError,
    UsageStats,
    validate_trace_dataset,
)
from memeval.schema.migration import migrate_diagnosis_record


def _passed(stage: DiagnosisStage) -> StageResult:
    return StageResult(True, None, f"{stage.value} passed", stage)


def test_validate_trace_dataset_accepts_legacy_shape_and_rejects_missing_fields():
    trace = {
        "conv-1": [
            {
                "qa_question": "q",
                "qa_answer": "a",
                "qa_response": "r",
                "person1": {"memories": []},
                "person2": {"memories": []},
            }
        ]
    }
    assert validate_trace_dataset(trace) == trace

    with pytest.raises(TraceValidationError, match="missing required fields"):
        validate_trace_dataset({"conv-1": [{"qa_question": "q"}]})


def test_migrate_diagnosis_record_is_non_destructive_and_adds_defaults():
    legacy = {"label": None, "stage": "0_consistency_check"}
    migrated = migrate_diagnosis_record(legacy)

    assert legacy == {"label": None, "stage": "0_consistency_check"}
    assert migrated["schema_version"] == "1.0"
    assert migrated["status"] == "completed"
    assert migrated["answer_correct"] is True


def test_pipeline_stops_at_first_failed_stage_and_shares_usage_object():
    calls: list[str] = []
    usage = UsageStats()

    def consistency(_qa, _usage):
        calls.append("0")
        return StageResult(False, "inconsistent", "answer differs", DiagnosisStage.CONSISTENCY_CHECK)

    def extraction(_qa, _memory, _usage):
        calls.append("1")
        return StageResult(False, "1.1", "missing", DiagnosisStage.MEMORY_EXTRACTION)

    def unexpected(*_args):
        raise AssertionError("later stage should not run")

    result = run_staged_diagnosis(
        QAData("q", "a", "r"),
        MemoryData(),
        StageHandlers(consistency, extraction, unexpected, unexpected, unexpected),
        usage,
    )

    assert calls == ["0", "1"]
    assert result.stage == DiagnosisStage.MEMORY_EXTRACTION
    assert result.status == DiagnosisStatus.COMPLETED
    assert result.usage_stats is usage
