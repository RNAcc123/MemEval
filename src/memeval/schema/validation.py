"""Strict validation for structured judge responses."""

from __future__ import annotations

from typing import Any

from memeval.schema.diagnosis import DiagnosisStage


VALID_STAGE_LABELS = {
    DiagnosisStage.MEMORY_EXTRACTION: {"1.1", "1.2", "1.3"},
    DiagnosisStage.MEMORY_UPDATE: {"2.1", "2.2", "2.3"},
    DiagnosisStage.MEMORY_RETRIEVAL: {"3.1", "3.2"},
    DiagnosisStage.REASONING: {"4.1", "4.2", "4.3"},
}


class InvalidJudgeResponse(ValueError):
    """Raised when a judge returns syntactically valid but invalid data."""


def validate_judge_response(stage: DiagnosisStage, result: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize the structured response for one diagnosis stage."""
    if not isinstance(result, dict):
        raise InvalidJudgeResponse(f"Expected a JSON object for {stage.value}")

    reason = result.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise InvalidJudgeResponse(f"Missing non-empty reason for {stage.value}")

    if stage == DiagnosisStage.CONSISTENCY_CHECK:
        is_consistent = result.get("is_consistent")
        if type(is_consistent) is not bool:
            raise InvalidJudgeResponse("Stage 0 requires boolean is_consistent")
        return {"is_consistent": is_consistent, "reason": reason.strip()}

    allowed_labels = VALID_STAGE_LABELS[stage]
    if stage == DiagnosisStage.REASONING:
        label = result.get("label")
        if label not in allowed_labels:
            raise InvalidJudgeResponse(
                f"Invalid label for {stage.value}: {label!r}; expected one of {sorted(allowed_labels)}"
            )
        return {"label": label, "reason": reason.strip()}

    is_sufficient = result.get("is_sufficient")
    if type(is_sufficient) is not bool:
        raise InvalidJudgeResponse(f"{stage.value} requires boolean is_sufficient")
    label = result.get("label")
    if is_sufficient and label is not None:
        raise InvalidJudgeResponse(f"{stage.value} must use label=null when is_sufficient=true")
    if not is_sufficient and label not in allowed_labels:
        raise InvalidJudgeResponse(
            f"Invalid label for {stage.value}: {label!r}; expected one of {sorted(allowed_labels)}"
        )
    return {"is_sufficient": is_sufficient, "label": label, "reason": reason.strip()}
