"""Provider- and storage-independent staged diagnosis traversal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from memeval.schema.diagnosis import (
    DiagnosisResult,
    DiagnosisStage,
    DiagnosisStatus,
    MemoryData,
    QAData,
    StageResult,
    UsageStats,
)


ConsistencyHandler = Callable[[QAData, UsageStats], StageResult]
MemoryHandler = Callable[[QAData, MemoryData, UsageStats], StageResult]


@dataclass(frozen=True)
class StageHandlers:
    """Injected stage implementations used by the traversal pipeline."""

    consistency: ConsistencyHandler
    extraction: MemoryHandler
    update: MemoryHandler
    retrieval: MemoryHandler
    reasoning: MemoryHandler


def _execution_error(stage_result: StageResult, usage: UsageStats) -> DiagnosisResult:
    return DiagnosisResult(
        label=None,
        reason=stage_result.reason,
        stage=DiagnosisStage.ERROR,
        status=DiagnosisStatus.ERROR,
        usage_stats=usage,
    )


def run_staged_diagnosis(
    qa_data: QAData,
    memory_data: MemoryData,
    handlers: StageHandlers,
    usage: UsageStats | None = None,
) -> DiagnosisResult:
    """Run stages in causal order and return at the first diagnosed failure."""
    stats = usage or UsageStats()

    try:
        consistency = handlers.consistency(qa_data, stats)
        if consistency.status == DiagnosisStatus.ERROR:
            return _execution_error(consistency, stats)
        if consistency.stage_passed:
            return DiagnosisResult(
                label=None,
                reason=consistency.reason,
                stage=DiagnosisStage.CONSISTENCY_CHECK,
                answer_correct=True,
                usage_stats=stats,
            )

        ordered_stages = (
            (DiagnosisStage.MEMORY_EXTRACTION, handlers.extraction),
            (DiagnosisStage.MEMORY_UPDATE, handlers.update),
            (DiagnosisStage.MEMORY_RETRIEVAL, handlers.retrieval),
        )
        for stage, handler in ordered_stages:
            stage_result = handler(qa_data, memory_data, stats)
            if stage_result.status == DiagnosisStatus.ERROR:
                return _execution_error(stage_result, stats)
            if not stage_result.stage_passed:
                return DiagnosisResult(
                    label=stage_result.label,
                    reason=stage_result.reason,
                    stage=stage,
                    usage_stats=stats,
                )

        reasoning = handlers.reasoning(qa_data, memory_data, stats)
        if reasoning.status == DiagnosisStatus.ERROR:
            return _execution_error(reasoning, stats)
        return DiagnosisResult(
            label=reasoning.label,
            reason=reasoning.reason,
            stage=DiagnosisStage.REASONING,
            usage_stats=stats,
        )
    except Exception as error:
        return DiagnosisResult(
            label=None,
            reason=f"Error during diagnosis: {error}",
            stage=DiagnosisStage.ERROR,
            status=DiagnosisStatus.ERROR,
            usage_stats=stats,
        )
