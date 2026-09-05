"""Single-model staged diagnosis."""

from typing import Dict, List, Optional

from memeval.config import DiagnosisConfig
from memeval.schema import (
    DiagnosisResult,
    DiagnosisStage,
    MemoryData,
    QAData,
    SubjectMemoryData,
    UsageStats,
)

from memeval.diagnosis.pipeline import StageHandlers, run_staged_diagnosis
from memeval.diagnosis.stages import (
    stage0_consistency_check,
    stage1_memory_extraction,
    stage2_memory_update,
    stage3_memory_retrieval,
    stage4_reasoning,
)

__all__ = ["analyze_qa_pair", "analyze_qa_pair_legacy"]


def analyze_qa_pair(
    qa_data: QAData,
    memory_data: MemoryData,
    model: str = "deepseek",
    config: Optional[DiagnosisConfig] = None
) -> DiagnosisResult:
    """Main staged diagnosis function.

    Execute diagnosis stages in order until an issue is found or all stages pass.

    Args:
        qa_data: QAData instance
        memory_data: MemoryData instance
        model: model to use
        config: diagnosis configuration

    Returns:
        DiagnosisResult containing the full diagnosis outcome
    """
    print(f"\n{'='*70}")
    print(f"🔍 Start staged diagnosis")
    print(f"📝 Question: {qa_data.question}")
    print(f"{'='*70}\n")

    stats = UsageStats()
    handlers = StageHandlers(
        consistency=lambda qa, usage: stage0_consistency_check(
            qa, model, config, usage_stats=usage
        ),
        extraction=lambda qa, memory, usage: stage1_memory_extraction(
            qa, memory, model, config, usage_stats=usage
        ),
        update=lambda qa, memory, usage: stage2_memory_update(
            qa, memory, model, config, usage_stats=usage
        ),
        retrieval=lambda qa, memory, usage: stage3_memory_retrieval(
            qa, memory, model, config, usage_stats=usage
        ),
        reasoning=lambda qa, memory, usage: stage4_reasoning(
            qa, memory, model, config, usage_stats=usage
        ),
    )
    result = run_staged_diagnosis(qa_data, memory_data, handlers, stats)
    stats.print_summary()
    return result


def _build_memory_data(subjects: List[Dict]) -> MemoryData:
    """Convert plain subject dicts (subject_id/memories/retrieval) into MemoryData."""
    return MemoryData(
        subjects=[
            SubjectMemoryData(
                subject_id=subject.get("subject_id", ""),
                memories=subject.get("memories", []),
                retrieval=subject.get("retrieval", []),
            )
            for subject in subjects
        ]
    )


def analyze_qa_pair_legacy(
    qa_question: str,
    qa_answer: str,
    qa_response: str,
    subjects: List[Dict],
    model: str = "deepseek"
) -> Dict:
    """Legacy compatibility wrapper.

    Keeps compatibility with older code by converting arguments into the new
    dataclasses and calling the new analysis function.

    Args:
        qa_question: question text
        qa_answer: reference answer
        qa_response: model response
        subjects: list of {"subject_id", "memories", "retrieval"} dicts, one per
            party whose memory contributed to this QA record
        model: model to use

    Returns:
        Diagnosis result dict (legacy format)
    """
    qa_data = QAData(
        question=qa_question,
        answer=qa_answer,
        response=qa_response
    )

    memory_data = _build_memory_data(subjects)

    result = analyze_qa_pair(qa_data, memory_data, model)

    return {
        "label": result.label,
        "reason": result.reason,
        "stage": result.stage.value if isinstance(result.stage, DiagnosisStage) else result.stage,
        "status": result.status.value,
        "answer_correct": result.answer_correct,
    }
