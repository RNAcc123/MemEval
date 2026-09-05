"""Diagnosis domain types and serialization contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


DIAGNOSIS_SCHEMA_VERSION = "1.0"


class ModelType(str, Enum):
    """Supported judge model families."""

    QWEN = "qwen"
    DEEPSEEK = "deepseek"
    GPT_4_1 = "gpt-4.1"
    GPT_5 = "gpt-5"
    GEMINI = "gemini-2.5-pro"


class DiagnosisStage(str, Enum):
    """Stages in the causal memory diagnosis pipeline."""

    CONSISTENCY_CHECK = "0_consistency_check"
    MEMORY_EXTRACTION = "1_memory_extraction"
    MEMORY_UPDATE = "2_memory_update"
    MEMORY_RETRIEVAL = "3_memory_retrieval"
    REASONING = "4_reasoning"
    ERROR = "error"


class DiagnosisStatus(str, Enum):
    """Execution status for a stage or complete diagnosis."""

    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class QAData:
    """Question, reference answer, and evaluated response."""

    question: str
    answer: str
    response: str
    category: str = ""

    def to_json_str(self, field_name: str) -> str:
        value = getattr(self, field_name.replace("qa_", ""))
        return json.dumps(value, ensure_ascii=False)


@dataclass
class SubjectMemoryData:
    """Memory evolution and retrieval data for one subject in a QA record."""

    subject_id: str = ""
    memories: list[dict] = field(default_factory=list)
    retrieval: list[dict] = field(default_factory=list)


@dataclass
class MemoryData:
    """Normalized memory evolution and retrieval data for a QA record.

    ``subjects`` holds one entry per party whose memory contributed to the
    QA record: length 1 for single-party datasets (e.g. LongMemEval), length 2
    for dual-party datasets (e.g. LoCoMo's speaker_a/speaker_b), and length N
    for any dataset with more parties. Diagnosis stages iterate this list
    instead of assuming exactly two hardcoded subjects.
    """

    subjects: list[SubjectMemoryData] = field(default_factory=list)

    @classmethod
    def from_qa_item(cls, qa_item: dict) -> "MemoryData":
        """Build from either the current ``subjects`` shape or legacy on-disk data.

        Legacy files (``data/input/**``) use fixed ``person1``/``person2`` +
        ``speaker_1_memories``/``speaker_2_memories`` keys; new trace output
        uses a ``subjects`` list. Both are accepted so existing datasets keep
        working without a migration pass.
        """
        if "subjects" in qa_item:
            return cls(subjects=[
                SubjectMemoryData(
                    subject_id=subject.get("subject_id", subject.get("name", "")),
                    memories=subject.get("memories", []),
                    retrieval=subject.get("retrieval", []),
                )
                for subject in (qa_item.get("subjects") or [])
            ])
        subjects = []
        for person_key, retrieval_key in (
            ("person1", "speaker_1_memories"),
            ("person2", "speaker_2_memories"),
        ):
            if person_key not in qa_item and retrieval_key not in qa_item:
                continue
            person = qa_item.get(person_key) or {}
            subjects.append(SubjectMemoryData(
                subject_id=person.get("name", ""),
                memories=person.get("memories", []),
                retrieval=qa_item.get(retrieval_key, []),
            ))
        return cls(subjects=subjects)


@dataclass
class UsageStats:
    """Aggregated API calls, latency, and token usage."""

    total_calls: int = 0
    total_latency: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    call_details: list[dict] = field(default_factory=list)

    def record_call(
        self,
        latency: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        model: str = "",
        stage: str = "",
    ) -> None:
        self.total_calls += 1
        self.total_latency += latency
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens += total_tokens
        self.call_details.append(
            {
                "model": model,
                "stage": stage,
                "latency_seconds": round(latency, 3),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "total_latency_seconds": round(self.total_latency, 3),
            "avg_latency_seconds": round(self.total_latency / self.total_calls, 3) if self.total_calls else 0,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "call_details": self.call_details,
        }

    def merge(self, other: UsageStats) -> None:
        self.total_calls += other.total_calls
        self.total_latency += other.total_latency
        self.total_prompt_tokens += other.total_prompt_tokens
        self.total_completion_tokens += other.total_completion_tokens
        self.total_tokens += other.total_tokens
        self.call_details.extend(other.call_details)

    def print_summary(self) -> None:
        print("  API call statistics:")
        print(f"     Calls: {self.total_calls}")
        print(f"     Total latency: {round(self.total_latency, 3)}s")
        if self.total_calls:
            print(f"     Average latency: {round(self.total_latency / self.total_calls, 3)}s")
        print(f"     Prompt tokens: {self.total_prompt_tokens}")
        print(f"     Completion tokens: {self.total_completion_tokens}")
        print(f"     Total tokens: {self.total_tokens}")


@dataclass
class StageResult:
    """Validated result for one diagnosis stage."""

    stage_passed: bool
    label: str | None
    reason: str
    stage: DiagnosisStage | None = None
    status: DiagnosisStatus = DiagnosisStatus.COMPLETED


@dataclass
class DiagnosisResult:
    """Final result for one staged diagnosis."""

    label: str | None
    reason: str
    stage: DiagnosisStage
    status: DiagnosisStatus = DiagnosisStatus.COMPLETED
    answer_correct: bool = False
    used_model: str | None = None
    voting_details: dict | None = None
    usage_stats: UsageStats | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": DIAGNOSIS_SCHEMA_VERSION,
            "label": self.label,
            "reason": self.reason,
            "stage": self.stage.value,
            "status": self.status.value,
            "answer_correct": self.answer_correct,
        }
        if self.used_model is not None:
            result["used_model"] = self.used_model
        if self.voting_details is not None:
            result["voting_details"] = self.voting_details
        if self.usage_stats is not None:
            result["usage_stats"] = self.usage_stats.to_dict()
        return result
