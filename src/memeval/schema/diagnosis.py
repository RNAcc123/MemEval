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
class MemoryData:
    """Normalized memory evolution and retrieval data for a QA record."""

    person1_memories: list[dict] = field(default_factory=list)
    person2_memories: list[dict] = field(default_factory=list)
    speaker1_retrieval: list[dict] = field(default_factory=list)
    speaker2_retrieval: list[dict] = field(default_factory=list)

    def to_json_str(self, field_name: str, exclude_keys: list[str] | None = None) -> str:
        value = getattr(self, field_name)
        if exclude_keys and isinstance(value, list):
            value = [
                {key: item_value for key, item_value in item.items() if key not in exclude_keys}
                if isinstance(item, dict)
                else item
                for item in value
            ]
        return json.dumps(value, ensure_ascii=False)


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
