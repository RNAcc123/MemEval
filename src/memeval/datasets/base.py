"""Dataset-neutral evaluation sample contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class EvaluationSample:
    """Normalized sample shared by dataset-specific trace runners."""

    sample_id: str
    sessions: list[list[dict[str, Any]]] = field(default_factory=list)
    session_ids: list[str] = field(default_factory=list)
    timestamps: list[str] = field(default_factory=list)
    questions: list[dict[str, Any]] = field(default_factory=list)
    subjects: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class DatasetAdapter(Protocol):
    name: str

    def load(self, path: Path) -> list[EvaluationSample]:
        """Load and normalize a dataset file."""

    def validate(self, raw_data: object) -> None:
        """Validate the source dataset before normalization."""


def validate_parallel_lengths(*fields: list[Any], context: str = "dataset") -> None:
    lengths = [len(field) for field in fields]
    if len(set(lengths)) > 1:
        raise ValueError(f"{context} fields have different lengths: {lengths}")
