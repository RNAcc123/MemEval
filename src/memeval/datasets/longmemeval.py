"""LongMemEval-S dataset adapter."""

from __future__ import annotations

import json
from pathlib import Path

from .base import EvaluationSample, validate_parallel_lengths


class LongMemEvalAdapter:
    name = "longmemeval_s"

    def validate(self, raw_data: object) -> None:
        if not isinstance(raw_data, list):
            raise ValueError("LongMemEval dataset must be a list")
        for index, item in enumerate(raw_data):
            if not isinstance(item, dict):
                raise ValueError(f"Sample {index} must be an object")
            required = ("haystack_dates", "haystack_session_ids", "haystack_sessions")
            missing = [key for key in required if key not in item]
            if missing:
                raise ValueError(f"Sample {index} missing keys: {', '.join(missing)}")
            validate_parallel_lengths(*(item[key] for key in required), context=f"Sample {index} haystack")

    def load(self, path: Path) -> list[EvaluationSample]:
        raw = json.loads(path.read_text(encoding="utf-8"))
        self.validate(raw)
        return [
            EvaluationSample(
                sample_id=str(item.get("question_id", index)),
                sessions=item["haystack_sessions"],
                session_ids=item["haystack_session_ids"],
                timestamps=item["haystack_dates"],
                questions=[item],
                subjects=[str(item.get("question_id", index))],
                metadata={"source": "longmemeval_s", "raw": item},
            )
            for index, item in enumerate(raw)
        ]
