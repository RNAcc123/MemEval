"""LoCoMo dataset adapter."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .base import EvaluationSample


class LoCoMoAdapter:
    name = "locomo"

    @staticmethod
    def session_keys(conversation: dict[str, Any]) -> list[str]:
        keys = [
            key for key, value in conversation.items()
            if re.fullmatch(r"session_\d+", key) and isinstance(value, list)
        ]
        return sorted(keys, key=lambda key: int(key.split("_")[1]))

    def validate(self, raw_data: object) -> None:
        if not isinstance(raw_data, list):
            raise ValueError("LoCoMo dataset must be a list")
        for index, item in enumerate(raw_data):
            if not isinstance(item, dict):
                raise ValueError(f"Sample {index} must be an object")
            missing = [key for key in ("sample_id", "conversation", "qa") if key not in item]
            if missing:
                raise ValueError(f"Sample {index} missing keys: {', '.join(missing)}")
            conversation = item["conversation"]
            if not isinstance(conversation, dict) or not {"speaker_a", "speaker_b"} <= conversation.keys():
                raise ValueError(f"Sample {index} missing speaker_a/speaker_b")
            if not self.session_keys(conversation):
                raise ValueError(f"Sample {index} has no session_N entries")

    def load(self, path: Path) -> list[EvaluationSample]:
        raw = json.loads(path.read_text(encoding="utf-8"))
        self.validate(raw)
        samples = []
        for item in raw:
            conversation = item["conversation"]
            keys = self.session_keys(conversation)
            samples.append(EvaluationSample(
                sample_id=str(item["sample_id"]),
                sessions=[conversation[key] for key in keys],
                session_ids=keys,
                timestamps=[conversation.get(f"{key}_date_time", "") for key in keys],
                questions=list(item["qa"]),
                subjects=[conversation["speaker_a"], conversation["speaker_b"]],
                metadata={"source": "locomo", "raw": item},
            ))
        return samples
