"""OpenAI-compatible implementation of the answer generation backend."""

from __future__ import annotations

from typing import Any

from memeval.generation.traced import GenerationResponse
from memeval.providers.base import normalize_usage


class OpenAIChatGenerationBackend:
    name = "openai-compatible"

    def __init__(self, client: Any):
        self.client = client

    def complete(
        self,
        messages: list[dict[str, str]],
        model: str,
        parameters: dict[str, Any],
    ) -> GenerationResponse:
        response = self.client.chat.completions.create(model=model, messages=messages, **parameters)
        choice = response.choices[0]
        text = choice.message.content
        if not isinstance(text, str) or not text.strip():
            raise ValueError("OpenAI-compatible generation returned empty content")
        raw = {
            "request_id": getattr(response, "id", None),
            "created": getattr(response, "created", None),
            "model": getattr(response, "model", model),
        }
        return GenerationResponse(
            text=text,
            usage=normalize_usage(getattr(response, "usage", None)),
            finish_reason=getattr(choice, "finish_reason", None),
            raw=raw,
        )
