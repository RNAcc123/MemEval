"""Provider-neutral contracts for judge model calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from memeval.config import DiagnosisConfig
from .errors import PermanentProviderError, RetryableProviderError


@dataclass(frozen=True)
class ProviderResponse:
    """Normalized text and token usage returned by a provider."""

    text: str
    usage: dict[str, int] = field(default_factory=dict)

    def to_legacy_dict(self) -> dict:
        return {"output": {"text": self.text}, "usage": self.usage}


class JudgeProvider(Protocol):
    name: str

    def complete(self, prompt: str, model: str, config: DiagnosisConfig) -> ProviderResponse:
        """Complete a single prompt using the configured provider."""


def normalize_usage(usage: object, *, input_key: str = "prompt_tokens", output_key: str = "completion_tokens") -> dict[str, int]:
    """Normalize SDK usage objects or mappings to the common token keys."""
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def value(key: str) -> int:
        if isinstance(usage, dict):
            raw = usage.get(key, 0)
        else:
            raw = getattr(usage, key, 0)
        try:
            return int(raw or 0)
        except (TypeError, ValueError):
            return 0

    prompt_tokens = value(input_key)
    completion_tokens = value(output_key)
    total = value("total_tokens") or prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total,
    }


def _status_code(error: Exception) -> int | None:
    response = getattr(error, "response", None)
    raw = getattr(error, "status_code", None) or getattr(response, "status_code", None)
    return raw if isinstance(raw, int) else None


def classify_provider_exception(error: Exception) -> Exception:
    """Translate SDK/HTTP exceptions into the shared taxonomy."""
    if isinstance(error, (RetryableProviderError, PermanentProviderError)):
        return error
    code = _status_code(error)
    name = {cls.__name__ for cls in type(error).__mro__}
    transient = code in {408, 409, 425, 429} or (code is not None and code >= 500)
    transient = transient or bool(name & {"APIConnectionError", "APITimeoutError", "InternalServerError", "RateLimitError", "ServiceUnavailableError", "TimeoutError", "Timeout"})
    if transient:
        return RetryableProviderError(str(error))
    return PermanentProviderError(str(error))

