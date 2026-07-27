from types import SimpleNamespace

import pytest

from memeval.config import APIConfig, DiagnosisConfig
from memeval.providers import (
    OpenAICompatibleProvider,
    PermanentProviderError,
    RetryableProviderError,
    classify_provider_exception,
    get_provider,
    normalize_usage,
)
from memeval.providers.base import ProviderResponse
from memeval.schema import ModelType


def test_registry_maps_supported_models():
    config = APIConfig()
    assert get_provider("deepseek", config).name == "deepseek"
    assert get_provider(ModelType.GPT_5, config).name == "openai"
    assert get_provider("qwen", config).name == "dashscope"
    assert get_provider("gemini", config).name == "gemini"


def test_registry_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unsupported model"):
        get_provider("unknown", APIConfig())


def test_usage_normalization_supports_dashscope_names():
    usage = normalize_usage({"input_tokens": "4", "output_tokens": 6}, input_key="input_tokens", output_key="output_tokens")
    assert usage == {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10}


def test_provider_response_keeps_legacy_shape():
    assert ProviderResponse("{}", {"total_tokens": 1}).to_legacy_dict() == {
        "output": {"text": "{}"},
        "usage": {"total_tokens": 1},
    }


def test_sdk_error_classification():
    class RateLimitError(Exception):
        pass

    assert isinstance(classify_provider_exception(RateLimitError()), RetryableProviderError)
    assert isinstance(classify_provider_exception(ValueError("bad request")), PermanentProviderError)


def test_openai_provider_normalizes_response_without_network(monkeypatch):
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))],
                usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5),
            )

    provider = OpenAICompatibleProvider(APIConfig(), name="fake", api_key="key")
    monkeypatch.setattr(provider, "_client", lambda: SimpleNamespace(chat=SimpleNamespace(completions=Completions())))
    response = provider.complete("prompt", "gpt-4.1", DiagnosisConfig())
    assert response.text == '{"ok": true}'
    assert response.usage["total_tokens"] == 5
    assert calls[0]["model"] == "gpt-4.1"
