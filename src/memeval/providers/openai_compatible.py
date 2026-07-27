"""Adapters for OpenAI-compatible chat completion endpoints."""

from __future__ import annotations

import logging

from memeval.config import APIConfig, DiagnosisConfig

from .base import JudgeProvider, ProviderResponse, classify_provider_exception, normalize_usage
from .errors import ProviderResponseError


class OpenAICompatibleProvider:
    def __init__(self, api_config: APIConfig, *, name: str, api_key: str, base_url: str = "", default_model: str | None = None):
        self.api_config = api_config
        self.name = name
        self.api_key = api_key
        self.base_url = base_url
        self.default_model = default_model

    def _client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ProviderResponseError("Please install the openai library: pip install openai") from exc
        kwargs = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return OpenAI(**kwargs)

    def complete(self, prompt: str, model: str, config: DiagnosisConfig) -> ProviderResponse:
        client = self._client()
        model = self.default_model or model
        kwargs = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False}
        if model != "gpt-5" and config.temperature is not None:
            kwargs["temperature"] = config.temperature
        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as exc:
            logging.error("%s API call failed: %r", self.name, exc)
            if "temperature" in repr(exc).lower() or "unsupported" in repr(exc).lower():
                kwargs.pop("temperature", None)
                try:
                    response = client.chat.completions.create(**kwargs)
                except Exception as retry_exc:
                    raise classify_provider_exception(retry_exc) from retry_exc
            else:
                raise classify_provider_exception(exc) from exc
        try:
            text = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise ProviderResponseError("OpenAI-compatible response has no message content") from exc
        if not isinstance(text, str) or not text.strip():
            raise ProviderResponseError("OpenAI-compatible response content is empty")
        return ProviderResponse(text=text, usage=normalize_usage(getattr(response, "usage", None)))
