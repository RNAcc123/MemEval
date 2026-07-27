"""DashScope (Qwen) provider adapter."""

from __future__ import annotations

import logging

from memeval.config import APIConfig, DiagnosisConfig

from .base import ProviderResponse, classify_provider_exception, normalize_usage
from .errors import ProviderResponseError


class DashScopeProvider:
    name = "dashscope"

    def __init__(self, api_config: APIConfig, *, default_model: str = "qwen-max"):
        self.api_config = api_config
        self.default_model = default_model

    def complete(self, prompt: str, model: str, config: DiagnosisConfig) -> ProviderResponse:
        try:
            import dashscope
            from dashscope import Generation
        except ImportError as exc:
            raise ProviderResponseError("Please install the dashscope library: pip install dashscope") from exc
        if self.api_config.dashscope_api_key and not dashscope.api_key:
            dashscope.api_key = self.api_config.dashscope_api_key
        try:
            response = Generation.call(
                model=self.default_model if model == "qwen" else model,
                prompt=prompt,
                temperature=config.temperature,
                result_format="json",
                timeout=config.timeout,
            )
        except Exception as exc:
            logging.error("DashScope API call failed: %r", exc)
            raise classify_provider_exception(exc) from exc
        try:
            text = response.output.text
        except (AttributeError, TypeError) as exc:
            raise ProviderResponseError("DashScope response has no output text") from exc
        if not isinstance(text, str) or not text.strip():
            raise ProviderResponseError("DashScope response content is empty")
        return ProviderResponse(text=text, usage=normalize_usage(getattr(response, "usage", None), input_key="input_tokens", output_key="output_tokens"))
