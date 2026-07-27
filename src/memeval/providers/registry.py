"""Model-to-provider registry."""

from __future__ import annotations

from memeval.config import APIConfig
from memeval.schema.diagnosis import ModelType

from .base import JudgeProvider
from .dashscope import DashScopeProvider
from .openai_compatible import OpenAICompatibleProvider


def get_provider(model: str | ModelType, api_config: APIConfig) -> JudgeProvider:
    model_name = model.value if isinstance(model, ModelType) else str(model)
    if model_name == ModelType.QWEN.value:
        return DashScopeProvider(api_config, default_model=api_config.dashscope_model)
    if model_name == ModelType.DEEPSEEK.value:
        return OpenAICompatibleProvider(api_config, name="deepseek", api_key=api_config.deepseek_api_key, base_url=api_config.deepseek_api_url, default_model=api_config.deepseek_model)
    if model_name in {ModelType.GPT_4_1.value, ModelType.GPT_5.value}:
        return OpenAICompatibleProvider(api_config, name="openai", api_key=api_config.openai_api_key, base_url=api_config.openai_api_url)
    if model_name in {ModelType.GEMINI.value, "gemini"}:
        return OpenAICompatibleProvider(api_config, name="gemini", api_key=api_config.gemini_api_key, base_url=api_config.gemini_url, default_model=api_config.gemini_model if model_name == "gemini" else None)
    raise ValueError(f"Unsupported model: {model_name}")
