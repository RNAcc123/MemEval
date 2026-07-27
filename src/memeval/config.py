"""Configuration models shared by MemEval commands and runners."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from memeval.schema.diagnosis import ModelType


@dataclass
class APIConfig:
    """Credentials and endpoints for supported judge providers."""

    dashscope_api_key: str = field(default_factory=lambda: os.getenv("DASHSCOPE_API_KEY", ""))
    deepseek_api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    deepseek_api_url: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_URL", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_api_url: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_URL", os.getenv("OPENAI_BASE_URL", ""))
    )
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_url: str = field(default_factory=lambda: os.getenv("GEMINI_URL", ""))
    dashscope_model: str = field(default_factory=lambda: os.getenv("DASHSCOPE_MODEL", "qwen-max"))
    deepseek_model: str = field(default_factory=lambda: os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner"))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-pro"))


@dataclass
class DiagnosisConfig:
    """Runtime settings for judge calls."""

    model: ModelType = ModelType.DEEPSEEK
    max_retries: int = 3
    retry_delay: int = 5
    temperature: float = 0.1
    timeout: int = 30
