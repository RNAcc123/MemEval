"""LLM API call helpers for the diagnosis pipeline."""

import json
import logging
import os
import re
import time
import warnings
from typing import Dict, Optional

from dotenv import load_dotenv

from memeval.config import APIConfig, DiagnosisConfig
from memeval.schema import ModelType, UsageStats
from memeval.providers import (
    RetryableProviderError,
    classify_provider_exception,
    get_provider,
)

__all__ = [
    "API_CONFIG",
    "call_llm_api",
    "clean_prompt",
    "extract_json_from_response",
    "is_retryable_provider_error",
]

logging.getLogger("grpc").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module="grpc")

load_dotenv()
os.environ["GRPC_ALTS_CREDENTIALS_ENVIRONMENT_OVERRIDE"] = "1"

API_CONFIG = APIConfig()

_ZERO_WIDTH_CHARS = "".join(
    chr(code) for code in (0x200B, 0x200C, 0x200D, 0xFEFF, 0x202A, 0x202B, 0x202C, 0x202D, 0x202E)
)
_ZERO_WIDTH_PATTERN = re.compile("[" + _ZERO_WIDTH_CHARS + "]")


def clean_prompt(prompt: str) -> str:
    """Remove special / zero-width characters from a prompt.

    Args:
        prompt: original prompt text

    Returns:
        Cleaned prompt text
    """
    return _ZERO_WIDTH_PATTERN.sub("", prompt)


def extract_json_from_response(response_text: str) -> Dict:
    """Extract a JSON object from the response text.

    Args:
        response_text: LLM response text

    Returns:
        Parsed JSON object

    Raises:
        Exception: raised when parsing fails
    """
    response_text = response_text.strip()
    start = response_text.find("{")
    end = response_text.rfind("}") + 1

    if start != -1 and end != 0:
        response_text = response_text[start:end]

    return json.loads(response_text)


def is_retryable_provider_error(error: Exception) -> bool:
    """Return whether an SDK/provider failure is likely to succeed on retry."""
    return isinstance(classify_provider_exception(error), RetryableProviderError)


def call_llm_api(
    prompt: str,
    model: str = "deepseek",
    config: Optional[DiagnosisConfig] = None,
    usage_stats: Optional[UsageStats] = None,
    stage_name: str = "",
) -> Dict:
    """Unified entrypoint for calling an LLM API.

    Args:
        prompt: input prompt text
        model: model name (string or ModelType)
        config: diagnosis configuration
        usage_stats: optional UsageStats tracker (calls/latency/tokens)
        stage_name: current diagnosis stage name (for stats)

    Returns:
        Parsed JSON response

    Raises:
        Exception: raised when API call or parsing fails
    """
    if config is None:
        config = DiagnosisConfig()

    prompt = clean_prompt(prompt)

    call_start_time = time.time()

    for attempt in range(config.max_retries):
        try:
            provider = get_provider(model, API_CONFIG)
            response = provider.complete(prompt, str(model.value if isinstance(model, ModelType) else model), config)
            break
        except KeyboardInterrupt:
            raise
        except Exception as e:
            provider_error = classify_provider_exception(e)
            if isinstance(provider_error, RetryableProviderError) and attempt < config.max_retries - 1:
                logging.warning(f"API call failed, retry {attempt + 1}/{config.max_retries}: {str(provider_error)}")
                time.sleep(config.retry_delay)
                continue
            if isinstance(provider_error, RetryableProviderError):
                raise RuntimeError(
                    f"API call failed after {config.max_retries} attempts: {str(provider_error)}"
                ) from provider_error
            raise provider_error from e

    call_latency = time.time() - call_start_time

    if usage_stats is not None:
        api_usage = response.usage
        prompt_tokens = api_usage.get("prompt_tokens", 0)
        completion_tokens = api_usage.get("completion_tokens", 0)
        total_tokens = api_usage.get("total_tokens", 0)

        usage_stats.record_call(
            latency=call_latency,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            model=str(model),
            stage=stage_name,
        )

    try:
        response_text = response.text.strip()

        return extract_json_from_response(response_text)
    except Exception as e:
        raise Exception(f"Failed to parse response: {str(e)}, raw response: {response_text[:200]}")
