"""
Memory diagnosis system - staged diagnosis for issues in QA pairs.

This module provides a staged diagnosis framework for identifying issue types
in a memory system:
- Stage 0: Consistency check
- Stage 1: Memory extraction diagnosis
- Stage 2: Memory update diagnosis
- Stage 3: Memory retrieval diagnosis
- Stage 4: Reasoning diagnosis
"""

# Standard library imports
import glob as glob_module
import json
import logging
import os
import re
import sys
import time
import threading
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

# Third-party imports
from dotenv import load_dotenv
from requests.exceptions import RequestException, Timeout

# Note: AI API-related imports are moved into their respective functions (lazy imports).
# This avoids breaking unrelated functionality when optional libraries are missing.

# ============================================================================
# Configuration and initialization
# ============================================================================

# Suppress gRPC warnings
logging.getLogger('grpc').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', module='grpc')

# Load environment variables
load_dotenv()
os.environ['GRPC_ALTS_CREDENTIALS_ENVIRONMENT_OVERRIDE'] = '1'


# ============================================================================
# Enums and constants
# ============================================================================

class ModelType(str, Enum):
    """Supported LLM model types."""
    QWEN = "qwen"
    DEEPSEEK = "deepseek"
    GPT_4_1 = "gpt-4.1"
    GPT_5 = "gpt-5"
    GEMINI = "gemini-2.5-pro"


class DiagnosisStage(str, Enum):
    """Diagnosis stage enum."""
    CONSISTENCY_CHECK = "0_consistency_check"
    MEMORY_EXTRACTION = "1_memory_extraction"
    MEMORY_UPDATE = "2_memory_update"
    MEMORY_RETRIEVAL = "3_memory_retrieval"
    REASONING = "4_reasoning"
    ERROR = "error"


# ============================================================================
# Config dataclasses
# ============================================================================

@dataclass
class APIConfig:
    """API configuration."""
    dashscope_api_key: str = field(default_factory=lambda: os.getenv("DASHSCOPE_API_KEY", ""))
    deepseek_api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    deepseek_api_url: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_URL", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_api_url: str = field(default_factory=lambda: os.getenv("OPENAI_API_URL", os.getenv("OPENAI_BASE_URL", "")))
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_url: str = field(default_factory=lambda: os.getenv("GEMINI_URL", ""))


@dataclass
class DiagnosisConfig:
    """Diagnosis configuration."""
    model: ModelType = ModelType.DEEPSEEK
    max_retries: int = 3
    retry_delay: int = 5
    temperature: float = 0.1
    timeout: int = 30


# ============================================================================
# Data containers
# ============================================================================

@dataclass
class QAData:
    """Container for QA data."""
    question: str
    answer: str
    response: str
    category: str = ""
    
    def to_json_str(self, field_name: str) -> str:
        """Convert the specified field to a JSON string."""
        value = getattr(self, field_name.replace("qa_", ""))
        return json.dumps(value, ensure_ascii=False)


@dataclass
class MemoryData:
    """Container for memory data."""
    person1_memories: List[dict] = field(default_factory=list)
    person2_memories: List[dict] = field(default_factory=list)
    speaker1_retrieval: List[Dict] = field(default_factory=list)
    speaker2_retrieval: List[Dict] = field(default_factory=list)
    
    def to_json_str(self, field_name: str, exclude_keys: Optional[List[str]] = None) -> str:
        """Convert the specified field to a JSON string.
        
        Args:
            field_name: field name
            exclude_keys: keys to exclude from each memory item
            
        Returns:
            JSON string
        """
        value = getattr(self, field_name)
        
        # If we need to exclude keys, filter the data
        if exclude_keys and isinstance(value, list):
            filtered_value = []
            for item in value:
                if isinstance(item, dict):
                    # Create a new dict excluding specific keys
                    filtered_item = {k: v for k, v in item.items() if k not in exclude_keys}
                    filtered_value.append(filtered_item)
                else:
                    filtered_value.append(item)
            return json.dumps(filtered_value, ensure_ascii=False)
        
        return json.dumps(value, ensure_ascii=False)


@dataclass
class StageResult:
    """Diagnosis result for a single stage."""
    stage_passed: bool
    label: Optional[str]
    reason: str
    stage: Optional[DiagnosisStage] = None


@dataclass
class DiagnosisResult:
    """Full diagnosis result."""
    label: Optional[str]
    reason: str
    stage: DiagnosisStage
    used_model: Optional[str] = None
    voting_details: Optional[Dict] = None
    usage_stats: Optional['UsageStats'] = None


@dataclass
class UsageStats:
    """API usage stats: track calls, latency, and token usage."""
    total_calls: int = 0
    total_latency: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    call_details: List[Dict] = field(default_factory=list)

    def record_call(
        self,
        latency: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        model: str = "",
        stage: str = "",
    ):
        """Record statistics for one API call."""
        self.total_calls += 1
        self.total_latency += latency
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens += total_tokens
        self.call_details.append({
            "model": model,
            "stage": stage,
            "latency_seconds": round(latency, 3),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        })

    def to_dict(self) -> Dict:
        """Convert to dict."""
        return {
            "total_calls": self.total_calls,
            "total_latency_seconds": round(self.total_latency, 3),
            "avg_latency_seconds": round(
                self.total_latency / self.total_calls, 3
            ) if self.total_calls > 0 else 0,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "call_details": self.call_details,
        }

    def merge(self, other: 'UsageStats'):
        """Merge another UsageStats into this one."""
        self.total_calls += other.total_calls
        self.total_latency += other.total_latency
        self.total_prompt_tokens += other.total_prompt_tokens
        self.total_completion_tokens += other.total_completion_tokens
        self.total_tokens += other.total_tokens
        self.call_details.extend(other.call_details)

    def print_summary(self):
        """Print a usage summary."""
        print(f"  📊 API call statistics:")
        print(f"     Calls: {self.total_calls}")
        print(f"     Total latency: {round(self.total_latency, 3)}s")
        if self.total_calls > 0:
            print(f"     Average latency: {round(self.total_latency / self.total_calls, 3)}s")
        print(f"     Prompt tokens: {self.total_prompt_tokens}")
        print(f"     Completion tokens: {self.total_completion_tokens}")
        print(f"     Total tokens: {self.total_tokens}")


# Initialize global config
API_CONFIG = APIConfig()
# ============================================================================
# Utility functions
# ============================================================================

def load_json_file(file_path: str) -> Dict:
    """Load a JSON file.
    
    Args:
        file_path: JSON file path
        
    Returns:
        Parsed dict
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_prompt(prompt: str) -> str:
    """Remove special / zero-width characters from a prompt.
    
    Args:
        prompt: original prompt text
        
    Returns:
        Cleaned prompt text
    """
    return re.sub(r"[\u200b\u200c\u200d\ufeff\u202a-\u202e]", "", prompt)


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


# ============================================================================
# LLM API call functions
# ============================================================================

def _extract_usage_from_response(response) -> Dict:
    """Extract token usage information from an OpenAI-compatible API response.

    Args:
        response: response object returned by the OpenAI client

    Returns:
        Dict containing prompt_tokens, completion_tokens, total_tokens
    """
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if hasattr(response, 'usage') and response.usage:
        usage["prompt_tokens"] = getattr(response.usage, 'prompt_tokens', 0) or 0
        usage["completion_tokens"] = getattr(response.usage, 'completion_tokens', 0) or 0
        usage["total_tokens"] = getattr(response.usage, 'total_tokens', 0) or 0
    return usage


def call_deepseek_api(prompt: str, temperature: float = 0.1) -> Dict:
    """Call the DeepSeek API.
    
    Args:
        prompt: input prompt
        temperature: temperature parameter
        
    Returns:
        Normalized response dict: {"output": {"text": "..."}}
        
    Raises:
        Exception: raised when the API call fails
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise Exception("Please install the openai library: pip install openai")
    
    client = OpenAI(
        api_key=API_CONFIG.deepseek_api_key,
        base_url=API_CONFIG.deepseek_api_url
    )
    
    try:
        kwargs = {
            "model": "deepseek-reasoner",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": temperature,
        }
        response = client.chat.completions.create(**kwargs)
        usage = _extract_usage_from_response(response)
        return {"output": {"text": response.choices[0].message.content}, "usage": usage}
    except Exception as e:
        # Print detailed debugging info
        try:
            resp = getattr(e, "response", None)
            if resp is not None:
                logging.error(f"DeepSeek response status: {getattr(resp, 'status_code', None)}")
                logging.error(f"DeepSeek response body: {getattr(resp, 'text', None)}")
        except Exception:
            pass
        
        logging.error(f"DeepSeek API call failed: {repr(e)}")
        
        # If this looks like a temperature-parameter issue, retry without temperature
        err_text = repr(e).lower()
        if "temperature" in err_text or "unsupported" in err_text:
            try:
                kwargs = {
                    "model": "deepseek-reasoner",
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                }
                response = client.chat.completions.create(**kwargs)
                usage = _extract_usage_from_response(response)
                return {"output": {"text": response.choices[0].message.content}, "usage": usage}
            except Exception as e2:
                logging.error(f"Retry (no temperature) also failed: {repr(e2)}")
                raise Exception(f"DeepSeek API error: {str(e2)}")
        
        raise Exception(f"DeepSeek API error: {str(e)}")


def call_openai_api(prompt: str, model: str = "gpt-4.1", temperature: float = 0.1) -> Dict:
    """Call the OpenAI API.
    
    Args:
        prompt: input prompt
        model: model name
        temperature: temperature parameter
        
    Returns:
        Normalized response dict: {"output": {"text": "..."}}
        
    Raises:
        Exception: raised when the API call fails
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise Exception("Please install the openai library: pip install openai")
    
    client_kwargs = {"api_key": API_CONFIG.openai_api_key}
    if API_CONFIG.openai_api_url:
        client_kwargs["base_url"] = API_CONFIG.openai_api_url
    client = OpenAI(**client_kwargs)
    
    # Some models do not support the temperature parameter
    temp_to_send = None if model == "gpt-5" else temperature
    
    try:
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        if temp_to_send is not None:
            kwargs["temperature"] = temp_to_send
            
        response = client.chat.completions.create(**kwargs)
        usage = _extract_usage_from_response(response)
        return {"output": {"text": response.choices[0].message.content}, "usage": usage}
    except Exception as e:
        logging.error(f"OpenAI API call failed: {repr(e)}")
        
        # If this looks like a temperature-parameter issue, retry without temperature
        err_text = repr(e).lower()
        if "temperature" in err_text or "unsupported" in err_text:
            try:
                kwargs = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                }
                response = client.chat.completions.create(**kwargs)
                usage = _extract_usage_from_response(response)
                return {"output": {"text": response.choices[0].message.content}, "usage": usage}
            except Exception as e2:
                logging.error(f"Retry (no temperature) also failed: {repr(e2)}")
                raise Exception(f"OpenAI API error: {str(e2)}")
        
        raise Exception(f"OpenAI API error: {str(e)}")


def call_gemini_api(prompt: str, model: str = "gemini-2.5-pro", temperature: float = 0.1) -> Dict:
    """Call the Gemini API.
    
    Args:
        prompt: input prompt
        model: model name
        temperature: temperature parameter
        
    Returns:
        Normalized response dict: {"output": {"text": "..."}}
        
    Raises:
        Exception: raised when the API call fails
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise Exception("Please install the openai library: pip install openai")
    
    try:
        client = OpenAI(api_key=API_CONFIG.gemini_api_key, base_url=API_CONFIG.gemini_url)
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
            
        response = client.chat.completions.create(**kwargs)
        usage = _extract_usage_from_response(response)
        return {"output": {"text": response.choices[0].message.content}, "usage": usage}
    except Exception as e:
        logging.error(f"Gemini API call failed: {repr(e)}")
        
        # If this looks like a temperature-parameter issue, retry without temperature
        err_text = repr(e).lower()
        if "temperature" in err_text or "unsupported" in err_text:
            try:
                kwargs = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                }
                response = client.chat.completions.create(**kwargs)
                usage = _extract_usage_from_response(response)
                return {"output": {"text": response.choices[0].message.content}, "usage": usage}
            except Exception as e2:
                logging.error(f"Retry (no temperature) also failed: {repr(e2)}")
                raise Exception(f"Gemini API error: {str(e2)}")
        
        raise Exception(f"Gemini API error: {str(e)}")

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
    
    # Clean special characters in the prompt
    prompt = clean_prompt(prompt)
    
    # Track call start time
    call_start_time = time.time()
    
    # Retry loop
    for attempt in range(config.max_retries):
        try:
            # Dispatch to the appropriate API based on model type
            if model == ModelType.QWEN or model == "qwen":
                try:
                    import dashscope
                    from dashscope import Generation
                    # Set API key (if not set yet)
                    if API_CONFIG.dashscope_api_key and not dashscope.api_key:
                        dashscope.api_key = API_CONFIG.dashscope_api_key
                except ImportError:
                    raise Exception("Please install the dashscope library: pip install dashscope")
                
                response = Generation.call(
                    model="qwen-max",
                    prompt=prompt,
                    temperature=config.temperature,
                    result_format="json",
                    timeout=config.timeout
                )
            elif model in [ModelType.GPT_4_1, ModelType.GPT_5, "gpt-4.1", "gpt-5"]:
                response = call_openai_api(
                    prompt,
                    model=model,
                    temperature=config.temperature
                )
            elif model == ModelType.DEEPSEEK or model == "deepseek":
                response = call_deepseek_api(
                    prompt,
                    temperature=config.temperature
                )
            else:
                response = call_gemini_api(
                    prompt,
                    temperature=config.temperature
                )
            break
        except (RequestException, Timeout, KeyboardInterrupt) as e:
            if attempt < config.max_retries - 1:
                logging.warning(f"API call failed, retry {attempt + 1}/{config.max_retries}: {str(e)}")
                time.sleep(config.retry_delay)
                continue
            else:
                raise Exception(f"API call failed (retried {config.max_retries} times): {str(e)}")
    
    # Compute call latency
    call_latency = time.time() - call_start_time
    
    # Extract token usage and record stats
    if usage_stats is not None:
        if model == ModelType.QWEN or model == "qwen":
            # Qwen/DashScope uses input_tokens / output_tokens
            qwen_usage = getattr(response, 'usage', None)
            prompt_tokens = getattr(qwen_usage, 'input_tokens', 0) or 0 if qwen_usage else 0
            completion_tokens = getattr(qwen_usage, 'output_tokens', 0) or 0 if qwen_usage else 0
            total_tokens = prompt_tokens + completion_tokens
        else:
            api_usage = response.get("usage", {})
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
    
    # Parse response
    try:
        if model == ModelType.QWEN or model == "qwen":
            response_text = response.output.text.strip()
        else:
            response_text = response["output"]["text"].strip()
        
        return extract_json_from_response(response_text)
    except Exception as e:
        raise Exception(f"Failed to parse response: {str(e)}, raw response: {response_text[:200]}")


# ============================================================================
# Diagnosis stage functions
# ============================================================================

def _print_stage_header(stage_name: str, stage_number: int = 0):
    """Print a stage header."""
    print("=" * 60)
    print(f"Stage {stage_number}: {stage_name}")
    print("=" * 60)


def stage0_consistency_check(
    qa_data: QAData,
    model: str = "deepseek",
    config: Optional[DiagnosisConfig] = None,
    usage_stats: Optional[UsageStats] = None,
) -> StageResult:
    """Stage 0: consistency check.
    
    Check whether the model response is consistent with the reference answer.
    
    Args:
        qa_data: QAData instance
        model: model to use
        config: diagnosis configuration
        usage_stats: optional UsageStats tracker
        
    Returns:
        StageResult containing the diagnosis outcome
    """
    _print_stage_header("Consistency Check", 0)
    
    qa_question_str = qa_data.to_json_str("question")
    qa_answer_str = qa_data.to_json_str("answer")
    qa_response_str = qa_data.to_json_str("response")
    
    prompt = f"""
You are an evaluation assistant. Determine whether qa_response is semantically consistent with qa_answer.

Consistency rules:
- All key information in qa_answer must appear in qa_response.
- Missing, incorrect or unclear details make it inconsistent.

Example :
qa_answer: "first weekend of August 2023"
qa_response: "5 August 2023."
→ inconsistent (incorrectly narrows the time range)

Now evaluate:
input:
- qa_question: {qa_question_str}
- qa_answer: {qa_answer_str}
- qa_response: {qa_response_str}

Output:
{{
  "is_consistent": true/false,
  "reason": "brief explanation"
}}
"""
    
    try:
        result = call_llm_api(prompt, model, config, usage_stats=usage_stats, stage_name="stage0_consistency_check")
        is_consistent = result.get("is_consistent", False)
        
        stage_result = StageResult(
            stage_passed=is_consistent,
            label=None if is_consistent else "inconsistent",
            reason=result.get("reason", ""),
            stage=DiagnosisStage.CONSISTENCY_CHECK
        )
        
        print(f"✓ Consistency check result: {'PASS' if is_consistent else 'FAIL'}")
        print(f"  Reason: {stage_result.reason}\n")
        
        return stage_result
    except Exception as e:
        logging.error(f"Stage 0 error: {str(e)}")
        return StageResult(
            stage_passed=False,
            label=None,
            reason=f"Stage 0 error: {str(e)}",
            stage=DiagnosisStage.ERROR
        )


def stage1_memory_extraction(
    qa_data: QAData,
    memory_data: MemoryData,
    model: str = "deepseek",
    config: Optional[DiagnosisConfig] = None,
    usage_stats: Optional[UsageStats] = None,
) -> StageResult:
    """Stage 1: memory extraction.
    
    Check whether the initially extracted memories are sufficient.
    
    Args:
        qa_data: QAData instance
        memory_data: MemoryData instance
        model: model to use
        config: diagnosis configuration
        usage_stats: optional UsageStats tracker
        
    Returns:
        StageResult containing the diagnosis outcome
    """
    _print_stage_header("Memory Extraction Stage", 1)
    
    qa_question_str = qa_data.to_json_str("question")
    qa_answer_str = qa_data.to_json_str("answer")
    qa_response_str = qa_data.to_json_str("response")
    # Stage 1 uses only initial_results; keep time_stamp for time-related reasoning when needed
    memories1_initial_results = [
        {
            "time_stamp": item.get("time_stamp", ""),
            "initial_results": item.get("initial_results", []),
        }
        for item in memory_data.person1_memories
    ]
    memories2_initial_results = [
        {
            "time_stamp": item.get("time_stamp", ""),
            "initial_results": item.get("initial_results", []),
        }
        for item in memory_data.person2_memories
    ]
    memories1_str = json.dumps(memories1_initial_results, ensure_ascii=False)
    memories2_str = json.dumps(memories2_initial_results, ensure_ascii=False)
    
    prompt = f"""
You are an evaluation assistant for the Memory Extraction Stage.
Task:
1. Use their initial_results (and time_stamp if needed) to determine whether the extracted memories are sufficient to answer qa_question.
2. If sufficient → is_sufficient = true (label = null).
3. If insufficient, classify the issue:
   - "1.1": Missing key information
   - "1.2": Incorrect or conflicting information
   - "1.3": Ambiguous or overly generic information

Examples:

Example 1:
qa_question: "Where did Caroline move from 4 years ago?"
qa_answer: "Sweden"
qa_response: "home country"
person1_memories: {{"initial_results": ["Caroline moved from her home country 4 years ago"]}}
person2_memories: {{"initial_results": []}}
Output:
{{
  "is_sufficient": false,
  "label": "1.1",
  "reason": "The extracted memory only says 'home country' and lacks the specific detail 'Sweden.'"
}}

Example 2:
qa_question: "What kind of films does Joanna enjoy?"
qa_answer: "Dramas and emotionally-driven films"
qa_response: "dramas and romantic comedies"
person1_memories: {{"initial_results": ["Joanna enjoys dramas and emotionally-driven films."]}}
person2_memories: {{"initial_results": ["Joanna enjoys dramas and romantic comedies."]}}
Output:
{{
  "is_sufficient": false,
  "label": "1.2",
  "reason": "The memories conflict—one mentions emotionally-driven films, the other romantic comedies—indicating incorrect/inconsistent extraction."
}}

Example 3:
qa_question: "What food item did Maria drop off at the homeless shelter?"
qa_answer: "Cakes"
qa_response: "baked goods"
person1_memories: {{"initial_results": ["Maria dropped off baked goods..."]}}
person2_memories: {{"initial_results": ["Maria dropped off baked goods..."]}}
Output:
{{
  "is_sufficient": false,
  "label": "1.3",
  "reason": "The extracted memory is too vague ('baked goods') and does not specify 'cakes.'"
}}

Now evaluate the following:

Input:
- qa_question: {qa_question_str}
- qa_answer: {qa_answer_str}
- qa_response: {qa_response_str}
- person1_memories: {memories1_str}
- person2_memories: {memories2_str}

Output format:
{{
  "is_sufficient": true/false,
  "label": "1.1" or "1.2" or "1.3" or null,
  "reason": "Detailed explanation"
}}

"""
    
    try:
        result = call_llm_api(prompt, model, config, usage_stats=usage_stats, stage_name="stage1_memory_extraction")
        is_sufficient = result.get("is_sufficient", False)
        
        stage_result = StageResult(
            stage_passed=is_sufficient,
            label=None if is_sufficient else result.get("label"),
            reason=result.get("reason", ""),
            stage=DiagnosisStage.MEMORY_EXTRACTION
        )
        
        print(f"✓ Memory extraction result: {'PASS' if is_sufficient else 'FAIL'}")
        if not is_sufficient:
            print(f"  Issue type: {stage_result.label}")
        print(f"  Reason: {stage_result.reason}\n")
        
        return stage_result
    except Exception as e:
        logging.error(f"Stage 1 error: {str(e)}")
        return StageResult(
            stage_passed=False,
            label=None,
            reason=f"Stage 1 error: {str(e)}",
            stage=DiagnosisStage.ERROR
        )


def stage2_memory_update(
    qa_data: QAData,
    memory_data: MemoryData,
    model: str = "deepseek",
    config: Optional[DiagnosisConfig] = None,
    usage_stats: Optional[UsageStats] = None,
) -> StageResult:
    """Stage 2: memory update.
    
    Check whether the memory update process is correct.
    
    Args:
        qa_data: QAData instance
        memory_data: MemoryData instance
        model: model to use
        config: diagnosis configuration
        usage_stats: optional UsageStats tracker
        
    Returns:
        StageResult containing the diagnosis outcome
    """
    _print_stage_header("Memory Update Stage", 2)
    
    qa_question_str = qa_data.to_json_str("question")
    qa_answer_str = qa_data.to_json_str("answer")
    qa_response_str = qa_data.to_json_str("response")
    # Stage 2 uses only update_chain; keep time_stamp to judge whether updates are time-consistent when needed
    memories1_update_chains = [
        {
            "time_stamp": item.get("time_stamp", ""),
            "update_chain": item.get("update_chain", []),
        }
        for item in memory_data.person1_memories
    ]
    memories2_update_chains = [
        {
            "time_stamp": item.get("time_stamp", ""),
            "update_chain": item.get("update_chain", []),
        }
        for item in memory_data.person2_memories
    ]
    memories1_str = json.dumps(memories1_update_chains, ensure_ascii=False)
    memories2_str = json.dumps(memories2_update_chains, ensure_ascii=False)
    
    prompt = f"""
You are an evaluation assistant for the Memory Update Stage.
Task:
1. From the update_chain, use only the final updated memory for each item.
2. Determine whether the updated memories contain sufficient and correct information to answer qa_question.
3. If sufficient → is_sufficient = true (label = null).
4. If insufficient, classify the issue according to the update error type:
   - "2.1": Incorrect update (added wrong or fabricated details)
   - "2.2": Deleted information (removed necessary memory entries)
   - "2.3": Weakened information (kept but diluted or less specific)

Examples:

Example 1:
qa_question: "What did James prepare for the first time in the cooking class?"
qa_answer: "Omelette"
qa_response: "omelette, meringue, dough"
update_chain: [{{
  "event": "UPDATE",
  "memory": "James ... made an omelette ... He also made meringue and learned how to make dough.",
  "previous_memory": "James ... made a great omelette for the first time."
}}]
Output:
{{
  "is_sufficient": false,
  "label": "2.1",
  "reason": "The update introduces incorrect new first-time dishes—meringue and dough—contradicting the original memory."
}}

Example 2:
qa_question: "When did Maria adopt Shadow?"
qa_answer: "The week before 13 August 2023"
qa_response: "13 August, 2023"
update_chain: [{{
  "event": "DELETE",
  "memory": "Maria adopted a cute puppy from a shelter last week, and she feels blessed to give her a home."
}}]
Output:
{{
  "is_sufficient": false,
  "label": "2.2",
  "reason": "Because the update event is a DELETE operation, it removes the memory stating that Maria adopted the puppy the previous week, eliminating the key information needed to infer the correct adoption timeframe."
}}

Example 3:
qa_question: "How many times has Jolene been to France?"
qa_answer: "two times"
qa_response: "None."
update_chain: [{{
  "event": "UPDATE",
  "memory": "Jolene has a pendant that represents freedom...",
  "previous_memory": "Jolene has a pendant her mother gave her in 2010 in Paris."
}}]
Output:
{{
  "is_sufficient": false,
  "label": "2.3",
  "reason": "The update removes the Paris detail, weakening the information needed to infer her past visits to France."
}}

Now evaluate the following:

Input:
- qa_question: {qa_question_str}
- qa_answer: {qa_answer_str}
- qa_response: {qa_response_str}
- person1_memories: {memories1_str}
- person2_memories: {memories2_str}

Output format:
{{
  "is_sufficient": true/false,
  "label": "2.1" or "2.2" or "2.3" or null,
  "reason": "Detailed explanation"
}}
"""
    
    try:
        result = call_llm_api(prompt, model, config, usage_stats=usage_stats, stage_name="stage2_memory_update")
        is_sufficient = result.get("is_sufficient", False)
        
        stage_result = StageResult(
            stage_passed=is_sufficient,
            label=None if is_sufficient else result.get("label"),
            reason=result.get("reason", ""),
            stage=DiagnosisStage.MEMORY_UPDATE
        )
        
        print(f"✓ Memory update result: {'PASS' if is_sufficient else 'FAIL'}")
        if not is_sufficient:
            print(f"  Issue type: {stage_result.label}")
        print(f"  Reason: {stage_result.reason}\n")
        
        return stage_result
    except Exception as e:
        logging.error(f"Stage 2 error: {str(e)}")
        return StageResult(
            stage_passed=False,
            label=None,
            reason=f"Stage 2 error: {str(e)}",
            stage=DiagnosisStage.ERROR
        )


def stage3_memory_retrieval(
    qa_data: QAData,
    memory_data: MemoryData,
    model: str = "deepseek",
    config: Optional[DiagnosisConfig] = None,
    usage_stats: Optional[UsageStats] = None,
) -> StageResult:
    """Stage 3: memory retrieval.
    
    Check whether memory retrieval is correct/sufficient.
    
    Args:
        qa_data: QAData instance
        memory_data: MemoryData instance
        model: model to use
        config: diagnosis configuration
        usage_stats: optional UsageStats tracker
        
    Returns:
        StageResult containing the diagnosis outcome
    """
    _print_stage_header("Memory Retrieval Stage", 3)
    
    qa_question_str = qa_data.to_json_str("question")
    qa_answer_str = qa_data.to_json_str("answer")
    speaker1_memories_str = memory_data.to_json_str("speaker1_retrieval")
    speaker2_memories_str = memory_data.to_json_str("speaker2_retrieval")
    
    prompt = f"""
You are an evaluation assistant for the Memory Retrieval Stage.
Task:
Based strictly on speaker1_retrieval and speaker2_retrieval:
1. Determine whether the retrieved memories contain enough correct information to answer qa_question.
2. If sufficient → is_sufficient = true (label = null).
3. If insufficient, determine the retrieval issue:
   - "3.1": Failed to recall correct information (missing the key facts)
   - "3.2": Unreasonable ranking (recalled irrelevant/common info while missing the most relevant facts)

Examples:

Example 1:
qa_question: "How does Melanie prioritize self-care?"
qa_answer: "by carving out some me-time each day for activities like running, reading, or playing the violin"
qa_response: "Running, pottery, charity races."
speaker1_retrieval: [
  "Melanie prioritizes her mental health...",
  "Melanie enjoys running as a way to de-stress...",
  "Melanie is thankful for her family..."
]
speaker2_retrieval: [
  "Melanie finds self-care to be a work in progress...",
  "Melanie has been running longer...",
  "Melanie values mental health..."
]
Output:
{{
  "is_sufficient": false,
  "label": "3.1",
  "reason": "The retrieved memories mention running and mental-health efforts but miss key self-care details such as reading, violin, and daily me-time."
}}

Now evaluate the following:

Input:
- qa_question: {qa_question_str}
- qa_answer: {qa_answer_str}
- speaker1_retrieval: {speaker1_memories_str}
- speaker2_retrieval: {speaker2_memories_str}

Output format:
{{
  "is_sufficient": true/false,
  "label": "3.1" or "3.2" or null,
  "reason": "Detailed explanation"
}}
"""
    
    try:
        result = call_llm_api(prompt, model, config, usage_stats=usage_stats, stage_name="stage3_memory_retrieval")
        is_sufficient = result.get("is_sufficient", False)
        
        stage_result = StageResult(
            stage_passed=is_sufficient,
            label=None if is_sufficient else result.get("label"),
            reason=result.get("reason", ""),
            stage=DiagnosisStage.MEMORY_RETRIEVAL
        )
        
        print(f"✓ Memory retrieval result: {'PASS' if is_sufficient else 'FAIL'}")
        if not is_sufficient:
            print(f"  Issue type: {stage_result.label}")
        print(f"  Reason: {stage_result.reason}\n")
        
        return stage_result
    except Exception as e:
        logging.error(f"Stage 3 error: {str(e)}")
        return StageResult(
            stage_passed=False,
            label=None,
            reason=f"Stage 3 error: {str(e)}",
            stage=DiagnosisStage.ERROR
        )


def stage4_reasoning(
    qa_data: QAData,
    memory_data: MemoryData,
    model: str = "deepseek",
    config: Optional[DiagnosisConfig] = None,
    usage_stats: Optional[UsageStats] = None,
) -> StageResult:
    """Stage 4: reasoning.
    
    If all previous stages pass, remaining issues are attributed to reasoning.
    
    Args:
        qa_data: QAData instance
        memory_data: MemoryData instance
        model: model to use
        config: diagnosis configuration
        usage_stats: optional UsageStats tracker
        
    Returns:
        StageResult containing the diagnosis outcome
    """
    _print_stage_header("Reasoning Stage", 4)
    
    qa_question_str = qa_data.to_json_str("question")
    qa_answer_str = qa_data.to_json_str("answer")
    qa_response_str = qa_data.to_json_str("response")
    speaker1_memories_str = memory_data.to_json_str("speaker1_retrieval")
    speaker2_memories_str = memory_data.to_json_str("speaker2_retrieval")
    
    prompt = f"""
You are an evaluation assistant for the Reasoning Stage.

Context:
All previous stages (extraction, update, retrieval) have passed, meaning the model had sufficient correct information.  
If qa_response still does not match qa_answer, the issue is a reasoning error.

Task:
Based on qa_question, qa_answer, qa_response, and the retrieved memories, classify the reasoning issue:
- "4.1": Correct memory entries were ignored (model overlooks correct memory entries present in retrieval)
- "4.2": Reasoning error (model invents details, over-specifies, or makes unsupported inferences)
- "4.3": Format or detail error (minor deviations such as missing qualifiers or altered phrasing that slightly change meaning)

Examples:

Example 1:
qa_question: "What does Melanie do with her family on hikes?"
qa_answer: "Roast marshmallows, tell stories"
qa_response: "explore nature and bond"
speaker1_retrieval: ["Melanie prioritizes her mental health..."]
speaker2_retrieval: ["Melanie ... roasted marshmallows ... and told stories..."]
Output:
{{
  "label": "4.1",
  "reason": "The retrieved memory clearly includes roasting marshmallows and telling stories, but the model ignored this memory entry"
}}

Example 2:
qa_question: "When did Caroline have a picnic?"
qa_answer: "The week before 6 July 2023"
qa_response: "29 June 2023."
Output:
{{
  "label": "4.2",
  "reason": "The answer only specifies a time range, but the model unjustifiably inferred an exact date."
}}

Example 3:
qa_question: "How often does John see sunsets like the one he shared with Maria?"
qa_answer: "At least once a week"
qa_response: "once a week"
Output:
{{
  "label": "4.3",
  "reason": "The model dropped the qualifier 'at least,' slightly altering the meaning."
}}

Now evaluate the following:

Input:
- qa_question: {qa_question_str}
- qa_answer: {qa_answer_str}
- qa_response: {qa_response_str}
- speaker1_retrieval: {speaker1_memories_str}
- speaker2_retrieval: {speaker2_memories_str}

Output format:
{{
  "label": "4.1" or "4.2" or "4.3",
  "reason": "Detailed explanation"
}}
"""
    
    try:
        result = call_llm_api(prompt, model, config, usage_stats=usage_stats, stage_name="stage4_reasoning")
        
        stage_result = StageResult(
            stage_passed=False,
            label=result.get("label"),
            reason=result.get("reason", ""),
            stage=DiagnosisStage.REASONING
        )
        
        print(f"✓ Reasoning issue type: {stage_result.label}")
        print(f"  Reason: {stage_result.reason}\n")
        
        return stage_result
    except Exception as e:
        logging.error(f"Stage 4 error: {str(e)}")
        return StageResult(
            stage_passed=False,
            label="4.2",
            reason=f"Stage 4 error: {str(e)}",
            stage=DiagnosisStage.ERROR
        )


# ============================================================================
# Main diagnosis function
# ============================================================================

def analyze_qa_pair(
    qa_data: QAData,
    memory_data: MemoryData,
    model: str = "deepseek",
    config: Optional[DiagnosisConfig] = None
) -> DiagnosisResult:
    """Main staged diagnosis function.
    
    Execute diagnosis stages in order until an issue is found or all stages pass.
    
    Args:
        qa_data: QAData instance
        memory_data: MemoryData instance
        model: model to use
        config: diagnosis configuration
        
    Returns:
        DiagnosisResult containing the full diagnosis outcome
    """
    print(f"\n{'='*70}")
    print(f"🔍 Start staged diagnosis")
    print(f"📝 Question: {qa_data.question}")
    print(f"{'='*70}\n")
    
    # Create a usage stats tracker
    stats = UsageStats()
    
    try:
        # Stage 0: consistency check
        stage0_result = stage0_consistency_check(qa_data, model, config, usage_stats=stats)
        if stage0_result.stage_passed:
            result = DiagnosisResult(
                label=None,
                reason=stage0_result.reason,
                stage=DiagnosisStage.CONSISTENCY_CHECK,
                usage_stats=stats,
            )
            stats.print_summary()
            return result
        
        # Stage 1: memory extraction
        stage1_result = stage1_memory_extraction(qa_data, memory_data, model, config, usage_stats=stats)
        if not stage1_result.stage_passed:
            result = DiagnosisResult(
                label=stage1_result.label,
                reason=stage1_result.reason,
                stage=DiagnosisStage.MEMORY_EXTRACTION,
                usage_stats=stats,
            )
            stats.print_summary()
            return result
        
        # Stage 2: memory update
        stage2_result = stage2_memory_update(qa_data, memory_data, model, config, usage_stats=stats)
        if not stage2_result.stage_passed:
            result = DiagnosisResult(
                label=stage2_result.label,
                reason=stage2_result.reason,
                stage=DiagnosisStage.MEMORY_UPDATE,
                usage_stats=stats,
            )
            stats.print_summary()
            return result
        
        # Stage 3: memory retrieval
        stage3_result = stage3_memory_retrieval(qa_data, memory_data, model, config, usage_stats=stats)
        if not stage3_result.stage_passed:
            result = DiagnosisResult(
                label=stage3_result.label,
                reason=stage3_result.reason,
                stage=DiagnosisStage.MEMORY_RETRIEVAL,
                usage_stats=stats,
            )
            stats.print_summary()
            return result
        
        # Stage 4: reasoning (previous stages passed; issue is in reasoning)
        stage4_result = stage4_reasoning(qa_data, memory_data, model, config, usage_stats=stats)
        result = DiagnosisResult(
            label=stage4_result.label,
            reason=stage4_result.reason,
            stage=DiagnosisStage.REASONING,
            usage_stats=stats,
        )
        stats.print_summary()
        return result
        
    except Exception as e:
        logging.error(f"Error during diagnosis: {str(e)}")
        return DiagnosisResult(
            label=None,
            reason=f"Error during diagnosis: {str(e)}",
            stage=DiagnosisStage.ERROR,
            usage_stats=stats,
        )


def analyze_qa_pair_legacy(
    qa_question: str,
    qa_answer: str,
    qa_response: str,
    memories1: List[dict],
    memories2: List[dict],
    speaker1_memories: List[Dict],
    speaker2_memories: List[Dict],
    model: str = "deepseek"
) -> Dict:
    """Legacy compatibility wrapper.
    
    Keeps compatibility with older code by converting arguments into the new
    dataclasses and calling the new analysis function.
    
    Args:
        qa_question: question text
        qa_answer: reference answer
        qa_response: model response
        memories1: person1 memory data
        memories2: person2 memory data
        speaker1_memories: retrieved memories for speaker1
        speaker2_memories: retrieved memories for speaker2
        model: model to use
        
    Returns:
        Diagnosis result dict (legacy format)
    """
    # Build data objects
    qa_data = QAData(
        question=qa_question,
        answer=qa_answer,
        response=qa_response
    )
    
    memory_data = MemoryData(
        person1_memories=memories1,
        person2_memories=memories2,
        speaker1_retrieval=speaker1_memories,
        speaker2_retrieval=speaker2_memories
    )
    
    # Call the new function
    result = analyze_qa_pair(qa_data, memory_data, model)
    
    # Convert to the legacy format
    return {
        "label": result.label,
        "reason": result.reason,
        "stage": result.stage.value if isinstance(result.stage, DiagnosisStage) else result.stage
    }

def analyze_qa_pair_with_voting(
    qa_question: str,
    qa_answer: str,
    qa_response: str,
    memories1: List[dict],
    memories2: List[dict],
    speaker1_memories: List[Dict],
    speaker2_memories: List[Dict],
    model: str = "deepseek",
    num_votes: int = 3
) -> Dict:
    """Analyze a QA pair (and retrieved memories) using a voting mechanism.
    
    Args:
        qa_question: question text
        qa_answer: reference answer
        qa_response: model response
        memories1: person1 memory data
        memories2: person2 memory data
        speaker1_memories: retrieved memories for speaker1
        speaker2_memories: retrieved memories for speaker2
        model: primary model to use
        num_votes: number of voting rounds
        
    Returns:
        Dict containing the final diagnosis result and voting details
    """
    print(f"\n🗳️  Question: {qa_question}")
    print(f"📊 Run {num_votes} voting rounds with {model} as the primary model (a different model may be used each round)\n")
    
    # Build data objects (create once and reuse)
    qa_data = QAData(
        question=qa_question,
        answer=qa_answer,
        response=qa_response
    )
    
    memory_data = MemoryData(
        person1_memories=memories1,
        person2_memories=memories2,
        speaker1_retrieval=speaker1_memories,
        speaker2_retrieval=speaker2_memories
    )
    
    # Store results for each round
    vote_results = []
    
    # Aggregate usage stats across all voting rounds
    aggregated_stats = UsageStats()
    
    # Define the model list for rotation
    models = ["deepseek", "gpt-4.1", "gpt-5"]
    
    # Ensure the primary model is in the list; insert if missing
    if model not in models:
        models.insert(0, model)
    else:
        # Move the primary model to the front
        models.remove(model)
        models.insert(0, model)
    
    # Run multiple voting rounds, using different models each round
    used_models = []
    for i in range(num_votes):
        # Select model: prefer unused models
        current_model = None
        for m in models:
            if m not in used_models:
                current_model = m
                break
        
        # If all models have been used, choose from non-primary models when possible
        if current_model is None:
            unused_models = [m for m in models if m != model]
            if unused_models:
                current_model = unused_models[len(used_models) % len(unused_models)]
            else:
                current_model = models[len(used_models) % len(models)]
        
        used_models.append(current_model)
        print(f"🔄 Round {i+1}/{num_votes}, model: {current_model}")
        
        try:
            # Use the new dataclass-based interface
            result = analyze_qa_pair(qa_data, memory_data, model=current_model)
            # Merge usage stats from this round
            if result.usage_stats is not None:
                aggregated_stats.merge(result.usage_stats)
            # Convert to dict and attach model info
            result_dict = {
                "label": result.label,
                "reason": result.reason,
                "stage": result.stage.value if isinstance(result.stage, DiagnosisStage) else result.stage,
                "used_model": current_model
            }
            vote_results.append(result_dict)
            print(f"   ✅ Round {i+1} completed: label={result.label}, model={current_model}\n")
        except Exception as e:
            logging.error(f"Round {i+1} analysis failed: {str(e)}")
            print(f"   ❌ Round {i+1} analysis failed: {str(e)}\n")
            # If a round fails, append a default result (label = null)
            vote_results.append({
                "label": None,
                "reason": f"API call failed: {str(e)}",
                "stage": "error",
                "used_model": current_model
            })
    
    # Count vote results (including None labels)
    labels = [result["label"] for result in vote_results]
    
    # Pick the most frequent label (including None)
    label_counter = Counter(labels)
    most_common_label = label_counter.most_common(1)[0][0]

    
    # Get the details for the final selected result
    final_result = None
    
    # Check whether all vote results are different (no repeated labels)
    all_different = len(label_counter) == len(vote_results) and len(vote_results) > 1
    
    if all_different:
        # If all results differ, use the primary model's result
        print(f"All voting results are different; use the primary model ({model}) result")
        for result in vote_results:
            if result.get("used_model") == model:
                final_result = result
                most_common_label = result["label"]
                break
        
        # If the primary model result is missing, fall back to the first result
        if final_result is None:
            print(f"Primary model result ({model}) not found; fall back to the first result")
            final_result = vote_results[0]
            most_common_label = final_result["label"]
    else:
        # Select the final result by the most-voted label
        for result in vote_results:
            if result["label"] == most_common_label:
                final_result = result
                break
        
        # If no matching result is found (should not happen), use the first result
        if final_result is None and vote_results:
            final_result = vote_results[0]
    
    # Simplify voting_details: include only per-round annotation results
    final_result["voting_details"] = {
        "label_votes": dict(label_counter),
        "individual_results": [
            {
                "label": result["label"],
                "used_model": result.get("used_model", "unknown"),
                "reason": result.get("reason", "")
            } 
            for result in vote_results
        ],
        "all_different": all_different  # Flag: whether all results are different
    }
    
    # Attach aggregated usage stats to the final result
    final_result["usage_stats"] = aggregated_stats.to_dict()
    
    # Print voting summary
    print(f"{'='*70}")
    print(f"📊 Voting summary")
    print(f"{'='*70}")
    print(f"🤖 Model order used: {used_models}")
    
    # Print vote result safely
    vote_count = label_counter[most_common_label] if not all_different else 1
    print(f"🏆 Final label: {most_common_label} (votes: {vote_count}/{num_votes})")
    if all_different:
        print(f"⚠️  All voting results are different; selected the DeepSeek model result")
    
    # Print API usage stats across all votes
    aggregated_stats.print_summary()
    print(f"{'='*70}\n")
    
    return final_result


# ============================================================================
# Single-file processing (thread-safe)
# ============================================================================

# Thread-safe print lock
_print_lock = threading.Lock()


def _thread_print(*args, **kwargs):
    """Thread-safe print helper."""
    with _print_lock:
        print(*args, **kwargs)


def process_single_file(
    input_file: str,
    output_file: str,
    model: str,
    use_voting: bool,
    num_votes: int,
    qa_threads: int = 1,
    thread_label: str = "",
) -> Tuple[int, UsageStats]:
    """Run diagnosis for a single input file.

    This function is safe to call from multiple threads; each thread processes
    an independent input/output file pair.

    Args:
        input_file: input JSON file path
        output_file: output JSON file path
        model: model name to use
        use_voting: whether to use voting
        num_votes: number of voting rounds
        qa_threads: number of worker threads for QA items within this file
        thread_label: thread label (for log prefix)

    Returns:
        Tuple of (num_processed_items, file_level_usage_stats)
    """
    prefix = f"[{thread_label}] " if thread_label else ""

    # Validate input file
    if not os.path.exists(input_file):
        _thread_print(f"{prefix}❌ Error: Input file does not exist: {input_file}")
        return 0, UsageStats()

    # Load input data
    try:
        data = load_json_file(input_file)
        _thread_print(f"{prefix}✅ Loaded {input_file} successfully, total conversations: {len(data)}")
    except Exception as e:
        logging.error(f"{prefix}Failed to load input file: {str(e)}")
        _thread_print(f"{prefix}❌ Error: Failed to parse input file: {str(e)}")
        return 0, UsageStats()

    # Load previously processed results (resume support)
    results = []
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                results = json.load(f)
                _thread_print(f"{prefix}📂 Loaded {len(results)} historical results (resume enabled)")
        except (json.JSONDecodeError, FileNotFoundError):
            results = []

    processed_items = {item["conv_id_question_id"] for item in results}

    # File-level usage tracker
    file_stats = UsageStats()

    try:
        total_convs = len(data)
        _thread_print(f"{prefix}📊 Start processing, total conversations: {total_convs}\n")

        pending_items = []
        for conv_idx, (conv_id, qa_list) in enumerate(data.items(), 1):
            _thread_print(f"\n{prefix}{'='*60}")
            _thread_print(f"{prefix}📝 Processing conversation {conv_id} ({conv_idx}/{total_convs})")
            _thread_print(f"{prefix}{'='*60}\n")

            for qa_idx, qa_item in enumerate(qa_list, 1):
                item_id = f"{conv_id}_{qa_idx-1}"

                if item_id in processed_items:
                    _thread_print(f"{prefix}⏭️  Skip already processed item: {item_id}")
                    continue

                pending_items.append((conv_id, qa_idx, len(qa_list), qa_item, item_id))

        def process_qa_item(task: Tuple[str, int, int, Dict, str]) -> Optional[Tuple[Dict, UsageStats]]:
            _conv_id, qa_idx, qa_count, qa_item, item_id = task
            _thread_print(f"{prefix}🔍 Processing question {qa_idx}/{qa_count}: {item_id}")

            try:
                p1 = qa_item.get("person1", {})
                p2 = qa_item.get("person2", {})
                memories1 = p1.get("memories", [])
                memories2 = p2.get("memories", [])
                item_stats = UsageStats()

                if use_voting:
                    analysis = analyze_qa_pair_with_voting(
                        qa_question=qa_item["qa_question"],
                        qa_answer=qa_item["qa_answer"],
                        qa_response=qa_item["qa_response"],
                        memories1=memories1,
                        memories2=memories2,
                        speaker1_memories=qa_item.get("speaker_1_memories", []),
                        speaker2_memories=qa_item.get("speaker_2_memories", []),
                        model=model,
                        num_votes=num_votes,
                    )

                    result = {
                        "conv_id_question_id": item_id,
                        "qa_question": qa_item["qa_question"],
                        "qa_answer": qa_item["qa_answer"],
                        "qa_response": qa_item["qa_response"],
                        "qa_category": qa_item.get("qa_category", ""),
                        "label": analysis["label"],
                        "reason": analysis["reason"],
                        "diagnosis_mode": f"voting_{num_votes}rounds",
                    }

                    if "voting_details" in analysis:
                        result["voting_details"] = {
                            "label_votes": analysis["voting_details"]["label_votes"],
                            "individual_results": [
                                {
                                    "label": ir["label"],
                                    "used_model": ir.get("used_model", "unknown"),
                                    "reason": ir.get("reason", ""),
                                }
                                for ir in analysis["voting_details"]["individual_results"]
                            ],
                            "all_different": analysis["voting_details"].get("all_different", False),
                        }

                    if "usage_stats" in analysis:
                        result["usage_stats"] = analysis["usage_stats"]
                        item_stats.total_calls = analysis["usage_stats"]["total_calls"]
                        item_stats.total_latency = analysis["usage_stats"]["total_latency_seconds"]
                        item_stats.total_prompt_tokens = analysis["usage_stats"]["total_prompt_tokens"]
                        item_stats.total_completion_tokens = analysis["usage_stats"]["total_completion_tokens"]
                        item_stats.total_tokens = analysis["usage_stats"]["total_tokens"]
                        item_stats.call_details = analysis["usage_stats"].get("call_details", [])
                else:
                    qa_data = QAData(
                        question=qa_item["qa_question"],
                        answer=qa_item["qa_answer"],
                        response=qa_item["qa_response"],
                    )
                    memory_data = MemoryData(
                        person1_memories=memories1,
                        person2_memories=memories2,
                        speaker1_retrieval=qa_item.get("speaker_1_memories", []),
                        speaker2_retrieval=qa_item.get("speaker_2_memories", []),
                    )

                    diagnosis = analyze_qa_pair(qa_data, memory_data, model=model)

                    result = {
                        "conv_id_question_id": item_id,
                        "qa_question": qa_item["qa_question"],
                        "qa_answer": qa_item["qa_answer"],
                        "qa_response": qa_item["qa_response"],
                        "qa_category": qa_item.get("qa_category", ""),
                        "label": diagnosis.label,
                        "reason": diagnosis.reason,
                        "stage": diagnosis.stage.value if isinstance(diagnosis.stage, DiagnosisStage) else diagnosis.stage,
                        "diagnosis_mode": f"single_model_{model}",
                    }

                    if diagnosis.usage_stats is not None:
                        result["usage_stats"] = diagnosis.usage_stats.to_dict()
                        item_stats.merge(diagnosis.usage_stats)

                return result, item_stats

            except Exception as e:
                logging.error(f"{prefix}Error while processing {item_id}: {str(e)}")
                _thread_print(f"{prefix}❌ {item_id} failed: {str(e)}\n")
                return None

        def save_processed_item(processed: Optional[Tuple[Dict, UsageStats]]) -> None:
            if processed is None:
                return

            result, item_stats = processed
            results.append(result)
            file_stats.merge(item_stats)

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            _thread_print(f"{prefix}✅ {result['conv_id_question_id']} completed and saved\n")

        effective_qa_threads = max(1, qa_threads)
        if effective_qa_threads <= 1 or len(pending_items) <= 1:
            for task in pending_items:
                save_processed_item(process_qa_item(task))
        else:
            effective_qa_threads = min(effective_qa_threads, len(pending_items))
            _thread_print(f"{prefix}🧵 Start {effective_qa_threads} QA threads inside this file...\n")
            with ThreadPoolExecutor(max_workers=effective_qa_threads) as executor:
                futures = [executor.submit(process_qa_item, task) for task in pending_items]
                for future in as_completed(futures):
                    save_processed_item(future.result())

    except KeyboardInterrupt:
        _thread_print(f"\n{prefix}⚠️  Processing interrupted, saving...\n")
    except Exception as e:
        logging.error(f"{prefix}Error occurred during processing: {str(e)}")
        _thread_print(f"\n{prefix}❌ Error occurred during processing: {str(e)}\n")
    finally:
        if results:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    _thread_print(f"{prefix}🎉 File processing completed: {len(results)} questions -> {output_file}")
    file_stats.print_summary()
    return len(results), file_stats


def _resolve_input_files(input_args: List[str]) -> List[str]:
    """Resolve input arguments (file paths, directory paths, and glob patterns).

    Args:
        input_args: list of input args

    Returns:
        De-duplicated list of JSON file paths
    """
    files = []
    for path in input_args:
        if os.path.isdir(path):
            files.extend(sorted(glob_module.glob(os.path.join(path, "*.json"))))
        elif os.path.isfile(path):
            files.append(path)
        else:
            expanded = sorted(glob_module.glob(path))
            if expanded:
                files.extend(expanded)
            else:
                logging.warning(f"Path does not exist or no files matched: {path}")
    seen = set()
    unique = []
    for f in files:
        real = os.path.realpath(f)
        if real not in seen:
            seen.add(real)
            unique.append(f)
    return unique


def _generate_output_path(
    input_file: str,
    model: str,
    use_voting: bool,
    num_votes: int,
    output_dir: Optional[str],
    timestamp: str,
) -> str:
    """Generate the output file path for a given input file."""
    input_basename = os.path.splitext(os.path.basename(input_file))[0]
    input_identifier = input_basename.replace(" ", "_").replace("(", "").replace(")", "")

    if output_dir is None:
        output_dir = "data/output/llm_annotation_voting" if use_voting else "data/output/llm_annotation_single"

    os.makedirs(output_dir, exist_ok=True)

    if use_voting:
        filename = f"{input_identifier}_voting_{num_votes}rounds_{model.replace('-', '_')}_{timestamp}.json"
    else:
        filename = f"{input_identifier}_single_{model.replace('-', '_')}_{timestamp}.json"

    return os.path.join(output_dir, filename)


def _print_stage_summary(global_stats: UsageStats):
    """Print per-stage aggregated statistics."""
    stage_stats: Dict[str, Dict] = {}
    for detail in global_stats.call_details:
        stage = detail.get("stage", "unknown")
        if stage not in stage_stats:
            stage_stats[stage] = {
                "calls": 0, "latency": 0.0,
                "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            }
        stage_stats[stage]["calls"] += 1
        stage_stats[stage]["latency"] += detail.get("latency_seconds", 0)
        stage_stats[stage]["prompt_tokens"] += detail.get("prompt_tokens", 0)
        stage_stats[stage]["completion_tokens"] += detail.get("completion_tokens", 0)
        stage_stats[stage]["total_tokens"] += detail.get("total_tokens", 0)

    if stage_stats:
        print(f"\n  📋 Per-stage summary:")
        for stage_name, s in sorted(stage_stats.items()):
            avg_lat = round(s["latency"] / s["calls"], 3) if s["calls"] > 0 else 0
            print(f"     {stage_name}: {s['calls']} calls, "
                  f"total latency {round(s['latency'], 3)}s (avg {avg_lat}s), "
                  f"tokens {s['total_tokens']}")


# ============================================================================
# Main entrypoint
# ============================================================================

def main():
    """Main entrypoint.

    Supports CLI usage:
        python run_diagnosis.py [model] [options]

    Arguments:
        model: model alias (deepseek, gpt4.1, gpt5), default: deepseek
        --voting: enable voting (default)
        --no-voting: disable voting and use a single model
        --num-votes N: voting rounds, default: 3
        -i, --input: input file/dir/glob paths (multiple supported)
        -o, --output-dir: output directory
        -f, --output-file: output filename (single-file mode only)
        -t, --threads: number of worker threads, default: 1
        --qa-threads: number of worker threads for QA items within each file

    Examples:
        python run_diagnosis.py deepseek --no-voting -i file.json
        python run_diagnosis.py deepseek -i data/input/mem0_mem/gpt4omini/ -t 5
        python run_diagnosis.py deepseek -i part1.json part2.json part3.json -t 3
        python run_diagnosis.py deepseek --num-votes 5 -i dir/ -t 5
    """
    import argparse
    import datetime

    parser = argparse.ArgumentParser(
        description="Memory diagnosis system - staged diagnosis for issues in QA pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    model_map = {
        "deepseek": "deepseek",
        "gpt4.1": "gpt-4.1",
        "gpt5": "gpt-5",
    }

    parser.add_argument(
        "model",
        nargs="?",
        default="deepseek",
        choices=list(model_map.keys()),
        help="Model to use (default: deepseek)",
    )
    parser.add_argument(
        "--voting",
        action="store_true",
        default=True,
        help="Enable voting mode (enabled by default)",
    )
    parser.add_argument(
        "--no-voting",
        action="store_true",
        help="Disable voting and use single-model diagnosis",
    )
    parser.add_argument(
        "--num-votes",
        type=int,
        default=3,
        help="Number of voting rounds (default: 3)",
    )
    parser.add_argument(
        "-i", "--input",
        nargs="+",
        default=["data/input/mem0_mem/gpt4omini/mem0_dataset_part1.json"],
        help="Input file path(s), directory path(s), or glob pattern(s) (supports multiple)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Output directory path (default: auto-selected by diagnosis mode)",
    )
    parser.add_argument(
        "-f", "--output-file",
        type=str,
        default=None,
        help="Output filename (single-file mode only)",
    )
    parser.add_argument(
        "-t", "--threads",
        type=int,
        default=1,
        help="Number of parallel threads (default: 1, recommended to match input file count)",
    )
    parser.add_argument(
        "--qa-threads",
        type=int,
        default=1,
        help="Number of QA worker threads inside each input file (default: 1)",
    )

    args = parser.parse_args()

    use_voting = args.voting and not args.no_voting
    model = model_map[args.model]
    num_threads = max(1, args.threads)
    qa_threads = max(1, args.qa_threads)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Resolve input files
    input_files = _resolve_input_files(args.input)
    if not input_files:
        print("❌ Error: No valid input files found")
        print(f"💡 Tip: Please check input paths {args.input}")
        return

    # Print startup info
    print("\n" + "=" * 70)
    print("🚀 Memory diagnosis system started")
    print("=" * 70)
    print(f"🤖 Model: {model}")
    print(f"📊 Diagnosis mode: {'Voting (' + str(args.num_votes) + ' rounds)' if use_voting else 'Single-model diagnosis'}")
    print(f"📁 Input files: {len(input_files)}")
    for f in input_files:
        print(f"   - {f}")
    print(f"🧵 File parallel threads: {num_threads}")
    print(f"🧵 QA parallel threads per file: {qa_threads}")
    print(f"⚙️  Config: {DiagnosisConfig()}")
    print("=" * 70 + "\n")

    # Generate output paths for each input file
    file_pairs: List[Tuple[str, str]] = []
    for idx, inp in enumerate(input_files):
        if len(input_files) == 1 and args.output_file:
            out_dir = args.output_dir or ("data/output/llm_annotation_voting" if use_voting else "data/output/llm_annotation_single")
            os.makedirs(out_dir, exist_ok=True)
            out = os.path.join(out_dir, args.output_file)
        else:
            out = _generate_output_path(inp, model, use_voting, args.num_votes, args.output_dir, timestamp)
        file_pairs.append((inp, out))
        print(f"📄 [{idx+1}] {inp}")
        print(f"   → {out}")
    print()

    # ---------------------------------------------------------------
    # Run diagnosis
    # ---------------------------------------------------------------
    global_stats = UsageStats()
    total_processed = 0

    if num_threads <= 1 or len(file_pairs) <= 1:
        # Single-threaded sequential processing
        for inp, out in file_pairs:
            count, stats = process_single_file(
                input_file=inp,
                output_file=out,
                model=model,
                use_voting=use_voting,
                num_votes=args.num_votes,
                qa_threads=qa_threads,
                thread_label=os.path.basename(inp),
            )
            total_processed += count
            global_stats.merge(stats)
    else:
        # Multi-threaded parallel processing
        effective_threads = min(num_threads, len(file_pairs))
        print(f"🧵 Start {effective_threads} threads to process {len(file_pairs)} files in parallel...\n")

        futures_map = {}
        with ThreadPoolExecutor(max_workers=effective_threads) as executor:
            for inp, out in file_pairs:
                future = executor.submit(
                    process_single_file,
                    input_file=inp,
                    output_file=out,
                    model=model,
                    use_voting=use_voting,
                    num_votes=args.num_votes,
                    qa_threads=qa_threads,
                    thread_label=os.path.basename(inp),
                )
                futures_map[future] = inp

            for future in as_completed(futures_map):
                inp = futures_map[future]
                try:
                    count, stats = future.result()
                    total_processed += count
                    global_stats.merge(stats)
                    print(f"✅ Thread completed: {os.path.basename(inp)} ({count} questions)")
                except Exception as e:
                    logging.error(f"Thread error while processing {inp}: {str(e)}")
                    print(f"❌ Thread failed: {os.path.basename(inp)}: {str(e)}")

    # ---------------------------------------------------------------
    # Global summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("🎉 All processing completed")
    print("=" * 70)
    print(f"✅ Processed {total_processed} questions in total ({len(file_pairs)} files)")
    for _, out in file_pairs:
        print(f"   📁 {out}")

    print(f"\n{'=' * 70}")
    print(f"📊 Global API call summary")
    print(f"{'=' * 70}")
    global_stats.print_summary()
    _print_stage_summary(global_stats)
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
