"""
记忆诊断系统 - 分阶段诊断QA对中的问题

该模块提供了一个分阶段的诊断框架，用于识别记忆系统中的问题类型：
- 阶段0: 一致性检查
- 阶段1: 记忆提取诊断
- 阶段2: 记忆更新诊断
- 阶段3: 记忆检索诊断
- 阶段4: 推理诊断
"""

# 标准库导入
import json
import logging
import os
import re
import sys
import time
import warnings
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

# 第三方库导入
from dotenv import load_dotenv
from requests.exceptions import RequestException, Timeout

# 注意：AI API 相关的导入已移到各自的函数中（延迟导入）
# 这样即使某些库未安装，也不会影响其他功能的使用

# ============================================================================
# 配置和初始化
# ============================================================================

# 抑制gRPC警告
logging.getLogger('grpc').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', module='grpc')

# 加载环境变量
load_dotenv()
os.environ['GRPC_ALTS_CREDENTIALS_ENVIRONMENT_OVERRIDE'] = '1'


# ============================================================================
# 枚举和常量定义
# ============================================================================

class ModelType(str, Enum):
    """支持的LLM模型类型"""
    QWEN = "qwen"
    DEEPSEEK = "deepseek"
    GPT_4_1 = "gpt-4.1"
    GPT_5 = "gpt-5"
    GEMINI = "gemini-2.5-pro"


class DiagnosisStage(str, Enum):
    """诊断阶段枚举"""
    CONSISTENCY_CHECK = "0_consistency_check"
    MEMORY_EXTRACTION = "1_memory_extraction"
    MEMORY_UPDATE = "2_memory_update"
    MEMORY_RETRIEVAL = "3_memory_retrieval"
    REASONING = "4_reasoning"
    ERROR = "error"


# ============================================================================
# 配置类
# ============================================================================

@dataclass
class APIConfig:
    """API配置"""
    dashscope_api_key: str = field(default_factory=lambda: os.getenv("DASHSCOPE_API_KEY", ""))
    deepseek_api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    deepseek_api_url: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_URL", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_url: str = field(default_factory=lambda: os.getenv("GEMINI_URL", ""))


@dataclass
class DiagnosisConfig:
    """诊断配置"""
    model: ModelType = ModelType.DEEPSEEK
    max_retries: int = 3
    retry_delay: int = 5
    temperature: float = 0.1
    timeout: int = 30


# ============================================================================
# 数据类
# ============================================================================

@dataclass
class QAData:
    """QA数据封装"""
    question: str
    answer: str
    response: str
    category: str = ""
    
    def to_json_str(self, field_name: str) -> str:
        """将指定字段转换为JSON字符串"""
        value = getattr(self, field_name.replace("qa_", ""))
        return json.dumps(value, ensure_ascii=False)


@dataclass
class MemoryData:
    """记忆数据封装"""
    person1_memories: List[dict] = field(default_factory=list)
    person2_memories: List[dict] = field(default_factory=list)
    speaker1_retrieval: List[Dict] = field(default_factory=list)
    speaker2_retrieval: List[Dict] = field(default_factory=list)
    
    def to_json_str(self, field_name: str, exclude_keys: Optional[List[str]] = None) -> str:
        """将指定字段转换为JSON字符串
        
        Args:
            field_name: 字段名称
            exclude_keys: 需要从每个记忆项中排除的键列表
            
        Returns:
            JSON字符串
        """
        value = getattr(self, field_name)
        
        # 如果需要排除某些键，则过滤数据
        if exclude_keys and isinstance(value, list):
            filtered_value = []
            for item in value:
                if isinstance(item, dict):
                    # 创建一个新字典，排除指定的键
                    filtered_item = {k: v for k, v in item.items() if k not in exclude_keys}
                    filtered_value.append(filtered_item)
                else:
                    filtered_value.append(item)
            return json.dumps(filtered_value, ensure_ascii=False)
        
        return json.dumps(value, ensure_ascii=False)


@dataclass
class StageResult:
    """阶段诊断结果"""
    stage_passed: bool
    label: Optional[str]
    reason: str
    stage: Optional[DiagnosisStage] = None


@dataclass
class DiagnosisResult:
    """完整诊断结果"""
    label: Optional[str]
    reason: str
    stage: DiagnosisStage
    used_model: Optional[str] = None
    voting_details: Optional[Dict] = None


# 初始化全局配置
API_CONFIG = APIConfig()
# ============================================================================
# 工具函数
# ============================================================================

def load_json_file(file_path: str) -> Dict:
    """加载JSON文件
    
    Args:
        file_path: JSON文件路径
        
    Returns:
        解析后的字典对象
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_prompt(prompt: str) -> str:
    """清理prompt中的特殊字符
    
    Args:
        prompt: 原始prompt文本
        
    Returns:
        清理后的prompt文本
    """
    return re.sub(r"[\u200b\u200c\u200d\ufeff\u202a-\u202e]", "", prompt)


def extract_json_from_response(response_text: str) -> Dict:
    """从响应文本中提取JSON对象
    
    Args:
        response_text: LLM响应文本
        
    Returns:
        解析后的JSON对象
        
    Raises:
        Exception: 解析失败时抛出异常
    """
    response_text = response_text.strip()
    start = response_text.find("{")
    end = response_text.rfind("}") + 1
    
    if start != -1 and end != 0:
        response_text = response_text[start:end]
    
    return json.loads(response_text)


# ============================================================================
# LLM API调用函数
# ============================================================================

def call_deepseek_api(prompt: str, temperature: float = 0.1) -> Dict:
    """调用DeepSeek API
    
    Args:
        prompt: 输入prompt
        temperature: 温度参数
        
    Returns:
        标准化的响应字典 {"output": {"text": "..."}}
        
    Raises:
        Exception: API调用失败时抛出异常
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise Exception("请安装 openai 库: pip install openai")
    
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
        return {"output": {"text": response.choices[0].message.content}}
    except Exception as e:
        # 打印详细的调试信息
        try:
            resp = getattr(e, "response", None)
            if resp is not None:
                logging.error(f"DeepSeek response status: {getattr(resp, 'status_code', None)}")
                logging.error(f"DeepSeek response body: {getattr(resp, 'text', None)}")
        except Exception:
            pass
        
        logging.error(f"DeepSeek API调用失败: {repr(e)}")
        
        # 如果是temperature参数问题，尝试不带temperature重试
        err_text = repr(e).lower()
        if "temperature" in err_text or "unsupported" in err_text:
            try:
                kwargs = {
                    "model": "deepseek-reasoner",
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                }
                response = client.chat.completions.create(**kwargs)
                return {"output": {"text": response.choices[0].message.content}}
            except Exception as e2:
                logging.error(f"Retry (no temperature) also failed: {repr(e2)}")
                raise Exception(f"DeepSeek API error: {str(e2)}")
        
        raise Exception(f"DeepSeek API error: {str(e)}")


def call_openai_api(prompt: str, model: str = "gpt-4.1", temperature: float = 0.1) -> Dict:
    """调用OpenAI API
    
    Args:
        prompt: 输入prompt
        model: 模型名称
        temperature: 温度参数
        
    Returns:
        标准化的响应字典 {"output": {"text": "..."}}
        
    Raises:
        Exception: API调用失败时抛出异常
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise Exception("请安装 openai 库: pip install openai")
    
    client = OpenAI(api_key=API_CONFIG.openai_api_key)
    
    # 某些模型不支持temperature参数
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
        return {"output": {"text": response.choices[0].message.content}}
    except Exception as e:
        logging.error(f"OpenAI API调用失败: {repr(e)}")
        
        # 如果是temperature参数问题，尝试不带temperature重试
        err_text = repr(e).lower()
        if "temperature" in err_text or "unsupported" in err_text:
            try:
                kwargs = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                }
                response = client.chat.completions.create(**kwargs)
                return {"output": {"text": response.choices[0].message.content}}
            except Exception as e2:
                logging.error(f"Retry (no temperature) also failed: {repr(e2)}")
                raise Exception(f"OpenAI API error: {str(e2)}")
        
        raise Exception(f"OpenAI API error: {str(e)}")


def call_gemini_api(prompt: str, model: str = "gemini-2.5-pro", temperature: float = 0.1) -> Dict:
    """调用Gemini API
    
    Args:
        prompt: 输入prompt
        model: 模型名称
        temperature: 温度参数
        
    Returns:
        标准化的响应字典 {"output": {"text": "..."}}
        
    Raises:
        Exception: API调用失败时抛出异常
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise Exception("请安装 openai 库: pip install openai")
    
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
        return {"output": {"text": response.choices[0].message.content}}
    except Exception as e:
        logging.error(f"Gemini API调用失败: {repr(e)}")
        
        # 如果是temperature参数问题，尝试不带temperature重试
        err_text = repr(e).lower()
        if "temperature" in err_text or "unsupported" in err_text:
            try:
                kwargs = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                }
                response = client.chat.completions.create(**kwargs)
                return {"output": {"text": response.choices[0].message.content}}
            except Exception as e2:
                logging.error(f"Retry (no temperature) also failed: {repr(e2)}")
                raise Exception(f"Gemini API error: {str(e2)}")
        
        raise Exception(f"Gemini API error: {str(e)}")

def call_llm_api(
    prompt: str,
    model: str = "deepseek",
    config: Optional[DiagnosisConfig] = None
) -> Dict:
    """调用LLM API的统一接口
    
    Args:
        prompt: 输入prompt文本
        model: 模型名称，可以是字符串或ModelType枚举值
        config: 诊断配置对象
        
    Returns:
        解析后的JSON响应
        
    Raises:
        Exception: API调用或解析失败时抛出异常
    """
    if config is None:
        config = DiagnosisConfig()
    
    # 清理prompt中的特殊字符
    prompt = clean_prompt(prompt)
    
    # 重试机制
    for attempt in range(config.max_retries):
        try:
            # 根据模型类型调用对应的API
            if model == ModelType.QWEN or model == "qwen":
                try:
                    import dashscope
                    from dashscope import Generation
                    # 设置API密钥（如果还没设置）
                    if API_CONFIG.dashscope_api_key and not dashscope.api_key:
                        dashscope.api_key = API_CONFIG.dashscope_api_key
                except ImportError:
                    raise Exception("请安装 dashscope 库: pip install dashscope")
                
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
                logging.warning(f"API调用失败，重试 {attempt + 1}/{config.max_retries}: {str(e)}")
                time.sleep(config.retry_delay)
                continue
            else:
                raise Exception(f"API调用失败（已重试{config.max_retries}次）: {str(e)}")
    
    # 解析响应
    try:
        if model == ModelType.QWEN or model == "qwen":
            response_text = response.output.text.strip()
        else:
            response_text = response["output"]["text"].strip()
        
        return extract_json_from_response(response_text)
    except Exception as e:
        raise Exception(f"解析响应失败: {str(e)}, 原始响应: {response_text[:200]}")


# ============================================================================
# 诊断阶段函数
# ============================================================================

def _print_stage_header(stage_name: str, stage_number: int = 0):
    """打印阶段标题"""
    print("=" * 60)
    print(f"阶段{stage_number}: {stage_name}")
    print("=" * 60)


def stage0_consistency_check(
    qa_data: QAData,
    model: str = "deepseek",
    config: Optional[DiagnosisConfig] = None
) -> StageResult:
    """阶段0：一致性检查
    
    检查模型回答是否与参考答案一致
    
    Args:
        qa_data: QA数据对象
        model: 使用的模型
        config: 诊断配置
        
    Returns:
        StageResult对象，包含诊断结果
    """
    _print_stage_header("一致性检查", 0)
    
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
        result = call_llm_api(prompt, model, config)
        is_consistent = result.get("is_consistent", False)
        
        stage_result = StageResult(
            stage_passed=is_consistent,
            label=None if is_consistent else "inconsistent",
            reason=result.get("reason", ""),
            stage=DiagnosisStage.CONSISTENCY_CHECK
        )
        
        print(f"✓ 一致性检查结果: {'通过' if is_consistent else '不通过'}")
        print(f"  原因: {stage_result.reason}\n")
        
        return stage_result
    except Exception as e:
        logging.error(f"阶段0错误: {str(e)}")
        return StageResult(
            stage_passed=False,
            label=None,
            reason=f"阶段0错误: {str(e)}",
            stage=DiagnosisStage.ERROR
        )


def stage1_memory_extraction(
    qa_data: QAData,
    memory_data: MemoryData,
    model: str = "deepseek",
    config: Optional[DiagnosisConfig] = None
) -> StageResult:
    """阶段1：记忆提取阶段
    
    检查初始记忆提取是否充分
    
    Args:
        qa_data: QA数据对象
        memory_data: 记忆数据对象
        model: 使用的模型
        config: 诊断配置
        
    Returns:
        StageResult对象，包含诊断结果
    """
    _print_stage_header("记忆提取阶段", 1)
    
    qa_question_str = qa_data.to_json_str("question")
    qa_answer_str = qa_data.to_json_str("answer")
    qa_response_str = qa_data.to_json_str("response")
    # 阶段1只看初始提取结果，同时保留time_stamp字段，方便进行时间相关判断
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
        result = call_llm_api(prompt, model, config)
        is_sufficient = result.get("is_sufficient", False)
        
        stage_result = StageResult(
            stage_passed=is_sufficient,
            label=None if is_sufficient else result.get("label"),
            reason=result.get("reason", ""),
            stage=DiagnosisStage.MEMORY_EXTRACTION
        )
        
        print(f"✓ 记忆提取结果: {'通过' if is_sufficient else '不通过'}")
        if not is_sufficient:
            print(f"  问题类型: {stage_result.label}")
        print(f"  原因: {stage_result.reason}\n")
        
        return stage_result
    except Exception as e:
        logging.error(f"阶段1错误: {str(e)}")
        return StageResult(
            stage_passed=False,
            label=None,
            reason=f"阶段1错误: {str(e)}",
            stage=DiagnosisStage.ERROR
        )


def stage2_memory_update(
    qa_data: QAData,
    memory_data: MemoryData,
    model: str = "deepseek",
    config: Optional[DiagnosisConfig] = None
) -> StageResult:
    """阶段2：记忆更新阶段
    
    检查记忆更新过程是否正确
    
    Args:
        qa_data: QA数据对象
        memory_data: 记忆数据对象
        model: 使用的模型
        config: 诊断配置
        
    Returns:
        StageResult对象，包含诊断结果
    """
    _print_stage_header("记忆更新阶段", 2)
    
    qa_question_str = qa_data.to_json_str("question")
    qa_answer_str = qa_data.to_json_str("answer")
    qa_response_str = qa_data.to_json_str("response")
    # 阶段2只看更新链，同时保留time_stamp字段，方便结合时间判断更新是否合理
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
        result = call_llm_api(prompt, model, config)
        is_sufficient = result.get("is_sufficient", False)
        
        stage_result = StageResult(
            stage_passed=is_sufficient,
            label=None if is_sufficient else result.get("label"),
            reason=result.get("reason", ""),
            stage=DiagnosisStage.MEMORY_UPDATE
        )
        
        print(f"✓ 记忆更新结果: {'通过' if is_sufficient else '不通过'}")
        if not is_sufficient:
            print(f"  问题类型: {stage_result.label}")
        print(f"  原因: {stage_result.reason}\n")
        
        return stage_result
    except Exception as e:
        logging.error(f"阶段2错误: {str(e)}")
        return StageResult(
            stage_passed=False,
            label=None,
            reason=f"阶段2错误: {str(e)}",
            stage=DiagnosisStage.ERROR
        )


def stage3_memory_retrieval(
    qa_data: QAData,
    memory_data: MemoryData,
    model: str = "deepseek",
    config: Optional[DiagnosisConfig] = None
) -> StageResult:
    """阶段3：记忆检索阶段
    
    检查记忆检索是否正确
    
    Args:
        qa_data: QA数据对象
        memory_data: 记忆数据对象
        model: 使用的模型
        config: 诊断配置
        
    Returns:
        StageResult对象，包含诊断结果
    """
    _print_stage_header("记忆检索阶段", 3)
    
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
        result = call_llm_api(prompt, model, config)
        is_sufficient = result.get("is_sufficient", False)
        
        stage_result = StageResult(
            stage_passed=is_sufficient,
            label=None if is_sufficient else result.get("label"),
            reason=result.get("reason", ""),
            stage=DiagnosisStage.MEMORY_RETRIEVAL
        )
        
        print(f"✓ 记忆检索结果: {'通过' if is_sufficient else '不通过'}")
        if not is_sufficient:
            print(f"  问题类型: {stage_result.label}")
        print(f"  原因: {stage_result.reason}\n")
        
        return stage_result
    except Exception as e:
        logging.error(f"阶段3错误: {str(e)}")
        return StageResult(
            stage_passed=False,
            label=None,
            reason=f"阶段3错误: {str(e)}",
            stage=DiagnosisStage.ERROR
        )


def stage4_reasoning(
    qa_data: QAData,
    memory_data: MemoryData,
    model: str = "deepseek",
    config: Optional[DiagnosisConfig] = None
) -> StageResult:
    """阶段4：推理阶段
    
    如果前面阶段都通过，问题出在推理环节
    
    Args:
        qa_data: QA数据对象
        memory_data: 记忆数据对象
        model: 使用的模型
        config: 诊断配置
        
    Returns:
        StageResult对象，包含诊断结果
    """
    _print_stage_header("推理阶段", 4)
    
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
        result = call_llm_api(prompt, model, config)
        
        stage_result = StageResult(
            stage_passed=False,
            label=result.get("label"),
            reason=result.get("reason", ""),
            stage=DiagnosisStage.REASONING
        )
        
        print(f"✓ 推理问题类型: {stage_result.label}")
        print(f"  原因: {stage_result.reason}\n")
        
        return stage_result
    except Exception as e:
        logging.error(f"阶段4错误: {str(e)}")
        return StageResult(
            stage_passed=False,
            label="4.2",
            reason=f"阶段4错误: {str(e)}",
            stage=DiagnosisStage.ERROR
        )


# ============================================================================
# 主诊断函数
# ============================================================================

def analyze_qa_pair(
    qa_data: QAData,
    memory_data: MemoryData,
    model: str = "deepseek",
    config: Optional[DiagnosisConfig] = None
) -> DiagnosisResult:
    """分阶段诊断系统主函数
    
    按顺序执行各个诊断阶段，直到发现问题或全部通过
    
    Args:
        qa_data: QA数据对象
        memory_data: 记忆数据对象
        model: 使用的模型
        config: 诊断配置
        
    Returns:
        DiagnosisResult对象，包含完整的诊断结果
    """
    print(f"\n{'='*70}")
    print(f"🔍 开始分阶段诊断")
    print(f"📝 问题: {qa_data.question}")
    print(f"{'='*70}\n")
    
    try:
        # 阶段0: 一致性检查
        stage0_result = stage0_consistency_check(qa_data, model, config)
        if stage0_result.stage_passed:
            return DiagnosisResult(
                label=None,
                reason=stage0_result.reason,
                stage=DiagnosisStage.CONSISTENCY_CHECK
            )
        
        # 阶段1: 记忆提取阶段
        stage1_result = stage1_memory_extraction(qa_data, memory_data, model, config)
        if not stage1_result.stage_passed:
            return DiagnosisResult(
                label=stage1_result.label,
                reason=stage1_result.reason,
                stage=DiagnosisStage.MEMORY_EXTRACTION
            )
        
        # 阶段2: 记忆更新阶段
        stage2_result = stage2_memory_update(qa_data, memory_data, model, config)
        if not stage2_result.stage_passed:
            return DiagnosisResult(
                label=stage2_result.label,
                reason=stage2_result.reason,
                stage=DiagnosisStage.MEMORY_UPDATE
            )
        
        # 阶段3: 记忆检索阶段
        stage3_result = stage3_memory_retrieval(qa_data, memory_data, model, config)
        if not stage3_result.stage_passed:
            return DiagnosisResult(
                label=stage3_result.label,
                reason=stage3_result.reason,
                stage=DiagnosisStage.MEMORY_RETRIEVAL
            )
        
        # 阶段4: 推理阶段（前面都通过了，问题在推理）
        stage4_result = stage4_reasoning(qa_data, memory_data, model, config)
        return DiagnosisResult(
            label=stage4_result.label,
            reason=stage4_result.reason,
            stage=DiagnosisStage.REASONING
        )
        
    except Exception as e:
        logging.error(f"诊断过程出错: {str(e)}")
        return DiagnosisResult(
            label=None,
            reason=f"诊断过程出错: {str(e)}",
            stage=DiagnosisStage.ERROR
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
    """兼容旧接口的分析函数
    
    该函数保持与原有代码的兼容性，将参数转换为新的数据类后调用新的分析函数
    
    Args:
        qa_question: 问题文本
        qa_answer: 参考答案
        qa_response: 模型回答
        memories1: person1的记忆数据
        memories2: person2的记忆数据
        speaker1_memories: speaker1的检索记忆
        speaker2_memories: speaker2的检索记忆
        model: 使用的模型
        
    Returns:
        诊断结果字典（兼容旧格式）
    """
    # 创建数据对象
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
    
    # 调用新函数
    result = analyze_qa_pair(qa_data, memory_data, model)
    
    # 转换为旧格式
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
    """使用投票机制分析QA对和检索记忆
    
    Args:
        qa_question: 问题文本
        qa_answer: 参考答案
        qa_response: 模型回答
        memories1: person1的记忆数据
        memories2: person2的记忆数据
        speaker1_memories: speaker1的检索记忆
        speaker2_memories: speaker2的检索记忆
        model: 主要使用的模型
        num_votes: 投票轮数
        
    Returns:
        包含最终诊断结果和投票详情的字典
    """
    print(f"\n🗳️  问题: {qa_question}")
    print(f"📊 使用 {model} 作为主模型进行 {num_votes} 轮投票（每轮使用不同模型）\n")
    
    # 创建数据对象（只需创建一次，可复用）
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
    
    # 存储每轮的结果
    vote_results = []
    
    # 定义模型列表，用于轮换
    models = ["deepseek", "gpt-4.1", "gpt-5"]
    
    # 确保主模型在列表中，如果不在则添加
    if model not in models:
        models.insert(0, model)
    else:
        # 将主模型移到首位
        models.remove(model)
        models.insert(0, model)
    
    # 进行多轮投票，确保每轮使用不同模型
    used_models = []
    for i in range(num_votes):
        # 选择模型：优先使用未使用过的模型
        current_model = None
        for m in models:
            if m not in used_models:
                current_model = m
                break
        
        # 如果所有模型都已使用过，则从除了主模型外的模型中选择
        if current_model is None:
            unused_models = [m for m in models if m != model]
            if unused_models:
                current_model = unused_models[len(used_models) % len(unused_models)]
            else:
                current_model = models[len(used_models) % len(models)]
        
        used_models.append(current_model)
        print(f"🔄 第 {i+1}/{num_votes} 轮分析，使用模型: {current_model}")
        
        try:
            # 使用新的数据类接口
            result = analyze_qa_pair(qa_data, memory_data, model=current_model)
            # 转换为字典格式并添加使用的模型信息
            result_dict = {
                "label": result.label,
                "reason": result.reason,
                "stage": result.stage.value if isinstance(result.stage, DiagnosisStage) else result.stage,
                "used_model": current_model
            }
            vote_results.append(result_dict)
            print(f"   ✅ 第 {i+1} 轮完成: label={result.label}, model={current_model}\n")
        except Exception as e:
            logging.error(f"第 {i+1} 轮分析失败: {str(e)}")
            print(f"   ❌ 第 {i+1} 轮分析失败: {str(e)}\n")
            # 如果某一轮失败，添加一个默认结果（标签为null）
            vote_results.append({
                "label": None,
                "reason": f"API调用失败: {str(e)}",
                "stage": "error",
                "used_model": current_model
            })
    
    # 统计投票结果（包含None标签）
    labels = [result["label"] for result in vote_results]
    
    # 选择出现次数最多的标签（包括None）
    label_counter = Counter(labels)
    most_common_label = label_counter.most_common(1)[0][0]

    
    # 获取最终结果的详细信息
    final_result = None
    
    # 检查是否所有投票结果都不同（即没有重复的标签）
    all_different = len(label_counter) == len(vote_results) and len(vote_results) > 1
    
    if all_different:
        # 如果所有投票结果都不同，则使用主模型的结果
        print(f"所有投票结果都不同，使用主模型 {model} 的结果")
        for result in vote_results:
            if result.get("used_model") == model:
                final_result = result
                most_common_label = result["label"]
                break
        
        # 如果没有找到主模型的结果，则使用第一个结果
        if final_result is None:
            print(f"未找到主模型 {model} 结果，使用第一个结果")
            final_result = vote_results[0]
            most_common_label = final_result["label"]
    else:
        # 根据得票数最多的标签来选择最终结果
        for result in vote_results:
            if result["label"] == most_common_label:
                final_result = result
                break
        
        # 如果没有找到匹配的结果（理论上不应该发生），使用第一个结果
        if final_result is None and vote_results:
            final_result = vote_results[0]
    
    # 简化 voting_details，只包含每轮的标注结果
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
        "all_different": all_different  # 添加标记，表示是否所有结果都不同
    }
    
    # 打印投票汇总
    print(f"{'='*70}")
    print(f"📊 投票汇总")
    print(f"{'='*70}")
    print(f"🤖 使用的模型顺序: {used_models}")
    
    # 安全地打印投票结果
    vote_count = label_counter[most_common_label] if not all_different else 1
    print(f"🏆 最终标签: {most_common_label} (得票数: {vote_count}/{num_votes})")
    if all_different:
        print(f"⚠️  所有投票结果都不同，已选择deepseek模型的结果")
    print(f"{'='*70}\n")
    
    return final_result
# ============================================================================
# 主程序入口
# ============================================================================

def main():
    """主程序入口函数
    
    支持命令行参数：
        python dignosis.py [model] [options]
        
    参数说明：
        model: 可选模型 (deepseek, gpt4.1, gpt5)，默认：deepseek
        --voting: 启用投票机制（默认）
        --no-voting: 禁用投票，使用单个模型
        --num-votes N: 投票轮数，默认：3
        -i, --input: 输入文件路径
        -o, --output-dir: 输出目录路径
        -f, --output-file: 输出文件名
        
    示例：
        python diagnosis.py deepseek --no-voting                    # 单模型诊断
        python diagnosis.py deepseek --voting                       # 投票诊断（3轮）
        python diagnosis.py deepseek --num-votes 5                  # 投票诊断（5轮）
        python diagnosis.py -i data/input.json -o results/         # 自定义输入输出
        python diagnosis.py --input data.json --output-file out.json # 指定文件
    """
    import argparse
    import datetime
    
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description="记忆诊断系统 - 分阶段诊断QA对中的问题",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # 支持的模型
    model_map = {
        "deepseek": "deepseek",
        "gpt4.1": "gpt-4.1",
        "gpt5": "gpt-5",
    }
    
    # 添加参数
    parser.add_argument(
        "model",
        nargs="?",
        default="deepseek",
        choices=list(model_map.keys()),
        help="选择使用的模型 (默认: deepseek)"
    )
    
    parser.add_argument(
        "--voting",
        action="store_true",
        default=True,
        help="启用投票机制（默认启用）"
    )
    
    parser.add_argument(
        "--no-voting",
        action="store_true",
        help="禁用投票，使用单个模型诊断"
    )
    
    parser.add_argument(
        "--num-votes",
        type=int,
        default=3,
        help="投票轮数 (默认: 3)"
    )
    
    parser.add_argument(
        "-i", "--input",
        type=str,
        default="data/input/mem0_mem/gpt4omini/mem0_dataset_part1.json",
        help="输入文件路径 (默认: data/input/mem0_mem/gpt4omini/mem0_dataset_part1.json)"
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="输出目录路径 (默认: 根据诊断模式自动选择)"
    )
    
    parser.add_argument(
        "-f", "--output-file",
        type=str,
        default=None,
        help="输出文件名 (默认: 根据诊断模式自动生成)"
    )
    
    # 解析参数
    args = parser.parse_args()
    
    # 确定是否使用投票
    use_voting = args.voting and not args.no_voting
    
    # 获取模型
    model = model_map[args.model]
    
    # 打印启动信息
    print("\n" + "="*70)
    print("🚀 记忆诊断系统启动")
    print("="*70)
    print(f"🤖 使用模型: {model}")
    print(f"📊 诊断模式: {'投票机制 (' + str(args.num_votes) + '轮)' if use_voting else '单模型诊断'}")
    print(f"⚙️  配置: {DiagnosisConfig()}")
    print("="*70 + "\n")
    
    # 设置输入输出文件路径
    input_file = args.input
    
    # 从输入文件名中提取标识（用于输出文件命名）
    input_basename = os.path.splitext(os.path.basename(input_file))[0]
    # 清理文件名中的特殊字符
    input_identifier = input_basename.replace(" ", "_").replace("(", "").replace(")", "")
    
    # 根据诊断模式选择输出目录和文件名
    if args.output_dir:
        # 用户指定了输出目录
        output_dir = args.output_dir
    else:
        # 自动选择输出目录
        if use_voting:
            output_dir = "data/output/llm_annotation_voting"
        else:
            output_dir = "data/output/llm_annotation_single"
    
    # 获取当前时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.output_file:
        # 用户指定了输出文件名
        output_filename = args.output_file
    else:
        # 自动生成输出文件名（包含输入文件标识和时间戳）
        if use_voting:
            output_filename = f"{input_identifier}_voting_{args.num_votes}rounds_{model.replace('-', '_')}_{timestamp}.json"
        else:
            output_filename = f"{input_identifier}_single_{model.replace('-', '_')}_{timestamp}.json"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 组合完整的输出文件路径
    output_file = os.path.join(output_dir, output_filename)
    
    print(f"📁 输入文件: {input_file}")
    print(f"📁 输出目录: {output_dir}")
    print(f"📁 输出文件: {output_file}\n")
    
    # 验证输入文件存在
    if not os.path.exists(input_file):
        print(f"❌ 错误: 输入文件不存在: {input_file}")
        print(f"💡 提示: 请检查文件路径或使用 -i 参数指定正确的输入文件")
        return
    
    # 加载输入数据
    try:
        data = load_json_file(input_file)
        print(f"✅ 成功加载 {len(data)} 个会话\n")
    except Exception as e:
        logging.error(f"加载输入文件失败: {str(e)}")
        print(f"❌ 错误: 无法解析输入文件: {str(e)}")
        return
    
    # 加载已处理的结果（支持断点续传）
    results = []
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                results = json.load(f)
                logging.info(f"已加载 {len(results)} 条历史结果")
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logging.warning(f"加载历史结果失败: {str(e)}，将从头开始")
            results = []
    
    processed_items = {item["conv_id_question_id"] for item in results}
    
    try:
        total_convs = len(data)
        print(f"📊 开始处理，共有 {total_convs} 个会话需要分析\n")
        
        for conv_idx, (conv_id, qa_list) in enumerate(data.items(), 1):
            print(f"\n{'='*70}")
            print(f"📝 处理会话 {conv_id} ({conv_idx}/{total_convs})")
            print(f"{'='*70}\n")
            
            for qa_idx, qa_item in enumerate(qa_list, 1):
                item_id = f"{conv_id}_{qa_idx-1}"
                
                # 检查是否已处理
                if item_id in processed_items:
                    print(f"⏭️  跳过已处理的问题: {item_id}\n")
                    continue
                
                print(f"🔍 开始处理问题 {qa_idx}/{len(qa_list)}: {item_id}")
                
                try:
                    # 提取数据
                    p1 = qa_item.get("person1", {})
                    p2 = qa_item.get("person2", {})
                    memories1 = p1.get("memories", [])
                    memories2 = p2.get("memories", [])
                    
                    # 根据配置选择诊断方式
                    if use_voting:
                        # 使用投票机制
                        analysis = analyze_qa_pair_with_voting(
                            qa_question=qa_item["qa_question"],
                            qa_answer=qa_item["qa_answer"],
                            qa_response=qa_item["qa_response"],
                            memories1=memories1,
                            memories2=memories2,
                            speaker1_memories=qa_item.get("speaker_1_memories", []),
                            speaker2_memories=qa_item.get("speaker_2_memories", []),
                            model=model,
                            num_votes=args.num_votes
                        )
                        
                        # 构建结果对象（投票模式）
                        result = {
                            "conv_id_question_id": item_id,
                            "qa_question": qa_item["qa_question"],
                            "qa_answer": qa_item["qa_answer"],
                            "qa_response": qa_item["qa_response"],
                            "qa_category": qa_item.get("qa_category", ""),
                            "label": analysis["label"],
                            "reason": analysis["reason"],
                            "diagnosis_mode": f"voting_{args.num_votes}rounds"
                        }
                        
                        # 添加投票详情
                        if "voting_details" in analysis:
                            result["voting_details"] = {
                                "label_votes": analysis["voting_details"]["label_votes"],
                                "individual_results": [
                                    {
                                        "label": ir["label"],
                                        "used_model": ir.get("used_model", "unknown"),
                                        "reason": ir.get("reason", "")
                                    }
                                    for ir in analysis["voting_details"]["individual_results"]
                                ],
                                "all_different": analysis["voting_details"].get("all_different", False)
                            }
                    else:
                        # 使用单模型诊断
                        qa_data = QAData(
                            question=qa_item["qa_question"],
                            answer=qa_item["qa_answer"],
                            response=qa_item["qa_response"]
                        )
                        
                        memory_data = MemoryData(
                            person1_memories=memories1,
                            person2_memories=memories2,
                            speaker1_retrieval=qa_item.get("speaker_1_memories", []),
                            speaker2_retrieval=qa_item.get("speaker_2_memories", [])
                        )
                        
                        diagnosis = analyze_qa_pair(qa_data, memory_data, model=model)
                        
                        # 构建结果对象（单模型模式）
                        result = {
                            "conv_id_question_id": item_id,
                            "qa_question": qa_item["qa_question"],
                            "qa_answer": qa_item["qa_answer"],
                            "qa_response": qa_item["qa_response"],
                            "qa_category": qa_item.get("qa_category", ""),
                            "label": diagnosis.label,
                            "reason": diagnosis.reason,
                            "stage": diagnosis.stage.value if isinstance(diagnosis.stage, DiagnosisStage) else diagnosis.stage,
                            "diagnosis_mode": f"single_model_{model}"
                        }
                    
                    results.append(result)
                    
                    # 立即保存到文件（支持断点续传）
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                    
                    print(f"✅ 问题 {item_id} 处理完成并已保存\n")
                    
                except Exception as e:
                    logging.error(f"处理问题 {item_id} 时发生错误: {str(e)}")
                    print(f"❌ 处理问题 {item_id} 失败: {str(e)}\n")
                    continue
                    
    except KeyboardInterrupt:
        print("\n⚠️  处理已中断，正在保存...\n")
    except Exception as e:
        logging.error(f"处理过程中发生未预期的错误: {str(e)}")
        print(f"\n❌ 处理过程中发生错误: {str(e)}\n")
    finally:
        # 确保最后结果被保存
        if results:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print("\n" + "="*70)
            print("🎉 处理完成")
            print("="*70)
            print(f"✅ 共处理 {len(results)} 个问题")
            print(f"📁 结果已保存到: {output_file}")
            print("="*70 + "\n")


if __name__ == "__main__":
    main()