"""
记忆诊断系统 - 多模型讨论版

该模块实现了一个多模型分阶段讨论框架：
- 阶段0: 一致性检查 - 三模型讨论
- 阶段1: 记忆提取诊断 - 三模型讨论
- 阶段2: 记忆更新诊断 - 三模型讨论
- 阶段3: 记忆检索诊断 - 三模型讨论
- 阶段4: 推理诊断 - 三模型讨论

每个阶段内，三个模型先独立判断，然后进行多轮讨论达成共识或投票决定
"""

# 标准库导入
import json
import logging
import os
import re
import time
import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import datetime
import argparse

# 第三方库导入
from dotenv import load_dotenv
from requests.exceptions import RequestException, Timeout

# ============================================================================
# 配置和初始化
# ============================================================================

logging.getLogger('grpc').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', module='grpc')

load_dotenv()
os.environ['GRPC_ALTS_CREDENTIALS_ENVIRONMENT_OVERRIDE'] = '1'

# 导入原有模块的类和函数
from run_diagnosis import (
    APIConfig, DiagnosisConfig, QAData, MemoryData, DiagnosisStage,
    StageResult, DiagnosisResult, API_CONFIG,
    load_json_file, clean_prompt, extract_json_from_response,
    call_llm_api
)


# ============================================================================
# 讨论相关的数据类
# ============================================================================

@dataclass
class StageOpinion:
    """单个模型在某阶段的意见"""
    model_name: str
    stage_passed: bool
    label: Optional[str]
    reason: str
    round_num: int
    changed_from_passed: Optional[bool] = None
    changed_from_label: Optional[str] = None


@dataclass
class StageDiscussionResult:
    """某阶段讨论的最终结果"""
    stage: DiagnosisStage
    consensus_reached: bool
    final_passed: bool
    final_label: Optional[str]
    final_reason: str
    total_rounds: int
    discussion_history: List[Dict]


@dataclass
class FullDiscussionResult:
    """完整诊断讨论结果"""
    final_label: Optional[str]
    final_reason: str
    final_stage: DiagnosisStage
    stage_results: Dict[str, StageDiscussionResult]


# ============================================================================
# 阶段讨论Prompt生成函数
# ============================================================================

def generate_stage0_prompt(qa_data: QAData, other_opinions: List[StageOpinion] = None) -> str:
    """生成阶段0的prompt（一致性检查）"""
    qa_question_str = json.dumps(qa_data.question, ensure_ascii=False)
    qa_answer_str = json.dumps(qa_data.answer, ensure_ascii=False)
    qa_response_str = json.dumps(qa_data.response, ensure_ascii=False)
    
    base_prompt = f"""
You are an evaluation assistant. Determine whether qa_response is semantically consistent with qa_answer.

Consistency rules:
- All key information in qa_answer must appear in qa_response.
- Missing, incorrect or unclear details make it inconsistent.

Example:
qa_answer: "first weekend of August 2023"
qa_response: "5 August 2023."
→ inconsistent (incorrectly narrows the time range)

Now evaluate:
input:
- qa_question: {qa_question_str}
- qa_answer: {qa_answer_str}
- qa_response: {qa_response_str}
"""
    
    if other_opinions:
        opinions_text = "\n=== Other Models' Opinions ===\n"
        for op in other_opinions:
            opinions_text += f"""
Model ({op.model_name}):
- is_consistent: {op.stage_passed}
- reason: {op.reason}
"""
        base_prompt += opinions_text
        base_prompt += """
Consider the other models' opinions. You may KEEP or CHANGE your judgment.
"""
    
    base_prompt += """
Output:
{
  "is_consistent": true/false,
  "reason": "brief explanation"
}
"""
    return base_prompt


def generate_stage1_prompt(
    qa_data: QAData, 
    memory_data: MemoryData, 
    other_opinions: List[StageOpinion] = None
) -> str:
    """生成阶段1的prompt（记忆提取）"""
    qa_question_str = json.dumps(qa_data.question, ensure_ascii=False)
    qa_answer_str = json.dumps(qa_data.answer, ensure_ascii=False)
    qa_response_str = json.dumps(qa_data.response, ensure_ascii=False)
    
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
    
    base_prompt = f"""
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

Now evaluate the following:

Input:
- qa_question: {qa_question_str}
- qa_answer: {qa_answer_str}
- qa_response: {qa_response_str}
- person1_memories: {memories1_str}
- person2_memories: {memories2_str}
"""
    
    if other_opinions:
        opinions_text = "\n=== Other Models' Opinions ===\n"
        for op in other_opinions:
            opinions_text += f"""
Model ({op.model_name}):
- is_sufficient: {op.stage_passed}
- label: {op.label}
- reason: {op.reason}
"""
        base_prompt += opinions_text
        base_prompt += """
Consider the other models' opinions carefully. You may KEEP or CHANGE your judgment.
"""
    
    base_prompt += """
Output format:
{
  "is_sufficient": true/false,
  "label": "1.1" or "1.2" or "1.3" or null,
  "reason": "Detailed explanation"
}
"""
    return base_prompt


def generate_stage2_prompt(
    qa_data: QAData, 
    memory_data: MemoryData, 
    other_opinions: List[StageOpinion] = None
) -> str:
    """生成阶段2的prompt（记忆更新）"""
    qa_question_str = json.dumps(qa_data.question, ensure_ascii=False)
    qa_answer_str = json.dumps(qa_data.answer, ensure_ascii=False)
    qa_response_str = json.dumps(qa_data.response, ensure_ascii=False)
    
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
    
    base_prompt = f"""
You are an evaluation assistant for the Memory Update Stage.
Task:
1. From the update_chain, use only the final updated memory for each item.
2. Determine whether the updated memories contain sufficient and correct information to answer qa_question.
3. If sufficient → is_sufficient = true (label = null).
4. If insufficient, classify the issue according to the update error type:
   - "2.1": Incorrect update (added wrong or fabricated details)
   - "2.2": Deleted information (removed necessary memory entries)
   - "2.3": Weakened information (kept but diluted or less specific)

Now evaluate the following:

Input:
- qa_question: {qa_question_str}
- qa_answer: {qa_answer_str}
- qa_response: {qa_response_str}
- person1_memories: {memories1_str}
- person2_memories: {memories2_str}
"""
    
    if other_opinions:
        opinions_text = "\n=== Other Models' Opinions ===\n"
        for op in other_opinions:
            opinions_text += f"""
Model ({op.model_name}):
- is_sufficient: {op.stage_passed}
- label: {op.label}
- reason: {op.reason}
"""
        base_prompt += opinions_text
        base_prompt += """
Consider the other models' opinions carefully. You may KEEP or CHANGE your judgment.
"""
    
    base_prompt += """
Output format:
{
  "is_sufficient": true/false,
  "label": "2.1" or "2.2" or "2.3" or null,
  "reason": "Detailed explanation"
}
"""
    return base_prompt


def generate_stage3_prompt(
    qa_data: QAData, 
    memory_data: MemoryData, 
    other_opinions: List[StageOpinion] = None
) -> str:
    """生成阶段3的prompt（记忆检索）"""
    qa_question_str = json.dumps(qa_data.question, ensure_ascii=False)
    qa_answer_str = json.dumps(qa_data.answer, ensure_ascii=False)
    speaker1_memories_str = json.dumps(memory_data.speaker1_retrieval, ensure_ascii=False)
    speaker2_memories_str = json.dumps(memory_data.speaker2_retrieval, ensure_ascii=False)
    
    base_prompt = f"""
You are an evaluation assistant for the Memory Retrieval Stage.
Task:
Based strictly on speaker1_retrieval and speaker2_retrieval:
1. Determine whether the retrieved memories contain enough correct information to answer qa_question.
2. If sufficient → is_sufficient = true (label = null).
3. If insufficient, determine the retrieval issue:
   - "3.1": Failed to recall correct information (missing the key facts)
   - "3.2": Unreasonable ranking (recalled irrelevant/common info while missing the most relevant facts)

Now evaluate the following:

Input:
- qa_question: {qa_question_str}
- qa_answer: {qa_answer_str}
- speaker1_retrieval: {speaker1_memories_str}
- speaker2_retrieval: {speaker2_memories_str}
"""
    
    if other_opinions:
        opinions_text = "\n=== Other Models' Opinions ===\n"
        for op in other_opinions:
            opinions_text += f"""
Model ({op.model_name}):
- is_sufficient: {op.stage_passed}
- label: {op.label}
- reason: {op.reason}
"""
        base_prompt += opinions_text
        base_prompt += """
Consider the other models' opinions carefully. You may KEEP or CHANGE your judgment.
"""
    
    base_prompt += """
Output format:
{
  "is_sufficient": true/false,
  "label": "3.1" or "3.2" or null,
  "reason": "Detailed explanation"
}
"""
    return base_prompt


def generate_stage4_prompt(
    qa_data: QAData, 
    memory_data: MemoryData, 
    other_opinions: List[StageOpinion] = None
) -> str:
    """生成阶段4的prompt（推理）"""
    qa_question_str = json.dumps(qa_data.question, ensure_ascii=False)
    qa_answer_str = json.dumps(qa_data.answer, ensure_ascii=False)
    qa_response_str = json.dumps(qa_data.response, ensure_ascii=False)
    speaker1_memories_str = json.dumps(memory_data.speaker1_retrieval, ensure_ascii=False)
    speaker2_memories_str = json.dumps(memory_data.speaker2_retrieval, ensure_ascii=False)
    
    base_prompt = f"""
You are an evaluation assistant for the Reasoning Stage.

Context:
All previous stages (extraction, update, retrieval) have passed, meaning the model had sufficient correct information.  
If qa_response still does not match qa_answer, the issue is a reasoning error.

Task:
Based on qa_question, qa_answer, qa_response, and the retrieved memories, classify the reasoning issue:
- "4.1": Correct memory entries were ignored (model overlooks correct memory entries present in retrieval)
- "4.2": Reasoning error (model invents details, over-specifies, or makes unsupported inferences)
- "4.3": Format or detail error (minor deviations such as missing qualifiers or altered phrasing that slightly change meaning)

Now evaluate the following:

Input:
- qa_question: {qa_question_str}
- qa_answer: {qa_answer_str}
- qa_response: {qa_response_str}
- speaker1_retrieval: {speaker1_memories_str}
- speaker2_retrieval: {speaker2_memories_str}
"""
    
    if other_opinions:
        opinions_text = "\n=== Other Models' Opinions ===\n"
        for op in other_opinions:
            opinions_text += f"""
Model ({op.model_name}):
- label: {op.label}
- reason: {op.reason}
"""
        base_prompt += opinions_text
        base_prompt += """
Consider the other models' opinions carefully. You may KEEP or CHANGE your judgment.
"""
    
    base_prompt += """
Output format:
{
  "label": "4.1" or "4.2" or "4.3",
  "reason": "Detailed explanation"
}
"""
    return base_prompt


# ============================================================================
# 单阶段讨论函数
# ============================================================================

def discuss_stage(
    stage: DiagnosisStage,
    qa_data: QAData,
    memory_data: MemoryData,
    models: List[str],
    max_rounds: int = 3,
    config: Optional[DiagnosisConfig] = None
) -> StageDiscussionResult:
    """在某个阶段进行多模型讨论
    
    Args:
        stage: 当前诊断阶段
        qa_data: QA数据
        memory_data: 记忆数据
        models: 参与讨论的模型列表
        max_rounds: 最大讨论轮次
        config: 诊断配置
        
    Returns:
        StageDiscussionResult对象
    """
    stage_name_map = {
        DiagnosisStage.CONSISTENCY_CHECK: "一致性检查",
        DiagnosisStage.MEMORY_EXTRACTION: "记忆提取",
        DiagnosisStage.MEMORY_UPDATE: "记忆更新",
        DiagnosisStage.MEMORY_RETRIEVAL: "记忆检索",
        DiagnosisStage.REASONING: "推理",
    }
    
    print(f"\n{'='*60}")
    print(f"📋 阶段: {stage.value} - {stage_name_map.get(stage, stage.value)}")
    print(f"{'='*60}")
    
    # 选择对应阶段的prompt生成函数
    prompt_generators = {
        DiagnosisStage.CONSISTENCY_CHECK: lambda ops: generate_stage0_prompt(qa_data, ops),
        DiagnosisStage.MEMORY_EXTRACTION: lambda ops: generate_stage1_prompt(qa_data, memory_data, ops),
        DiagnosisStage.MEMORY_UPDATE: lambda ops: generate_stage2_prompt(qa_data, memory_data, ops),
        DiagnosisStage.MEMORY_RETRIEVAL: lambda ops: generate_stage3_prompt(qa_data, memory_data, ops),
        DiagnosisStage.REASONING: lambda ops: generate_stage4_prompt(qa_data, memory_data, ops),
    }
    
    generate_prompt = prompt_generators[stage]
    discussion_history = []
    current_opinions = []
    
    # 第一轮：独立判断
    print(f"\n🔄 第 1 轮：独立判断")
    
    for model in models:
        print(f"   🤖 {model} 正在分析...")
        prompt = generate_prompt(None)  # 第一轮没有其他意见
        
        try:
            result = call_llm_api(clean_prompt(prompt), model, config)
            
            # 根据阶段解析结果
            if stage == DiagnosisStage.CONSISTENCY_CHECK:
                stage_passed = result.get("is_consistent", False)
                label = None
            elif stage == DiagnosisStage.REASONING:
                stage_passed = False  # 阶段4总是返回一个label
                label = result.get("label")
            else:
                stage_passed = result.get("is_sufficient", False)
                label = None if stage_passed else result.get("label")
            
            reason = result.get("reason", "")
            
            opinion = StageOpinion(
                model_name=model,
                stage_passed=stage_passed,
                label=label,
                reason=reason,
                round_num=1
            )
            current_opinions.append(opinion)
            
            print(f"      ✓ {model}: passed={stage_passed}, label={label}")
            
        except Exception as e:
            logging.error(f"模型 {model} 阶段 {stage.value} 分析失败: {str(e)}")
            opinion = StageOpinion(
                model_name=model,
                stage_passed=False,
                label=None,
                reason=f"分析失败: {str(e)}",
                round_num=1
            )
            current_opinions.append(opinion)
    
    # 记录第一轮历史
    discussion_history.append({
        "round": 1,
        "opinions": [
            {"model": op.model_name, "passed": op.stage_passed, "label": op.label, "reason": op.reason}
            for op in current_opinions
        ]
    })
    
    # 检查是否达成共识
    def check_consensus(opinions: List[StageOpinion]) -> Tuple[bool, Optional[bool], Optional[str]]:
        """检查是否达成共识，返回 (是否共识, 共识的passed值, 共识的label)"""
        if stage == DiagnosisStage.REASONING:
            # 阶段4只看label
            labels = [op.label for op in opinions]
            if len(set(labels)) == 1:
                return True, False, labels[0]
            return False, None, None
        else:
            # 其他阶段先看passed，如果都不通过再看label
            passed_values = [op.stage_passed for op in opinions]
            if len(set(passed_values)) == 1:
                if passed_values[0]:  # 都通过
                    return True, True, None
                else:  # 都不通过，检查label
                    labels = [op.label for op in opinions]
                    if len(set(labels)) == 1:
                        return True, False, labels[0]
            return False, None, None
    
    consensus, consensus_passed, consensus_label = check_consensus(current_opinions)
    
    if consensus:
        print(f"\n🎉 第 1 轮即达成共识！passed={consensus_passed}, label={consensus_label}")
        return StageDiscussionResult(
            stage=stage,
            consensus_reached=True,
            final_passed=consensus_passed,
            final_label=consensus_label,
            final_reason=current_opinions[0].reason,
            total_rounds=1,
            discussion_history=discussion_history
        )
    
    # 后续轮次：讨论
    for round_num in range(2, max_rounds + 1):
        print(f"\n🔄 第 {round_num} 轮：讨论")
        
        new_opinions = []
        
        for model in models:
            # 获取其他模型的意见
            other_opinions = [op for op in current_opinions if op.model_name != model]
            current_model_opinion = next(op for op in current_opinions if op.model_name == model)
            
            print(f"   🤖 {model} 正在参考其他意见...")
            prompt = generate_prompt(other_opinions)
            
            try:
                result = call_llm_api(clean_prompt(prompt), model, config)
                
                # 根据阶段解析结果
                if stage == DiagnosisStage.CONSISTENCY_CHECK:
                    stage_passed = result.get("is_consistent", False)
                    label = None
                elif stage == DiagnosisStage.REASONING:
                    stage_passed = False
                    label = result.get("label")
                else:
                    stage_passed = result.get("is_sufficient", False)
                    label = None if stage_passed else result.get("label")
                
                reason = result.get("reason", "")
                
                # 记录是否改变了意见
                changed_from_passed = None
                changed_from_label = None
                if stage_passed != current_model_opinion.stage_passed:
                    changed_from_passed = current_model_opinion.stage_passed
                    print(f"      ↪️ {model} 修改了判断: passed {current_model_opinion.stage_passed} → {stage_passed}")
                elif label != current_model_opinion.label:
                    changed_from_label = current_model_opinion.label
                    print(f"      ↪️ {model} 修改了标签: {current_model_opinion.label} → {label}")
                else:
                    print(f"      ✓ {model} 保持判断: passed={stage_passed}, label={label}")
                
                opinion = StageOpinion(
                    model_name=model,
                    stage_passed=stage_passed,
                    label=label,
                    reason=reason,
                    round_num=round_num,
                    changed_from_passed=changed_from_passed,
                    changed_from_label=changed_from_label
                )
                new_opinions.append(opinion)
                
            except Exception as e:
                logging.error(f"模型 {model} 讨论失败: {str(e)}")
                # 保持原意见
                opinion = StageOpinion(
                    model_name=model,
                    stage_passed=current_model_opinion.stage_passed,
                    label=current_model_opinion.label,
                    reason=f"讨论失败，保持原意见: {str(e)}",
                    round_num=round_num
                )
                new_opinions.append(opinion)
        
        current_opinions = new_opinions
        
        # 记录本轮历史
        discussion_history.append({
            "round": round_num,
            "opinions": [
                {
                    "model": op.model_name, 
                    "passed": op.stage_passed, 
                    "label": op.label, 
                    "reason": op.reason,
                    "changed_from_passed": op.changed_from_passed,
                    "changed_from_label": op.changed_from_label
                }
                for op in current_opinions
            ]
        })
        
        # 检查共识
        consensus, consensus_passed, consensus_label = check_consensus(current_opinions)
        
        if consensus:
            print(f"\n🎉 第 {round_num} 轮达成共识！passed={consensus_passed}, label={consensus_label}")
            return StageDiscussionResult(
                stage=stage,
                consensus_reached=True,
                final_passed=consensus_passed,
                final_label=consensus_label,
                final_reason=current_opinions[0].reason,
                total_rounds=round_num,
                discussion_history=discussion_history
            )
    
    # 未达成共识，投票决定
    print(f"\n⚠️ {max_rounds} 轮后未达成共识，进行投票")
    
    # 辅助函数：检查是否所有结果都不同，如果是则使用 gpt-5 的结果
    def get_gpt5_opinion(opinions: List[StageOpinion]) -> Optional[StageOpinion]:
        """获取 gpt-5 的意见"""
        for op in opinions:
            if op.model_name == "gpt-5":
                return op
        return None
    
    if stage == DiagnosisStage.REASONING:
        # 阶段4只投票label
        labels = [op.label for op in current_opinions]
        label_counter = Counter(labels)
        
        # 检查是否所有结果都不同（1:1:1）
        all_different = len(label_counter) == len(current_opinions) and len(current_opinions) > 1
        
        if all_different:
            # 所有结果都不同，使用 gpt-5 的结果
            gpt5_op = get_gpt5_opinion(current_opinions)
            if gpt5_op:
                final_label = gpt5_op.label
                print(f"📊 投票结果: {dict(label_counter)}")
                print(f"⚠️ 所有结果都不同，使用 gpt-5 的结果")
                print(f"🏆 选择标签: {final_label}")
            else:
                final_label = label_counter.most_common(1)[0][0]
                print(f"📊 投票结果: {dict(label_counter)}")
                print(f"🏆 选择标签: {final_label}")
        else:
            final_label = label_counter.most_common(1)[0][0]
            print(f"📊 投票结果: {dict(label_counter)}")
            print(f"🏆 选择标签: {final_label} (得票最多)")
        final_passed = False
    else:
        # 先投票passed
        passed_values = [op.stage_passed for op in current_opinions]
        passed_counter = Counter(passed_values)
        final_passed = passed_counter.most_common(1)[0][0]
        
        if final_passed:
            final_label = None
            print(f"📊 投票结果: passed={dict(passed_counter)}")
            print(f"🏆 阶段通过")
        else:
            # 不通过时投票label
            labels = [op.label for op in current_opinions if not op.stage_passed]
            if labels:
                label_counter = Counter(labels)
                
                # 检查是否所有结果都不同（1:1:1）
                all_different = len(label_counter) == len(labels) and len(labels) > 1
                
                if all_different:
                    # 所有结果都不同，使用 gpt-5 的结果
                    gpt5_op = get_gpt5_opinion([op for op in current_opinions if not op.stage_passed])
                    if gpt5_op:
                        final_label = gpt5_op.label
                        print(f"📊 投票结果: passed={dict(passed_counter)}, labels={dict(label_counter)}")
                        print(f"⚠️ 所有结果都不同，使用 gpt-5 的结果")
                        print(f"🏆 选择标签: {final_label}")
                    else:
                        final_label = label_counter.most_common(1)[0][0]
                        print(f"📊 投票结果: passed={dict(passed_counter)}, labels={dict(label_counter)}")
                        print(f"🏆 选择标签: {final_label}")
                else:
                    final_label = label_counter.most_common(1)[0][0]
                    print(f"📊 投票结果: passed={dict(passed_counter)}, labels={dict(label_counter)}")
                    print(f"🏆 选择标签: {final_label} (得票最多)")
            else:
                final_label = None
    
    # 获取对应意见的reason
    final_opinion = None
    for op in current_opinions:
        if stage == DiagnosisStage.REASONING:
            if op.label == final_label:
                final_opinion = op
                break
        else:
            if op.stage_passed == final_passed and op.label == final_label:
                final_opinion = op
                break
    
    return StageDiscussionResult(
        stage=stage,
        consensus_reached=False,
        final_passed=final_passed,
        final_label=final_label,
        final_reason=final_opinion.reason if final_opinion else "",
        total_rounds=max_rounds,
        discussion_history=discussion_history
    )


# ============================================================================
# 完整诊断讨论函数
# ============================================================================

def analyze_qa_pair_with_discussion(
    qa_question: str,
    qa_answer: str,
    qa_response: str,
    memories1: List[dict],
    memories2: List[dict],
    speaker1_memories: List[Dict],
    speaker2_memories: List[Dict],
    models: List[str] = None,
    max_rounds: int = 3,
    config: Optional[DiagnosisConfig] = None
) -> Dict:
    """使用多模型分阶段讨论机制分析QA对
    
    按顺序执行各阶段讨论，每个阶段三模型先独立判断再讨论达成共识
    
    Args:
        qa_question: 问题文本
        qa_answer: 参考答案
        qa_response: 模型回答
        memories1: person1的记忆数据
        memories2: person2的记忆数据
        speaker1_memories: speaker1的检索记忆
        speaker2_memories: speaker2的检索记忆
        models: 参与讨论的模型列表
        max_rounds: 每个阶段最大讨论轮次
        config: 诊断配置
        
    Returns:
        包含讨论结果的字典
    """
    if models is None:
        models = ["deepseek", "gpt-4.1", "gpt-5"]
    
    print(f"\n{'='*70}")
    print(f"🗣️  多模型分阶段讨论诊断")
    print(f"{'='*70}")
    print(f"📝 问题: {qa_question}")
    print(f"🤖 参与模型: {', '.join(models)}")
    print(f"🔄 每阶段最大轮次: {max_rounds}")
    print(f"{'='*70}")
    
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
    
    stage_results = {}
    
    # ========== 阶段0：一致性检查 ==========
    stage0_result = discuss_stage(
        DiagnosisStage.CONSISTENCY_CHECK,
        qa_data, memory_data, models, max_rounds, config
    )
    stage_results["0_consistency_check"] = stage0_result
    
    if stage0_result.final_passed:
        # 一致，直接返回
        print(f"\n{'='*70}")
        print(f"✅ 诊断完成：回答与答案一致")
        print(f"{'='*70}\n")
        
        return {
            "label": None,
            "reason": stage0_result.final_reason,
            "stage": DiagnosisStage.CONSISTENCY_CHECK.value,
            "consensus_reached": stage0_result.consensus_reached,
            "total_stage_rounds": {"0_consistency_check": stage0_result.total_rounds},
            "stage_results": _serialize_stage_results(stage_results)
        }
    
    # ========== 阶段1：记忆提取 ==========
    stage1_result = discuss_stage(
        DiagnosisStage.MEMORY_EXTRACTION,
        qa_data, memory_data, models, max_rounds, config
    )
    stage_results["1_memory_extraction"] = stage1_result
    
    if not stage1_result.final_passed:
        print(f"\n{'='*70}")
        print(f"❌ 诊断完成：记忆提取阶段发现问题")
        print(f"   标签: {stage1_result.final_label}")
        print(f"{'='*70}\n")
        
        return {
            "label": stage1_result.final_label,
            "reason": stage1_result.final_reason,
            "stage": DiagnosisStage.MEMORY_EXTRACTION.value,
            "consensus_reached": stage1_result.consensus_reached,
            "total_stage_rounds": {
                "0_consistency_check": stage0_result.total_rounds,
                "1_memory_extraction": stage1_result.total_rounds
            },
            "stage_results": _serialize_stage_results(stage_results)
        }
    
    # ========== 阶段2：记忆更新 ==========
    stage2_result = discuss_stage(
        DiagnosisStage.MEMORY_UPDATE,
        qa_data, memory_data, models, max_rounds, config
    )
    stage_results["2_memory_update"] = stage2_result
    
    if not stage2_result.final_passed:
        print(f"\n{'='*70}")
        print(f"❌ 诊断完成：记忆更新阶段发现问题")
        print(f"   标签: {stage2_result.final_label}")
        print(f"{'='*70}\n")
        
        return {
            "label": stage2_result.final_label,
            "reason": stage2_result.final_reason,
            "stage": DiagnosisStage.MEMORY_UPDATE.value,
            "consensus_reached": stage2_result.consensus_reached,
            "total_stage_rounds": {
                "0_consistency_check": stage0_result.total_rounds,
                "1_memory_extraction": stage1_result.total_rounds,
                "2_memory_update": stage2_result.total_rounds
            },
            "stage_results": _serialize_stage_results(stage_results)
        }
    
    # ========== 阶段3：记忆检索 ==========
    stage3_result = discuss_stage(
        DiagnosisStage.MEMORY_RETRIEVAL,
        qa_data, memory_data, models, max_rounds, config
    )
    stage_results["3_memory_retrieval"] = stage3_result
    
    if not stage3_result.final_passed:
        print(f"\n{'='*70}")
        print(f"❌ 诊断完成：记忆检索阶段发现问题")
        print(f"   标签: {stage3_result.final_label}")
        print(f"{'='*70}\n")
        
        return {
            "label": stage3_result.final_label,
            "reason": stage3_result.final_reason,
            "stage": DiagnosisStage.MEMORY_RETRIEVAL.value,
            "consensus_reached": stage3_result.consensus_reached,
            "total_stage_rounds": {
                "0_consistency_check": stage0_result.total_rounds,
                "1_memory_extraction": stage1_result.total_rounds,
                "2_memory_update": stage2_result.total_rounds,
                "3_memory_retrieval": stage3_result.total_rounds
            },
            "stage_results": _serialize_stage_results(stage_results)
        }
    
    # ========== 阶段4：推理 ==========
    stage4_result = discuss_stage(
        DiagnosisStage.REASONING,
        qa_data, memory_data, models, max_rounds, config
    )
    stage_results["4_reasoning"] = stage4_result
    
    print(f"\n{'='*70}")
    print(f"❌ 诊断完成：推理阶段发现问题")
    print(f"   标签: {stage4_result.final_label}")
    print(f"{'='*70}\n")
    
    return {
        "label": stage4_result.final_label,
        "reason": stage4_result.final_reason,
        "stage": DiagnosisStage.REASONING.value,
        "consensus_reached": stage4_result.consensus_reached,
        "total_stage_rounds": {
            "0_consistency_check": stage0_result.total_rounds,
            "1_memory_extraction": stage1_result.total_rounds,
            "2_memory_update": stage2_result.total_rounds,
            "3_memory_retrieval": stage3_result.total_rounds,
            "4_reasoning": stage4_result.total_rounds
        },
        "stage_results": _serialize_stage_results(stage_results)
    }


def _serialize_stage_results(stage_results: Dict[str, StageDiscussionResult]) -> Dict:
    """将阶段结果序列化为可JSON化的格式"""
    result = {}
    for stage_name, stage_data in stage_results.items():
        result[stage_name] = {
            "consensus_reached": stage_data.consensus_reached,
            "final_passed": stage_data.final_passed,
            "final_label": stage_data.final_label,
            "final_reason": stage_data.final_reason,
            "total_rounds": stage_data.total_rounds,
            "discussion_history": stage_data.discussion_history
        }
    return result


# ============================================================================
# 主程序入口
# ============================================================================

def main():
    """主程序入口函数
    
    支持命令行参数：
        python run_diagnosis_discussion.py [options]
        
    参数说明：
        --max-rounds N: 每个阶段最大讨论轮次，默认：3
        --models: 参与讨论的模型，默认：deepseek gpt-4.1 gpt-5
        -i, --input: 输入文件路径
        -o, --output-dir: 输出目录路径
        -f, --output-file: 输出文件名
        
    示例：
        python run_diagnosis_discussion.py --max-rounds 3
        python run_diagnosis_discussion.py --models deepseek gpt-4.1 gpt-5
        python run_diagnosis_discussion.py -i data/input.json -o results/
    """
    parser = argparse.ArgumentParser(
        description="记忆诊断系统 - 多模型分阶段讨论版",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help="每个阶段最大讨论轮次 (默认: 3)"
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        default=["deepseek", "gpt-4.1", "gpt-5"],
        help="参与讨论的模型列表 (默认: deepseek gpt-4.1 gpt-5)"
    )
    
    parser.add_argument(
        "-i", "--input",
        type=str,
        default="data/input/mem0_mem/sample/sampled_qa_50.json",
        help="输入文件路径"
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="data/output/llm_annotation_discussion",
        help="输出目录路径"
    )
    
    parser.add_argument(
        "-f", "--output-file",
        type=str,
        default=None,
        help="输出文件名 (默认: 自动生成)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🚀 记忆诊断系统 - 多模型分阶段讨论版 启动")
    print("="*70)
    print(f"🤖 参与模型: {', '.join(args.models)}")
    print(f"🔄 每阶段最大讨论轮次: {args.max_rounds}")
    print(f"⚙️  配置: {DiagnosisConfig()}")
    print("="*70 + "\n")
    
    input_file = args.input
    input_basename = os.path.splitext(os.path.basename(input_file))[0]
    input_identifier = input_basename.replace(" ", "_").replace("(", "").replace(")", "")
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.output_file:
        output_filename = args.output_file
    else:
        models_str = "_".join([m.replace("-", "").replace(".", "") for m in args.models])
        output_filename = f"{input_identifier}_discussion_{args.max_rounds}rounds_{models_str}_{timestamp}.json"
    
    output_file = os.path.join(output_dir, output_filename)
    
    print(f"📁 输入文件: {input_file}")
    print(f"📁 输出目录: {output_dir}")
    print(f"📁 输出文件: {output_file}\n")
    
    if not os.path.exists(input_file):
        print(f"❌ 错误: 输入文件不存在: {input_file}")
        return
    
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
                # 优先使用 original_source 中的索引，否则使用列表索引
                original_source = qa_item.get("original_source", {})
                if original_source and original_source.get("qa_index") is not None:
                    item_id = f"{conv_id}_{original_source['qa_index']}"
                else:
                    item_id = f"{conv_id}_{qa_idx-1}"
                
                if item_id in processed_items:
                    print(f"⏭️  跳过已处理的问题: {item_id}\n")
                    continue
                
                print(f"🔍 开始处理问题 {qa_idx}/{len(qa_list)}: {item_id}")
                
                try:
                    p1 = qa_item.get("person1", {})
                    p2 = qa_item.get("person2", {})
                    memories1 = p1.get("memories", [])
                    memories2 = p2.get("memories", [])
                    
                    analysis = analyze_qa_pair_with_discussion(
                        qa_question=qa_item["qa_question"],
                        qa_answer=qa_item["qa_answer"],
                        qa_response=qa_item["qa_response"],
                        memories1=memories1,
                        memories2=memories2,
                        speaker1_memories=qa_item.get("speaker_1_memories", []),
                        speaker2_memories=qa_item.get("speaker_2_memories", []),
                        models=args.models,
                        max_rounds=args.max_rounds
                    )
                    
                    # 获取原始位置信息
                    original_source = qa_item.get("original_source", {})
                    if original_source:
                        original_id = f"{original_source.get('file', '')}_{original_source.get('key', '')}_{original_source.get('qa_index', '')}"
                    else:
                        original_id = item_id
                    
                    result = {
                        "conv_id_question_id": item_id,
                        "original_id": original_id,
                        "original_source": original_source,
                        "qa_question": qa_item["qa_question"],
                        "qa_answer": qa_item["qa_answer"],
                        "qa_response": qa_item["qa_response"],
                        "qa_category": qa_item.get("qa_category", ""),
                        "label": analysis["label"],
                        "reason": analysis["reason"],
                        "stage": analysis["stage"],
                        "diagnosis_mode": f"discussion_{args.max_rounds}rounds_per_stage",
                        "consensus_reached": analysis["consensus_reached"],
                        "total_stage_rounds": analysis["total_stage_rounds"],
                        "stage_results": analysis["stage_results"]
                    }
                    
                    results.append(result)
                    
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
        if results:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            consensus_count = sum(1 for r in results if r.get("consensus_reached", False))
            
            print("\n" + "="*70)
            print("🎉 处理完成")
            print("="*70)
            print(f"✅ 共处理 {len(results)} 个问题")
            print(f"🤝 达成共识: {consensus_count}/{len(results)} ({100*consensus_count/len(results):.1f}%)")
            print(f"📁 结果已保存到: {output_file}")
            print("="*70 + "\n")


if __name__ == "__main__":
    main()

