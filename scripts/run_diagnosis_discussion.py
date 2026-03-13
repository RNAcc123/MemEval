"""
Memory diagnosis system - multi-model discussion version

This module implements a multi-model, stage-wise discussion framework:
- Stage 0: Consistency check (3-model discussion)
- Stage 1: Memory extraction diagnosis (3-model discussion)
- Stage 2: Memory update diagnosis (3-model discussion)
- Stage 3: Memory retrieval diagnosis (3-model discussion)
- Stage 4: Reasoning diagnosis (3-model discussion)

Within each stage, three models first make independent judgments, then run
multiple discussion rounds to reach consensus or decide by voting.
"""

# Standard library imports
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

# Third-party imports
from dotenv import load_dotenv
from requests.exceptions import RequestException, Timeout

# ============================================================================
# Configuration and initialization
# ============================================================================

logging.getLogger('grpc').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', module='grpc')

load_dotenv()
os.environ['GRPC_ALTS_CREDENTIALS_ENVIRONMENT_OVERRIDE'] = '1'

# Import classes and helpers from the baseline module
from run_diagnosis import (
    APIConfig, DiagnosisConfig, QAData, MemoryData, DiagnosisStage,
    StageResult, DiagnosisResult, API_CONFIG,
    load_json_file, clean_prompt, extract_json_from_response,
    call_llm_api
)


# ============================================================================
# Discussion-related dataclasses
# ============================================================================

@dataclass
class StageOpinion:
    """A single model's opinion for a given stage."""
    model_name: str
    stage_passed: bool
    label: Optional[str]
    reason: str
    round_num: int
    changed_from_passed: Optional[bool] = None
    changed_from_label: Optional[str] = None


@dataclass
class StageDiscussionResult:
    """Final discussion outcome for a stage."""
    stage: DiagnosisStage
    consensus_reached: bool
    final_passed: bool
    final_label: Optional[str]
    final_reason: str
    total_rounds: int
    discussion_history: List[Dict]


@dataclass
class FullDiscussionResult:
    """Full diagnosis result produced by multi-stage discussions."""
    final_label: Optional[str]
    final_reason: str
    final_stage: DiagnosisStage
    stage_results: Dict[str, StageDiscussionResult]


# ============================================================================
# Prompt builders (per stage)
# ============================================================================

def generate_stage0_prompt(qa_data: QAData, other_opinions: List[StageOpinion] = None) -> str:
    """Build the stage-0 prompt (consistency check)."""
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
    """Build the stage-1 prompt (memory extraction)."""
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
    """Build the stage-2 prompt (memory update)."""
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
    """Build the stage-3 prompt (memory retrieval)."""
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
    """Build the stage-4 prompt (reasoning)."""
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
# Single-stage discussion
# ============================================================================

def discuss_stage(
    stage: DiagnosisStage,
    qa_data: QAData,
    memory_data: MemoryData,
    models: List[str],
    max_rounds: int = 3,
    config: Optional[DiagnosisConfig] = None
) -> StageDiscussionResult:
    """Run a multi-model discussion for a single stage.
    
    Args:
        stage: current diagnosis stage
        qa_data: QA data
        memory_data: memory data
        models: list of models participating in the discussion
        max_rounds: maximum number of discussion rounds
        config: diagnosis configuration
        
    Returns:
        A StageDiscussionResult instance.
    """
    stage_name_map = {
        DiagnosisStage.CONSISTENCY_CHECK: "Consistency Check",
        DiagnosisStage.MEMORY_EXTRACTION: "Memory Extraction",
        DiagnosisStage.MEMORY_UPDATE: "Memory Update",
        DiagnosisStage.MEMORY_RETRIEVAL: "Memory Retrieval",
        DiagnosisStage.REASONING: "Reasoning",
    }
    
    print(f"\n{'='*60}")
    print(f"📋 Stage: {stage.value} - {stage_name_map.get(stage, stage.value)}")
    print(f"{'='*60}")
    
    # Select the prompt generator for this stage
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
    
    # Round 1: independent judgments
    print(f"\n🔄 Round 1: Independent judgments")
    
    for model in models:
        print(f"   🤖 {model} is analyzing...")
        prompt = generate_prompt(None)  # No other opinions in round 1
        
        try:
            result = call_llm_api(clean_prompt(prompt), model, config)
            
            # Parse results depending on the stage
            if stage == DiagnosisStage.CONSISTENCY_CHECK:
                stage_passed = result.get("is_consistent", False)
                label = None
            elif stage == DiagnosisStage.REASONING:
                stage_passed = False  # Stage 4 always returns a label
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
            logging.error(f"Model {model} failed to analyze stage {stage.value}: {str(e)}")
            opinion = StageOpinion(
                model_name=model,
                stage_passed=False,
                label=None,
                reason=f"Analysis failed: {str(e)}",
                round_num=1
            )
            current_opinions.append(opinion)
    
    # Record round-1 history
    discussion_history.append({
        "round": 1,
        "opinions": [
            {"model": op.model_name, "passed": op.stage_passed, "label": op.label, "reason": op.reason}
            for op in current_opinions
        ]
    })
    
    # Check whether a consensus is reached
    def check_consensus(opinions: List[StageOpinion]) -> Tuple[bool, Optional[bool], Optional[str]]:
        """Return (consensus?, consensus_passed, consensus_label)."""
        if stage == DiagnosisStage.REASONING:
            # Stage 4: check label only
            labels = [op.label for op in opinions]
            if len(set(labels)) == 1:
                return True, False, labels[0]
            return False, None, None
        else:
            # Other stages: check passed first; if all failed then check label
            passed_values = [op.stage_passed for op in opinions]
            if len(set(passed_values)) == 1:
                if passed_values[0]:  # all passed
                    return True, True, None
                else:  # all failed, check label
                    labels = [op.label for op in opinions]
                    if len(set(labels)) == 1:
                        return True, False, labels[0]
            return False, None, None
    
    consensus, consensus_passed, consensus_label = check_consensus(current_opinions)
    
    if consensus:
        print(f"\n🎉 Consensus reached in round 1! passed={consensus_passed}, label={consensus_label}")
        return StageDiscussionResult(
            stage=stage,
            consensus_reached=True,
            final_passed=consensus_passed,
            final_label=consensus_label,
            final_reason=current_opinions[0].reason,
            total_rounds=1,
            discussion_history=discussion_history
        )
    
    # Subsequent rounds: discussion
    for round_num in range(2, max_rounds + 1):
        print(f"\n🔄 Round {round_num}: Discussion")
        
        new_opinions = []
        
        for model in models:
            # Collect other models' opinions
            other_opinions = [op for op in current_opinions if op.model_name != model]
            current_model_opinion = next(op for op in current_opinions if op.model_name == model)
            
            print(f"   🤖 {model} is considering other opinions...")
            prompt = generate_prompt(other_opinions)
            
            try:
                result = call_llm_api(clean_prompt(prompt), model, config)
                
                # Parse results depending on the stage
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
                
                # Track whether the opinion changed
                changed_from_passed = None
                changed_from_label = None
                if stage_passed != current_model_opinion.stage_passed:
                    changed_from_passed = current_model_opinion.stage_passed
                    print(f"      ↪️ {model} changed judgment: passed {current_model_opinion.stage_passed} -> {stage_passed}")
                elif label != current_model_opinion.label:
                    changed_from_label = current_model_opinion.label
                    print(f"      ↪️ {model} changed label: {current_model_opinion.label} -> {label}")
                else:
                    print(f"      ✓ {model} kept judgment: passed={stage_passed}, label={label}")
                
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
                logging.error(f"Model {model} discussion failed: {str(e)}")
                # Keep the previous opinion
                opinion = StageOpinion(
                    model_name=model,
                    stage_passed=current_model_opinion.stage_passed,
                    label=current_model_opinion.label,
                    reason=f"Discussion failed, keep previous opinion: {str(e)}",
                    round_num=round_num
                )
                new_opinions.append(opinion)
        
        current_opinions = new_opinions
        
        # Record this round's history
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
        
        # Check consensus
        consensus, consensus_passed, consensus_label = check_consensus(current_opinions)
        
        if consensus:
            print(f"\n🎉 Consensus reached in round {round_num}! passed={consensus_passed}, label={consensus_label}")
            return StageDiscussionResult(
                stage=stage,
                consensus_reached=True,
                final_passed=consensus_passed,
                final_label=consensus_label,
                final_reason=current_opinions[0].reason,
                total_rounds=round_num,
                discussion_history=discussion_history
            )
    
    # No consensus; decide by voting
    print(f"\n⚠️ No consensus after {max_rounds} rounds, proceed to voting")
    
    # Helper: if all results differ, prefer the gpt-5 opinion (when available)
    def get_gpt5_opinion(opinions: List[StageOpinion]) -> Optional[StageOpinion]:
        """Get the opinion from gpt-5 (if present)."""
        for op in opinions:
            if op.model_name == "gpt-5":
                return op
        return None
    
    if stage == DiagnosisStage.REASONING:
        # Stage 4: vote on label only
        labels = [op.label for op in current_opinions]
        label_counter = Counter(labels)
        
        # Check whether all results are different (e.g., 1:1:1)
        all_different = len(label_counter) == len(current_opinions) and len(current_opinions) > 1
        
        if all_different:
            # All results differ; use gpt-5's result
            gpt5_op = get_gpt5_opinion(current_opinions)
            if gpt5_op:
                final_label = gpt5_op.label
                print(f"📊 Voting results: {dict(label_counter)}")
                print(f"⚠️ All results are different, use gpt-5 result")
                print(f"🏆 Selected label: {final_label}")
            else:
                final_label = label_counter.most_common(1)[0][0]
                print(f"📊 Voting results: {dict(label_counter)}")
                print(f"🏆 Selected label: {final_label}")
        else:
            final_label = label_counter.most_common(1)[0][0]
            print(f"📊 Voting results: {dict(label_counter)}")
            print(f"🏆 Selected label: {final_label} (most votes)")
        final_passed = False
    else:
        # Vote on passed first
        passed_values = [op.stage_passed for op in current_opinions]
        passed_counter = Counter(passed_values)
        final_passed = passed_counter.most_common(1)[0][0]
        
        if final_passed:
            final_label = None
            print(f"📊 Voting results: passed={dict(passed_counter)}")
            print(f"🏆 Stage passed")
        else:
            # If not passed, vote on label
            labels = [op.label for op in current_opinions if not op.stage_passed]
            if labels:
                label_counter = Counter(labels)
                
                # Check whether all results are different (e.g., 1:1:1)
                all_different = len(label_counter) == len(labels) and len(labels) > 1
                
                if all_different:
                    # All results differ; use gpt-5's result
                    gpt5_op = get_gpt5_opinion([op for op in current_opinions if not op.stage_passed])
                    if gpt5_op:
                        final_label = gpt5_op.label
                        print(f"📊 Voting results: passed={dict(passed_counter)}, labels={dict(label_counter)}")
                        print(f"⚠️ All results are different, use gpt-5 result")
                        print(f"🏆 Selected label: {final_label}")
                    else:
                        final_label = label_counter.most_common(1)[0][0]
                        print(f"📊 Voting results: passed={dict(passed_counter)}, labels={dict(label_counter)}")
                        print(f"🏆 Selected label: {final_label}")
                else:
                    final_label = label_counter.most_common(1)[0][0]
                    print(f"📊 Voting results: passed={dict(passed_counter)}, labels={dict(label_counter)}")
                    print(f"🏆 Selected label: {final_label} (most votes)")
            else:
                final_label = None
    
    # Get the reason associated with the selected opinion
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
# Full multi-stage discussion
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
    """Analyze a QA pair using the multi-model, stage-wise discussion mechanism.
    
    Stages are executed in order. In each stage, three models make independent
    judgments first, then discuss for multiple rounds to reach consensus.
    
    Args:
        qa_question: question text
        qa_answer: reference answer
        qa_response: model response
        memories1: person1 memory data
        memories2: person2 memory data
        speaker1_memories: retrieved memories for speaker1
        speaker2_memories: retrieved memories for speaker2
        models: list of models participating in discussion
        max_rounds: max discussion rounds per stage
        config: diagnosis configuration
        
    Returns:
        A dict containing the discussion result.
    """
    if models is None:
        models = ["deepseek", "gpt-4.1", "gpt-5"]
    
    print(f"\n{'='*70}")
    print(f"🗣️  Multi-model staged discussion diagnosis")
    print(f"{'='*70}")
    print(f"📝 Question: {qa_question}")
    print(f"🤖 Participating models: {', '.join(models)}")
    print(f"🔄 Max rounds per stage: {max_rounds}")
    print(f"{'='*70}")
    
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
    
    stage_results = {}
    
    # ========== Stage 0: consistency check ==========
    stage0_result = discuss_stage(
        DiagnosisStage.CONSISTENCY_CHECK,
        qa_data, memory_data, models, max_rounds, config
    )
    stage_results["0_consistency_check"] = stage0_result
    
    if stage0_result.final_passed:
        # Consistent, return directly
        print(f"\n{'='*70}")
        print(f"✅ Diagnosis completed: response is consistent with the answer")
        print(f"{'='*70}\n")
        
        return {
            "label": None,
            "reason": stage0_result.final_reason,
            "stage": DiagnosisStage.CONSISTENCY_CHECK.value,
            "consensus_reached": stage0_result.consensus_reached,
            "total_stage_rounds": {"0_consistency_check": stage0_result.total_rounds},
            "stage_results": _serialize_stage_results(stage_results)
        }
    
    # ========== Stage 1: memory extraction ==========
    stage1_result = discuss_stage(
        DiagnosisStage.MEMORY_EXTRACTION,
        qa_data, memory_data, models, max_rounds, config
    )
    stage_results["1_memory_extraction"] = stage1_result
    
    if not stage1_result.final_passed:
        print(f"\n{'='*70}")
        print(f"❌ Diagnosis completed: issue found at memory extraction stage")
        print(f"   Label: {stage1_result.final_label}")
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
    
    # ========== Stage 2: memory update ==========
    stage2_result = discuss_stage(
        DiagnosisStage.MEMORY_UPDATE,
        qa_data, memory_data, models, max_rounds, config
    )
    stage_results["2_memory_update"] = stage2_result
    
    if not stage2_result.final_passed:
        print(f"\n{'='*70}")
        print(f"❌ Diagnosis completed: issue found at memory update stage")
        print(f"   Label: {stage2_result.final_label}")
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
    
    # ========== Stage 3: memory retrieval ==========
    stage3_result = discuss_stage(
        DiagnosisStage.MEMORY_RETRIEVAL,
        qa_data, memory_data, models, max_rounds, config
    )
    stage_results["3_memory_retrieval"] = stage3_result
    
    if not stage3_result.final_passed:
        print(f"\n{'='*70}")
        print(f"❌ Diagnosis completed: issue found at memory retrieval stage")
        print(f"   Label: {stage3_result.final_label}")
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
    
    # ========== Stage 4: reasoning ==========
    stage4_result = discuss_stage(
        DiagnosisStage.REASONING,
        qa_data, memory_data, models, max_rounds, config
    )
    stage_results["4_reasoning"] = stage4_result
    
    print(f"\n{'='*70}")
    print(f"❌ Diagnosis completed: issue found at reasoning stage")
    print(f"   Label: {stage4_result.final_label}")
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
    """Serialize stage results into a JSON-serializable structure."""
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
# Main entrypoint
# ============================================================================

def main():
    """Main entrypoint.
    
    Supports CLI arguments:
        python run_diagnosis_discussion.py [options]
        
    Arguments:
        --max-rounds N: max discussion rounds per stage (default: 3)
        --models: models participating in discussions (default: deepseek gpt-4.1 gpt-5)
        -i, --input: input file path
        -o, --output-dir: output directory
        -f, --output-file: output filename
        
    Examples:
        python run_diagnosis_discussion.py --max-rounds 3
        python run_diagnosis_discussion.py --models deepseek gpt-4.1 gpt-5
        python run_diagnosis_discussion.py -i data/input.json -o results/
    """
    parser = argparse.ArgumentParser(
        description="Memory diagnosis system - multi-model staged discussion edition",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help="Maximum discussion rounds per stage (default: 3)"
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        default=["deepseek", "gpt-4.1", "gpt-5"],
        help="List of models participating in discussion (default: deepseek gpt-4.1 gpt-5)"
    )
    
    parser.add_argument(
        "-i", "--input",
        type=str,
        default="data/input/mem0_mem/sample/sampled_qa_50.json",
        help="Input file path"
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="data/output/llm_annotation_discussion",
        help="Output directory path"
    )
    
    parser.add_argument(
        "-f", "--output-file",
        type=str,
        default=None,
        help="Output filename (default: auto-generated)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🚀 Memory diagnosis system - multi-model staged discussion edition started")
    print("="*70)
    print(f"🤖 Participating models: {', '.join(args.models)}")
    print(f"🔄 Max rounds per stage: {args.max_rounds}")
    print(f"⚙️  Config: {DiagnosisConfig()}")
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
    
    print(f"📁 Input file: {input_file}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📁 Output file: {output_file}\n")
    
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file does not exist: {input_file}")
        return
    
    try:
        data = load_json_file(input_file)
        print(f"✅ Loaded {len(data)} conversations successfully\n")
    except Exception as e:
        logging.error(f"Failed to load input file: {str(e)}")
        print(f"❌ Error: Failed to parse input file: {str(e)}")
        return
    
    # Load previously processed results (supports resume)
    results = []
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                results = json.load(f)
                logging.info(f"Loaded {len(results)} historical results")
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logging.warning(f"Failed to load historical results: {str(e)}, starting from scratch")
            results = []
    
    processed_items = {item["conv_id_question_id"] for item in results}
    
    try:
        total_convs = len(data)
        print(f"📊 Start processing, total conversations to analyze: {total_convs}\n")
        
        for conv_idx, (conv_id, qa_list) in enumerate(data.items(), 1):
            print(f"\n{'='*70}")
            print(f"📝 Processing conversation {conv_id} ({conv_idx}/{total_convs})")
            print(f"{'='*70}\n")
            
            for qa_idx, qa_item in enumerate(qa_list, 1):
                # Prefer the index from original_source; otherwise use list index
                original_source = qa_item.get("original_source", {})
                if original_source and original_source.get("qa_index") is not None:
                    item_id = f"{conv_id}_{original_source['qa_index']}"
                else:
                    item_id = f"{conv_id}_{qa_idx-1}"
                
                if item_id in processed_items:
                    print(f"⏭️  Skip already processed question: {item_id}\n")
                    continue
                
                print(f"🔍 Start processing question {qa_idx}/{len(qa_list)}: {item_id}")
                
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
                    
                    # Capture original location info
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
                    
                    print(f"✅ Question {item_id} processed and saved\n")
                    
                except Exception as e:
                    logging.error(f"Error while processing question {item_id}: {str(e)}")
                    print(f"❌ Failed to process question {item_id}: {str(e)}\n")
                    continue
                    
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted, saving...\n")
    except Exception as e:
        logging.error(f"Unexpected error during processing: {str(e)}")
        print(f"\n❌ Error occurred during processing: {str(e)}\n")
    finally:
        if results:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            consensus_count = sum(1 for r in results if r.get("consensus_reached", False))
            
            print("\n" + "="*70)
            print("🎉 Processing completed")
            print("="*70)
            print(f"✅ Total processed questions: {len(results)}")
            print(f"🤝 Consensus reached: {consensus_count}/{len(results)} ({100*consensus_count/len(results):.1f}%)")
            print(f"📁 Results saved to: {output_file}")
            print("="*70 + "\n")


if __name__ == "__main__":
    main()

