"""Multi-model discussion diagnosis."""

import json
import logging
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from memeval.config import DiagnosisConfig
from memeval.schema import (
    DiagnosisStage,
    DiagnosisStatus,
    MemoryData,
    QAData,
    SubjectMemoryData,
)
from memeval.schema.validation import validate_judge_response

from memeval.diagnosis.llm import call_llm_api, clean_prompt
from memeval.diagnosis.stages import subjects_field_str

__all__ = [
    "StageOpinion",
    "StageDiscussionResult",
    "generate_stage0_prompt",
    "generate_stage1_prompt",
    "generate_stage2_prompt",
    "generate_stage3_prompt",
    "generate_stage4_prompt",
    "discuss_stage",
    "analyze_qa_pair_with_discussion",
]


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
    status: DiagnosisStatus = DiagnosisStatus.COMPLETED


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
    status: DiagnosisStatus = DiagnosisStatus.COMPLETED


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

    subjects_memories_str = subjects_field_str(memory_data, "memories", "initial_results")

    base_prompt = f"""
You are an evaluation assistant for the Memory Extraction Stage.
Task:
1. Use their initial_results (and time_stamp if needed) to determine whether the extracted memories are sufficient to answer qa_question.
2. If sufficient → is_sufficient = true (label = null).
3. If insufficient, classify the issue:
   - "1.1": Missing key information
   - "1.2": Incorrect or conflicting information
   - "1.3": Ambiguous or overly generic information

Examples (subjects_memories pools every subject's extraction results, tagged by subject_id):

Example 1:
qa_question: "Where did Caroline move from 4 years ago?"
qa_answer: "Sweden"
qa_response: "home country"
subjects_memories: [{{"subject_id": "Caroline", "initial_results": ["Caroline moved from her home country 4 years ago"]}}, {{"subject_id": "Melanie", "initial_results": []}}]
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
- subjects_memories: {subjects_memories_str}
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

    subjects_memories_str = subjects_field_str(memory_data, "memories", "update_chain")

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
- subjects_memories: {subjects_memories_str}
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
    subjects_retrieval_str = subjects_field_str(memory_data, "retrieval")

    base_prompt = f"""
You are an evaluation assistant for the Memory Retrieval Stage.
Task:
Based strictly on subjects_retrieval (pooled retrieval results from every subject, tagged by subject_id):
1. Determine whether the retrieved memories contain enough correct information to answer qa_question.
2. If sufficient → is_sufficient = true (label = null).
3. If insufficient, determine the retrieval issue:
   - "3.1": Failed to recall correct information (missing the key facts)
   - "3.2": Unreasonable ranking (recalled irrelevant/common info while missing the most relevant facts)

Now evaluate the following:

Input:
- qa_question: {qa_question_str}
- qa_answer: {qa_answer_str}
- subjects_retrieval: {subjects_retrieval_str}
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
    subjects_retrieval_str = subjects_field_str(memory_data, "retrieval")

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
- subjects_retrieval: {subjects_retrieval_str}
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
    minimum_valid_opinions = len(models) // 2 + 1

    def parse_opinion(result: Dict) -> Tuple[bool, Optional[str], str]:
        validated = validate_judge_response(stage, result)
        if stage == DiagnosisStage.CONSISTENCY_CHECK:
            return validated["is_consistent"], None, validated["reason"]
        if stage == DiagnosisStage.REASONING:
            return False, validated["label"], validated["reason"]
        return validated["is_sufficient"], validated["label"], validated["reason"]

    # Round 1: independent judgments
    print(f"\n🔄 Round 1: Independent judgments")

    for model in models:
        print(f"   🤖 {model} is analyzing...")
        prompt = generate_prompt(None)  # No other opinions in round 1

        try:
            result = call_llm_api(clean_prompt(prompt), model, config)

            stage_passed, label, reason = parse_opinion(result)

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
                round_num=1,
                status=DiagnosisStatus.ERROR,
            )
            current_opinions.append(opinion)

    discussion_history.append({
        "round": 1,
        "opinions": [
            {
                "model": op.model_name,
                "passed": op.stage_passed,
                "label": op.label,
                "reason": op.reason,
                "status": op.status.value,
            }
            for op in current_opinions
        ]
    })

    def check_consensus(opinions: List[StageOpinion]) -> Tuple[bool, Optional[bool], Optional[str]]:
        """Return (consensus?, consensus_passed, consensus_label)."""
        opinions = [op for op in opinions if op.status == DiagnosisStatus.COMPLETED]
        if len(opinions) < minimum_valid_opinions:
            return False, None, None
        if stage == DiagnosisStage.REASONING:
            labels = [op.label for op in opinions]
            if len(set(labels)) == 1:
                return True, False, labels[0]
            return False, None, None
        else:
            passed_values = [op.stage_passed for op in opinions]
            if len(set(passed_values)) == 1:
                if passed_values[0]:
                    return True, True, None
                else:
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
            other_opinions = [
                op
                for op in current_opinions
                if op.model_name != model and op.status == DiagnosisStatus.COMPLETED
            ]
            current_model_opinion = next(op for op in current_opinions if op.model_name == model)

            print(f"   🤖 {model} is considering other opinions...")
            prompt = generate_prompt(other_opinions)

            try:
                result = call_llm_api(clean_prompt(prompt), model, config)

                stage_passed, label, reason = parse_opinion(result)

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
                opinion = StageOpinion(
                    model_name=model,
                    stage_passed=current_model_opinion.stage_passed,
                    label=current_model_opinion.label,
                    reason=f"Discussion failed, keep previous opinion: {str(e)}",
                    round_num=round_num,
                    status=current_model_opinion.status,
                )
                new_opinions.append(opinion)

        current_opinions = new_opinions

        discussion_history.append({
            "round": round_num,
            "opinions": [
                {
                    "model": op.model_name,
                    "passed": op.stage_passed,
                    "label": op.label,
                    "reason": op.reason,
                    "status": op.status.value,
                    "changed_from_passed": op.changed_from_passed,
                    "changed_from_label": op.changed_from_label
                }
                for op in current_opinions
            ]
        })

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
    valid_opinions = [op for op in current_opinions if op.status == DiagnosisStatus.COMPLETED]
    if len(valid_opinions) < minimum_valid_opinions:
        return StageDiscussionResult(
            stage=stage,
            consensus_reached=False,
            final_passed=False,
            final_label=None,
            final_reason=(
                f"Insufficient valid model opinions: {len(valid_opinions)}/{minimum_valid_opinions}"
            ),
            total_rounds=max_rounds,
            discussion_history=discussion_history,
            status=DiagnosisStatus.ERROR,
        )
    current_opinions = valid_opinions

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

        all_different = len(label_counter) == len(current_opinions) and len(current_opinions) > 1

        if all_different:
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
            labels = [op.label for op in current_opinions if not op.stage_passed]
            if labels:
                label_counter = Counter(labels)

                all_different = len(label_counter) == len(labels) and len(labels) > 1

                if all_different:
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


def analyze_qa_pair_with_discussion(
    qa_question: str,
    qa_answer: str,
    qa_response: str,
    subjects: List[Dict],
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
        subjects: list of {"subject_id", "memories", "retrieval"} dicts, one per
            party whose memory contributed to this QA record
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

    qa_data = QAData(
        question=qa_question,
        answer=qa_answer,
        response=qa_response
    )

    memory_data = MemoryData(
        subjects=[
            SubjectMemoryData(
                subject_id=subject.get("subject_id", ""),
                memories=subject.get("memories", []),
                retrieval=subject.get("retrieval", []),
            )
            for subject in subjects
        ]
    )

    stage_results = {}

    def total_stage_rounds() -> Dict[str, int]:
        return {
            stage_name: stage_result.total_rounds
            for stage_name, stage_result in stage_results.items()
        }

    def execution_error(result: StageDiscussionResult) -> Dict:
        return {
            "label": None,
            "reason": result.final_reason,
            "stage": DiagnosisStage.ERROR.value,
            "status": DiagnosisStatus.ERROR.value,
            "answer_correct": False,
            "consensus_reached": False,
            "total_stage_rounds": total_stage_rounds(),
            "stage_results": _serialize_stage_results(stage_results),
        }


    # ========== Stage 0: consistency check ==========
    stage0_result = discuss_stage(
        DiagnosisStage.CONSISTENCY_CHECK,
        qa_data, memory_data, models, max_rounds, config
    )
    stage_results["0_consistency_check"] = stage0_result
    if stage0_result.status == DiagnosisStatus.ERROR:
        return execution_error(stage0_result)

    if stage0_result.final_passed:
        print(f"\n{'='*70}")
        print(f"✅ Diagnosis completed: response is consistent with the answer")
        print(f"{'='*70}\n")

        return {
            "label": None,
            "reason": stage0_result.final_reason,
            "stage": DiagnosisStage.CONSISTENCY_CHECK.value,
            "status": DiagnosisStatus.COMPLETED.value,
            "answer_correct": True,
            "consensus_reached": stage0_result.consensus_reached,
            "total_stage_rounds": total_stage_rounds(),
            "stage_results": _serialize_stage_results(stage_results)
        }

    # ========== Stage 1: memory extraction ==========
    stage1_result = discuss_stage(
        DiagnosisStage.MEMORY_EXTRACTION,
        qa_data, memory_data, models, max_rounds, config
    )
    stage_results["1_memory_extraction"] = stage1_result
    if stage1_result.status == DiagnosisStatus.ERROR:
        return execution_error(stage1_result)

    if not stage1_result.final_passed:
        print(f"\n{'='*70}")
        print(f"❌ Diagnosis completed: issue found at memory extraction stage")
        print(f"   Label: {stage1_result.final_label}")
        print(f"{'='*70}\n")

        return {
            "label": stage1_result.final_label,
            "reason": stage1_result.final_reason,
            "stage": DiagnosisStage.MEMORY_EXTRACTION.value,
            "status": DiagnosisStatus.COMPLETED.value,
            "answer_correct": False,
            "consensus_reached": stage1_result.consensus_reached,
            "total_stage_rounds": total_stage_rounds(),
            "stage_results": _serialize_stage_results(stage_results)
        }

    # ========== Stage 2: memory update ==========
    stage2_result = discuss_stage(
        DiagnosisStage.MEMORY_UPDATE,
        qa_data, memory_data, models, max_rounds, config
    )
    stage_results["2_memory_update"] = stage2_result
    if stage2_result.status == DiagnosisStatus.ERROR:
        return execution_error(stage2_result)

    if not stage2_result.final_passed:
        print(f"\n{'='*70}")
        print(f"❌ Diagnosis completed: issue found at memory update stage")
        print(f"   Label: {stage2_result.final_label}")
        print(f"{'='*70}\n")

        return {
            "label": stage2_result.final_label,
            "reason": stage2_result.final_reason,
            "stage": DiagnosisStage.MEMORY_UPDATE.value,
            "status": DiagnosisStatus.COMPLETED.value,
            "answer_correct": False,
            "consensus_reached": stage2_result.consensus_reached,
            "total_stage_rounds": total_stage_rounds(),
            "stage_results": _serialize_stage_results(stage_results)
        }


    # ========== Stage 3: memory retrieval ==========
    stage3_result = discuss_stage(
        DiagnosisStage.MEMORY_RETRIEVAL,
        qa_data, memory_data, models, max_rounds, config
    )
    stage_results["3_memory_retrieval"] = stage3_result
    if stage3_result.status == DiagnosisStatus.ERROR:
        return execution_error(stage3_result)

    if not stage3_result.final_passed:
        print(f"\n{'='*70}")
        print(f"❌ Diagnosis completed: issue found at memory retrieval stage")
        print(f"   Label: {stage3_result.final_label}")
        print(f"{'='*70}\n")

        return {
            "label": stage3_result.final_label,
            "reason": stage3_result.final_reason,
            "stage": DiagnosisStage.MEMORY_RETRIEVAL.value,
            "status": DiagnosisStatus.COMPLETED.value,
            "answer_correct": False,
            "consensus_reached": stage3_result.consensus_reached,
            "total_stage_rounds": total_stage_rounds(),
            "stage_results": _serialize_stage_results(stage_results)
        }

    # ========== Stage 4: reasoning ==========
    stage4_result = discuss_stage(
        DiagnosisStage.REASONING,
        qa_data, memory_data, models, max_rounds, config
    )
    stage_results["4_reasoning"] = stage4_result
    if stage4_result.status == DiagnosisStatus.ERROR:
        return execution_error(stage4_result)

    print(f"\n{'='*70}")
    print(f"❌ Diagnosis completed: issue found at reasoning stage")
    print(f"   Label: {stage4_result.final_label}")
    print(f"{'='*70}\n")

    return {
        "label": stage4_result.final_label,
        "reason": stage4_result.final_reason,
        "stage": DiagnosisStage.REASONING.value,
        "status": DiagnosisStatus.COMPLETED.value,
        "answer_correct": False,
        "consensus_reached": stage4_result.consensus_reached,
        "total_stage_rounds": total_stage_rounds(),
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
            "discussion_history": stage_data.discussion_history,
            "status": stage_data.status.value,
        }
    return result

