"""Individual diagnosis stage functions."""

import json
import logging
from typing import List, Optional

from memeval.config import DiagnosisConfig
from memeval.schema import (
    DiagnosisStage,
    DiagnosisStatus,
    MemoryData,
    QAData,
    StageResult,
    UsageStats,
)
from memeval.schema.validation import validate_judge_response

from memeval.diagnosis.llm import call_llm_api

__all__ = [
    "subjects_payload",
    "subjects_field_str",
    "stage0_consistency_check",
    "stage1_memory_extraction",
    "stage2_memory_update",
    "stage3_memory_retrieval",
    "stage4_reasoning",
]


def _print_stage_header(stage_name: str, stage_number: int = 0):
    """Print a stage header."""
    print("=" * 60)
    print(f"Stage {stage_number}: {stage_name}")
    print("=" * 60)


def subjects_payload(memory_data: MemoryData, list_field: str, item_field: Optional[str] = None) -> List[dict]:
    """Flatten every subject's memories/retrieval into one pooled, tagged list.

    Each item is tagged with its subject_id so an N=1 or N=2+ subject dataset
    produces the same shape; stages judge the pooled evidence rather than
    comparing fixed person1/person2 fields.
    """
    payload = []
    for subject in memory_data.subjects:
        for item in getattr(subject, list_field):
            if item_field is not None:
                payload.append({
                    "subject_id": subject.subject_id,
                    "time_stamp": item.get("time_stamp", ""),
                    item_field: item.get(item_field, []),
                })
            elif isinstance(item, dict):
                payload.append({"subject_id": subject.subject_id, **item})
            else:
                payload.append({"subject_id": subject.subject_id, "value": item})
    return payload


def subjects_field_str(memory_data: MemoryData, list_field: str, item_field: Optional[str] = None) -> str:
    return json.dumps(subjects_payload(memory_data, list_field, item_field), ensure_ascii=False)


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
        result = validate_judge_response(DiagnosisStage.CONSISTENCY_CHECK, result)
        is_consistent = result["is_consistent"]

        stage_result = StageResult(
            stage_passed=is_consistent,
            label=None if is_consistent else "inconsistent",
            reason=result["reason"],
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
            stage=DiagnosisStage.ERROR,
            status=DiagnosisStatus.ERROR,
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
    subjects_memories_str = subjects_field_str(memory_data, "memories", "initial_results")

    prompt = f"""
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

Example 2:
qa_question: "What kind of films does Joanna enjoy?"
qa_answer: "Dramas and emotionally-driven films"
qa_response: "dramas and romantic comedies"
subjects_memories: [{{"subject_id": "Joanna", "initial_results": ["Joanna enjoys dramas and emotionally-driven films."]}}, {{"subject_id": "Melanie", "initial_results": ["Joanna enjoys dramas and romantic comedies."]}}]
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
subjects_memories: [{{"subject_id": "Maria", "initial_results": ["Maria dropped off baked goods..."]}}, {{"subject_id": "John", "initial_results": ["Maria dropped off baked goods..."]}}]
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
- subjects_memories: {subjects_memories_str}

Output format:
{{
  "is_sufficient": true/false,
  "label": "1.1" or "1.2" or "1.3" or null,
  "reason": "Detailed explanation"
}}

"""

    try:
        result = call_llm_api(prompt, model, config, usage_stats=usage_stats, stage_name="stage1_memory_extraction")
        result = validate_judge_response(DiagnosisStage.MEMORY_EXTRACTION, result)
        is_sufficient = result["is_sufficient"]

        stage_result = StageResult(
            stage_passed=is_sufficient,
            label=result["label"],
            reason=result["reason"],
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
            stage=DiagnosisStage.ERROR,
            status=DiagnosisStatus.ERROR,
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
    subjects_memories_str = subjects_field_str(memory_data, "memories", "update_chain")

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
- subjects_memories: {subjects_memories_str}

Output format:
{{
  "is_sufficient": true/false,
  "label": "2.1" or "2.2" or "2.3" or null,
  "reason": "Detailed explanation"
}}
"""

    try:
        result = call_llm_api(prompt, model, config, usage_stats=usage_stats, stage_name="stage2_memory_update")
        result = validate_judge_response(DiagnosisStage.MEMORY_UPDATE, result)
        is_sufficient = result["is_sufficient"]

        stage_result = StageResult(
            stage_passed=is_sufficient,
            label=result["label"],
            reason=result["reason"],
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
            stage=DiagnosisStage.ERROR,
            status=DiagnosisStatus.ERROR,
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
    subjects_retrieval_str = subjects_field_str(memory_data, "retrieval")

    prompt = f"""
You are an evaluation assistant for the Memory Retrieval Stage.
Task:
Based strictly on subjects_retrieval (pooled retrieval results from every subject, tagged by subject_id):
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
subjects_retrieval: [
  {{"subject_id": "Melanie", "text": "Melanie prioritizes her mental health..."}},
  {{"subject_id": "Melanie", "text": "Melanie enjoys running as a way to de-stress..."}},
  {{"subject_id": "Melanie", "text": "Melanie is thankful for her family..."}},
  {{"subject_id": "John", "text": "Melanie finds self-care to be a work in progress..."}},
  {{"subject_id": "John", "text": "Melanie has been running longer..."}},
  {{"subject_id": "John", "text": "Melanie values mental health..."}}
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
- subjects_retrieval: {subjects_retrieval_str}

Output format:
{{
  "is_sufficient": true/false,
  "label": "3.1" or "3.2" or null,
  "reason": "Detailed explanation"
}}
"""

    try:
        result = call_llm_api(prompt, model, config, usage_stats=usage_stats, stage_name="stage3_memory_retrieval")
        result = validate_judge_response(DiagnosisStage.MEMORY_RETRIEVAL, result)
        is_sufficient = result["is_sufficient"]

        stage_result = StageResult(
            stage_passed=is_sufficient,
            label=result["label"],
            reason=result["reason"],
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
            stage=DiagnosisStage.ERROR,
            status=DiagnosisStatus.ERROR,
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
    subjects_retrieval_str = subjects_field_str(memory_data, "retrieval")

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
subjects_retrieval: [{{"subject_id": "Melanie", "text": "Melanie prioritizes her mental health..."}}, {{"subject_id": "John", "text": "Melanie ... roasted marshmallows ... and told stories..."}}]
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
- subjects_retrieval: {subjects_retrieval_str}

Output format:
{{
  "label": "4.1" or "4.2" or "4.3",
  "reason": "Detailed explanation"
}}
"""

    try:
        result = call_llm_api(prompt, model, config, usage_stats=usage_stats, stage_name="stage4_reasoning")
        result = validate_judge_response(DiagnosisStage.REASONING, result)

        stage_result = StageResult(
            stage_passed=False,
            label=result["label"],
            reason=result["reason"],
            stage=DiagnosisStage.REASONING
        )

        print(f"✓ Reasoning issue type: {stage_result.label}")
        print(f"  Reason: {stage_result.reason}\n")

        return stage_result
    except Exception as e:
        logging.error(f"Stage 4 error: {str(e)}")
        return StageResult(
            stage_passed=False,
            label=None,
            reason=f"Stage 4 error: {str(e)}",
            stage=DiagnosisStage.ERROR,
            status=DiagnosisStatus.ERROR,
        )

