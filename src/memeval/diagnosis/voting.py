"""Multi-round voting diagnosis."""

import logging
from collections import Counter
from typing import Dict, List, Optional

from memeval.schema import DiagnosisStage, DiagnosisStatus, QAData, UsageStats

from memeval.diagnosis.single import _build_memory_data, analyze_qa_pair

__all__ = ["analyze_qa_pair_with_voting"]


def analyze_qa_pair_with_voting(
    qa_question: str,
    qa_answer: str,
    qa_response: str,
    subjects: List[Dict],
    model: str = "deepseek",
    num_votes: int = 3,
    min_valid_votes: Optional[int] = None,
) -> Dict:
    """Analyze a QA pair (and retrieved memories) using a voting mechanism.

    Args:
        qa_question: question text
        qa_answer: reference answer
        qa_response: model response
        subjects: list of {"subject_id", "memories", "retrieval"} dicts, one per
            party whose memory contributed to this QA record
        model: primary model to use
        num_votes: number of voting rounds

    Returns:
        Dict containing the final diagnosis result and voting details
    """
    if num_votes < 1:
        raise ValueError("num_votes must be at least 1")
    if min_valid_votes is None:
        min_valid_votes = num_votes // 2 + 1
    if min_valid_votes < 1 or min_valid_votes > num_votes:
        raise ValueError("min_valid_votes must be between 1 and num_votes")

    print(f"\n🗳️  Question: {qa_question}")
    print(f"📊 Run {num_votes} voting rounds with {model} as the primary model (a different model may be used each round)\n")

    qa_data = QAData(
        question=qa_question,
        answer=qa_answer,
        response=qa_response
    )

    memory_data = _build_memory_data(subjects)

    vote_results = []

    aggregated_stats = UsageStats()

    models = ["deepseek", "gpt-4.1", "gpt-5"]

    if model not in models:
        models.insert(0, model)
    else:
        models.remove(model)
        models.insert(0, model)

    used_models = []
    for i in range(num_votes):
        current_model = None
        for m in models:
            if m not in used_models:
                current_model = m
                break

        if current_model is None:
            unused_models = [m for m in models if m != model]
            if unused_models:
                current_model = unused_models[len(used_models) % len(unused_models)]
            else:
                current_model = models[len(used_models) % len(models)]

        used_models.append(current_model)
        print(f"🔄 Round {i+1}/{num_votes}, model: {current_model}")

        try:
            result = analyze_qa_pair(qa_data, memory_data, model=current_model)
            if result.usage_stats is not None:
                aggregated_stats.merge(result.usage_stats)
            result_dict = {
                "label": result.label,
                "reason": result.reason,
                "stage": result.stage.value if isinstance(result.stage, DiagnosisStage) else result.stage,
                "status": result.status.value,
                "answer_correct": result.answer_correct,
                "used_model": current_model,
            }
            vote_results.append(result_dict)
            print(f"   ✅ Round {i+1} completed: label={result.label}, model={current_model}\n")
        except Exception as e:
            logging.error(f"Round {i+1} analysis failed: {str(e)}")
            print(f"   ❌ Round {i+1} analysis failed: {str(e)}\n")
            vote_results.append({
                "label": None,
                "reason": f"API call failed: {str(e)}",
                "stage": "error",
                "status": DiagnosisStatus.ERROR.value,
                "answer_correct": False,
                "used_model": current_model
            })

    valid_results = [result for result in vote_results if result["status"] == DiagnosisStatus.COMPLETED.value]
    failed_results = [result for result in vote_results if result["status"] != DiagnosisStatus.COMPLETED.value]
    voting_details = {
        "requested_votes": num_votes,
        "minimum_valid_votes": min_valid_votes,
        "valid_votes": len(valid_results),
        "failed_votes": len(failed_results),
        "label_votes": {},
        "individual_results": vote_results,
        "tie": False,
        "tie_policy": "primary_model",
    }

    if len(valid_results) < min_valid_votes:
        error_result = {
            "label": None,
            "reason": f"Insufficient valid votes: {len(valid_results)}/{min_valid_votes}",
            "stage": DiagnosisStage.ERROR.value,
            "status": DiagnosisStatus.ERROR.value,
            "answer_correct": False,
            "used_model": None,
            "voting_details": voting_details,
            "usage_stats": aggregated_stats.to_dict(),
        }
        print(f"❌ {error_result['reason']}")
        return error_result

    label_counter = Counter(result["label"] for result in valid_results)
    voting_details["label_votes"] = dict(label_counter)
    highest_count = max(label_counter.values())
    winning_labels = {label for label, count in label_counter.items() if count == highest_count}
    voting_details["tie"] = len(winning_labels) > 1

    if len(winning_labels) == 1:
        winning_label = next(iter(winning_labels))
    else:
        primary_result = next(
            (result for result in valid_results if result["used_model"] == model and result["label"] in winning_labels),
            None,
        )
        if primary_result is None:
            primary_result = next(result for result in valid_results if result["label"] in winning_labels)
        winning_label = primary_result["label"]

    final_result = next(result.copy() for result in valid_results if result["label"] == winning_label)
    final_result["voting_details"] = voting_details
    final_result["usage_stats"] = aggregated_stats.to_dict()

    print(f"{'='*70}")
    print(f"📊 Voting summary")
    print(f"{'='*70}")
    print(f"🤖 Model order used: {used_models}")

    vote_count = label_counter[winning_label]
    print(f"🏆 Final label: {winning_label} (valid votes: {vote_count}/{len(valid_results)})")
    if voting_details["tie"]:
        print(f"⚠️  Vote tied; selected label {winning_label!r} using primary-model policy")

    aggregated_stats.print_summary()
    print(f"{'='*70}\n")

    return final_result
