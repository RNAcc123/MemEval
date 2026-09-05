from __future__ import annotations

import json

from memeval.analysis.label_matching import (
    analyze_model_label_matching_exact,
    analyze_model_label_matching_strict,
    collect_phase_confusion_voting_final,
    is_completed_result,
)


def _write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def test_analyze_model_label_matching_strict_and_exact(tmp_path):
    human_file = tmp_path / "human.json"
    llm_file = tmp_path / "llm.json"

    _write_json(
        human_file,
        {
            "conv1": {"qa_category": "1", "labels": ["1.1"]},
            "conv2": {"qa_category": "2", "labels": ["2.2"]},
            "conv3": {"qa_category": "5", "labels": ["5.1"]},  # excluded category
        },
    )
    _write_json(
        llm_file,
        [
            {
                "conv_id_question_id": "conv1",
                "status": "completed",
                "label": "1.1",
                "voting_details": {
                    "individual_results": [
                        {"used_model": "judge-a", "label": "1.1", "status": "completed"},
                        {"used_model": "judge-b", "label": "2.1", "status": "completed"},
                    ]
                },
            },
            {
                "conv_id_question_id": "conv2",
                "status": "completed",
                "label": None,
                "voting_details": {
                    "individual_results": [
                        {"used_model": "judge-a", "label": None, "status": "completed"},
                    ]
                },
            },
        ],
    )

    stats_phase = analyze_model_label_matching_strict(str(human_file), str(llm_file))
    # judge-a matched conv1's phase (1 == 1) and did not match conv2 (EMPTY == EMPTY, so it matches)
    assert stats_phase["judge-a"]["1"]["1"] == {"total": 1, "matched": 1}
    assert stats_phase["judge-b"]["1"]["1"] == {"total": 1, "matched": 0}
    assert stats_phase["voting_final"]["1"]["1"] == {"total": 1, "matched": 1}
    assert stats_phase["voting_final"]["2"]["2"] == {"total": 1, "matched": 0}

    stats_exact = analyze_model_label_matching_exact(str(human_file), str(llm_file))
    assert stats_exact["judge-a"]["1"]["1.1"] == {"total": 1, "matched": 1}
    assert stats_exact["judge-b"]["1"]["1.1"] == {"total": 1, "matched": 0}


def test_collect_phase_confusion_voting_final(tmp_path):
    human_file = tmp_path / "human.json"
    llm_file = tmp_path / "llm.json"

    _write_json(
        human_file,
        {
            "conv1": {"qa_category": "1", "labels": ["1.1"]},
            "conv2": {"qa_category": "2", "labels": ["2.2"]},
        },
    )
    _write_json(
        llm_file,
        [
            {"conv_id_question_id": "conv1", "status": "completed", "label": "1.3"},
            {"conv_id_question_id": "conv2", "status": "completed", "label": "3.1"},
        ],
    )

    conf = collect_phase_confusion_voting_final(str(human_file), str(llm_file))
    assert conf["1"]["1"] == 1
    assert conf["2"]["3"] == 1


def test_is_completed_result_defaults_to_completed_for_legacy_records():
    assert is_completed_result({"label": "1.1"}) is True
    assert is_completed_result({"status": "error"}) is False
    assert is_completed_result("not-a-dict") is False
