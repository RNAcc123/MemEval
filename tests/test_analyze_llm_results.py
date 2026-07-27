from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_llm_results.py"
SPEC = importlib.util.spec_from_file_location("analyze_llm_results", MODULE_PATH)
analyze_llm_results = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(analyze_llm_results)


def test_collect_stats_excludes_error_records_and_votes(tmp_path):
    input_file = tmp_path / "diagnoses.json"
    input_file.write_text(
        json.dumps(
            [
                {
                    "qa_category": 1,
                    "label": "1.1",
                    "status": "completed",
                    "voting_details": {
                        "individual_results": [
                            {"used_model": "judge-a", "label": "1.1", "status": "completed"},
                            {"used_model": "judge-b", "label": None, "status": "error"},
                        ]
                    },
                },
                {
                    "qa_category": 1,
                    "label": None,
                    "status": "error",
                },
            ]
        ),
        encoding="utf-8",
    )

    final_stats, model_stats, labels, coverage = analyze_llm_results.collect_stats([str(input_file)])

    assert final_stats[1]["total_items"] == 1
    assert final_stats[1]["label_counts"] == {"1.1": 1}
    assert model_stats["judge-a"][1]["total_items"] == 1
    assert "judge-b" not in model_stats
    assert labels == ["1.1"]
    assert coverage == {
        "completed_records": 1,
        "excluded_error_records": 1,
        "excluded_error_votes": 1,
    }
