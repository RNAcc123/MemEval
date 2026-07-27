from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_diagnosis_discussion.py"
SPEC = importlib.util.spec_from_file_location("run_diagnosis_discussion", MODULE_PATH)
run_diagnosis_discussion = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(run_diagnosis_discussion)


class DiscussionDiagnosisTest(unittest.TestCase):
    def test_stage_returns_error_when_valid_opinions_are_insufficient(self):
        invalid = {"is_sufficient": False, "label": None, "reason": "missing label"}

        with patch.object(run_diagnosis_discussion, "call_llm_api", return_value=invalid):
            result = run_diagnosis_discussion.discuss_stage(
                run_diagnosis_discussion.DiagnosisStage.MEMORY_EXTRACTION,
                run_diagnosis_discussion.QAData("q", "a", "r"),
                run_diagnosis_discussion.MemoryData(),
                ["model-a", "model-b", "model-c"],
                max_rounds=1,
            )

        self.assertEqual(result.status, run_diagnosis_discussion.DiagnosisStatus.ERROR)
        self.assertIsNone(result.final_label)
        self.assertIn("Insufficient valid model opinions", result.final_reason)

    def test_stage_can_reach_consensus_with_one_failed_model(self):
        valid = {"is_sufficient": False, "label": "3.1", "reason": "evidence was not retrieved"}
        invalid = {"is_sufficient": False, "label": None, "reason": "missing label"}

        with patch.object(
            run_diagnosis_discussion,
            "call_llm_api",
            side_effect=[valid, invalid, valid],
        ):
            result = run_diagnosis_discussion.discuss_stage(
                run_diagnosis_discussion.DiagnosisStage.MEMORY_RETRIEVAL,
                run_diagnosis_discussion.QAData("q", "a", "r"),
                run_diagnosis_discussion.MemoryData(),
                ["model-a", "model-b", "model-c"],
                max_rounds=1,
            )

        self.assertEqual(result.status, run_diagnosis_discussion.DiagnosisStatus.COMPLETED)
        self.assertTrue(result.consensus_reached)
        self.assertEqual(result.final_label, "3.1")


if __name__ == "__main__":
    unittest.main()
