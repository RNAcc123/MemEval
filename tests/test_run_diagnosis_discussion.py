from __future__ import annotations

import unittest
from unittest.mock import patch

from memeval.diagnosis.discussion import discuss_stage
from memeval.schema import DiagnosisStage, DiagnosisStatus, MemoryData, QAData


class DiscussionDiagnosisTest(unittest.TestCase):
    def test_stage_returns_error_when_valid_opinions_are_insufficient(self):
        invalid = {"is_sufficient": False, "label": None, "reason": "missing label"}

        with patch("memeval.diagnosis.discussion.call_llm_api", return_value=invalid):
            result = discuss_stage(
                DiagnosisStage.MEMORY_EXTRACTION,
                QAData("q", "a", "r"),
                MemoryData(),
                ["model-a", "model-b", "model-c"],
                max_rounds=1,
            )

        self.assertEqual(result.status, DiagnosisStatus.ERROR)
        self.assertIsNone(result.final_label)
        self.assertIn("Insufficient valid model opinions", result.final_reason)

    def test_stage_can_reach_consensus_with_one_failed_model(self):
        valid = {"is_sufficient": False, "label": "3.1", "reason": "evidence was not retrieved"}
        invalid = {"is_sufficient": False, "label": None, "reason": "missing label"}

        with patch(
            "memeval.diagnosis.discussion.call_llm_api",
            side_effect=[valid, invalid, valid],
        ):
            result = discuss_stage(
                DiagnosisStage.MEMORY_RETRIEVAL,
                QAData("q", "a", "r"),
                MemoryData(),
                ["model-a", "model-b", "model-c"],
                max_rounds=1,
            )

        self.assertEqual(result.status, DiagnosisStatus.COMPLETED)
        self.assertTrue(result.consensus_reached)
        self.assertEqual(result.final_label, "3.1")


if __name__ == "__main__":
    unittest.main()
