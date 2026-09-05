from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from memeval.diagnosis.llm import is_retryable_provider_error
from memeval.diagnosis.single import analyze_qa_pair
from memeval.diagnosis.voting import analyze_qa_pair_with_voting
from memeval.schema import (
    DiagnosisResult,
    DiagnosisStage,
    DiagnosisStatus,
    MemoryData,
    QAData,
    StageResult,
)
from memeval.schema.validation import InvalidJudgeResponse, validate_judge_response


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_diagnosis.py"
SPEC = importlib.util.spec_from_file_location("run_diagnosis", MODULE_PATH)
run_diagnosis = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(run_diagnosis)


class RunDiagnosisTest(unittest.TestCase):
    def test_validate_judge_response_rejects_missing_failure_label(self):
        with self.assertRaises(InvalidJudgeResponse):
            validate_judge_response(
                DiagnosisStage.MEMORY_EXTRACTION,
                {"is_sufficient": False, "label": None, "reason": "Missing memory."},
            )

    def test_analyze_qa_pair_stops_when_a_stage_has_execution_error(self):
        qa_data = QAData(question="q", answer="a", response="r")
        memory_data = MemoryData()
        stage_error = StageResult(
            stage_passed=False,
            label=None,
            reason="provider unavailable",
            stage=DiagnosisStage.ERROR,
            status=DiagnosisStatus.ERROR,
        )

        with patch("memeval.diagnosis.single.stage0_consistency_check", return_value=stage_error):
            with patch("memeval.diagnosis.single.stage1_memory_extraction") as stage1:
                result = analyze_qa_pair(qa_data, memory_data)

        self.assertEqual(result.status, DiagnosisStatus.ERROR)
        self.assertEqual(result.stage, DiagnosisStage.ERROR)
        self.assertIsNone(result.label)
        stage1.assert_not_called()

    def test_voting_excludes_failed_judgments(self):
        completed = DiagnosisResult(
            label="3.1",
            reason="retrieval missed evidence",
            stage=DiagnosisStage.MEMORY_RETRIEVAL,
        )
        failed = DiagnosisResult(
            label=None,
            reason="provider unavailable",
            stage=DiagnosisStage.ERROR,
            status=DiagnosisStatus.ERROR,
        )

        with patch(
            "memeval.diagnosis.voting.analyze_qa_pair",
            side_effect=[completed, failed, completed],
        ):
            result = analyze_qa_pair_with_voting(
                qa_question="q",
                qa_answer="a",
                qa_response="r",
                subjects=[],
                num_votes=3,
            )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["label"], "3.1")
        self.assertEqual(result["voting_details"]["valid_votes"], 2)
        self.assertEqual(result["voting_details"]["failed_votes"], 1)
        self.assertEqual(result["voting_details"]["label_votes"], {"3.1": 2})

    def test_voting_returns_error_when_valid_votes_are_insufficient(self):
        failed = DiagnosisResult(
            label=None,
            reason="provider unavailable",
            stage=DiagnosisStage.ERROR,
            status=DiagnosisStatus.ERROR,
        )
        completed = DiagnosisResult(
            label=None,
            reason="answer is correct",
            stage=DiagnosisStage.CONSISTENCY_CHECK,
            answer_correct=True,
        )

        with patch(
            "memeval.diagnosis.voting.analyze_qa_pair",
            side_effect=[failed, failed, completed],
        ):
            result = analyze_qa_pair_with_voting(
                qa_question="q",
                qa_answer="a",
                qa_response="r",
                subjects=[],
                num_votes=3,
            )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["stage"], "error")
        self.assertIsNone(result["label"])
        self.assertEqual(result["voting_details"]["valid_votes"], 1)

    def test_retryable_provider_error_uses_sdk_type_name(self):
        class RateLimitError(Exception):
            pass

        self.assertTrue(is_retryable_provider_error(RateLimitError()))
        self.assertFalse(is_retryable_provider_error(ValueError("invalid configuration")))

    def test_process_single_file_can_process_qa_items_concurrently(self):
        data = {
            "conv-1": [
                {
                    "qa_question": "q1",
                    "qa_answer": "a1",
                    "qa_response": "r1",
                    "person1": {"memories": []},
                    "person2": {"memories": []},
                },
                {
                    "qa_question": "q2",
                    "qa_answer": "a2",
                    "qa_response": "r2",
                    "person1": {"memories": []},
                    "person2": {"memories": []},
                },
            ]
        }

        def fake_voting(**_kwargs):
            return {
                "label": "4.1",
                "reason": "mocked",
                "usage_stats": {
                    "total_calls": 1,
                    "total_latency_seconds": 0.1,
                    "total_prompt_tokens": 2,
                    "total_completion_tokens": 3,
                    "total_tokens": 5,
                    "call_details": [],
                },
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input.json"
            output_file = Path(tmpdir) / "output.json"
            input_file.write_text(json.dumps(data), encoding="utf-8")

            with patch.object(run_diagnosis, "analyze_qa_pair_with_voting", side_effect=fake_voting):
                count, stats = run_diagnosis.process_single_file(
                    input_file=str(input_file),
                    output_file=str(output_file),
                    model="deepseek",
                    use_voting=True,
                    num_votes=3,
                    qa_threads=2,
                )

            results = json.loads(output_file.read_text(encoding="utf-8"))

        self.assertEqual(count, 2)
        self.assertEqual(stats.total_calls, 2)
        self.assertEqual({item["conv_id_question_id"] for item in results}, {"conv-1_0", "conv-1_1"})


if __name__ == "__main__":
    unittest.main()
