from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_diagnosis.py"
SPEC = importlib.util.spec_from_file_location("run_diagnosis", MODULE_PATH)
run_diagnosis = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(run_diagnosis)


class RunDiagnosisTest(unittest.TestCase):
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
