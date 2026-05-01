from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parent / "run_diagnosis_longmemeval_s.py"
SPEC = importlib.util.spec_from_file_location("run_diagnosis_longmemeval_s", MODULE_PATH)
run_diagnosis_longmemeval_s = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(run_diagnosis_longmemeval_s)


class LongMemEvalTaskTest(unittest.TestCase):
    def test_iter_diagnosis_tasks_preserves_longmemeval_metadata(self):
        data = {
            "25": [
                {
                    "qa_question": "How many comedians did I watch?",
                    "qa_answer": "10",
                    "qa_response": "You watched 10.",
                    "qa_category": "single-session-user",
                    "question_id": "95bcc1c8",
                    "question_date": "2023/05/30 (Tue) 22:04",
                    "answer_session_ids": ["answer_cb742a61"],
                    "retrieved_answer_session_ids": ["answer_cb742a61"],
                    "retrieval_hit": True,
                    "person1": {
                        "name": "longmemeval_s_25",
                        "memories": [
                            {
                                "session_id": "s1",
                                "time_stamp": "2023/05/23 (Tue) 02:29",
                                "evidence_sentence": "session_id=s1",
                                "initial_results": [
                                    {"id": "m1", "event": "ADD", "memory": "User watched 10 comedians."}
                                ],
                                "update_chain": [
                                    {"id": "m1", "event": "ADD", "memory": "User watched 10 comedians."}
                                ],
                            }
                        ],
                    },
                    "person2": {"name": "", "memories": []},
                    "speaker_1_memories": [
                        {
                            "memory": "User watched 10 amateur comedians.",
                            "timestamp": "2023/05/23 (Tue) 02:29",
                            "session_id": "s1",
                            "score": 0.72,
                        }
                    ],
                    "speaker_2_memories": [],
                }
            ]
        }

        tasks = list(run_diagnosis_longmemeval_s.iter_diagnosis_tasks(data, processed_ids=set()))

        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        self.assertEqual(task.item_id, "longmemeval_s_25_95bcc1c8")
        self.assertEqual(task.sample_key, "25")
        self.assertEqual(task.question_index, 0)
        self.assertEqual(task.qa_data.question, "How many comedians did I watch?")
        self.assertEqual(task.memory_data.person2_memories, [])
        self.assertEqual(task.memory_data.speaker2_retrieval, [])
        self.assertEqual(
            task.output_metadata,
            {
                "sample_key": "25",
                "question_index": 0,
                "question_id": "95bcc1c8",
                "question_date": "2023/05/30 (Tue) 22:04",
                "answer_session_ids": ["answer_cb742a61"],
                "retrieved_answer_session_ids": ["answer_cb742a61"],
                "retrieval_hit": True,
            },
        )

    def test_iter_diagnosis_tasks_skips_processed_ids(self):
        data = {
            "25": [
                {
                    "qa_question": "Q",
                    "qa_answer": "A",
                    "qa_response": "R",
                    "question_id": "qid",
                    "person1": {"memories": []},
                    "person2": {"memories": []},
                }
            ]
        }

        tasks = list(
            run_diagnosis_longmemeval_s.iter_diagnosis_tasks(
                data,
                processed_ids={"longmemeval_s_25_qid"},
            )
        )

        self.assertEqual(tasks, [])


if __name__ == "__main__":
    unittest.main()
