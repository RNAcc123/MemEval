from __future__ import annotations

import importlib.util
import threading
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[2] / "eval" / "locomo-memory" / "run_locomo_mem0_trace.py"
SPEC = importlib.util.spec_from_file_location("run_locomo_mem0_trace", MODULE_PATH)
run_locomo_mem0_trace = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(run_locomo_mem0_trace)


class EvidenceMemoryTraceTest(unittest.TestCase):
    def test_maps_evidence_sentence_to_session_memory_path(self):
        conversation = {
            "session_1": [
                {"dia_id": "d1", "speaker": "Alice", "text": "I adopted a cat yesterday."},
                {"dia_id": "d2", "speaker": "Bob", "text": "That is wonderful."},
            ],
            "session_1_date_time": "2024-01-02",
        }
        person = {
            "name": "Alice_0",
            "memories": [
                {
                    "session_id": "session_1",
                    "time_stamp": "2024-01-02",
                    "initial_results": [
                        {"id": "m1", "event": "ADD", "memory": "Alice adopted a cat yesterday."}
                    ],
                    "update_chain": [
                        {"id": "m1", "event": "ADD", "memory": "Alice adopted a cat yesterday."},
                        {
                            "id": "m1",
                            "event": "UPDATE",
                            "memory": "Alice adopted a cat on 2024-01-01.",
                            "previous_memory": "Alice adopted a cat yesterday.",
                        },
                    ],
                }
            ],
        }

        traces = run_locomo_mem0_trace.build_evidence_memory_traces(conversation, person, {"d1"})

        self.assertEqual(
            traces,
            [
                {
                    "dia_id": "d1",
                    "speaker": "Alice",
                    "evidence_sentence": "Alice: I adopted a cat yesterday.",
                    "session_id": "session_1",
                    "time_stamp": "2024-01-02",
                    "initial_results": [
                        {"id": "m1", "event": "ADD", "memory": "Alice adopted a cat yesterday."}
                    ],
                    "initial_memory_ids": ["m1"],
                    "memory_update_paths": [
                        {
                            "memory_id": "m1",
                            "initial_results": [
                                {"id": "m1", "event": "ADD", "memory": "Alice adopted a cat yesterday."}
                            ],
                            "update_chain": [
                                {"id": "m1", "event": "ADD", "memory": "Alice adopted a cat yesterday."},
                                {
                                    "id": "m1",
                                    "event": "UPDATE",
                                    "memory": "Alice adopted a cat on 2024-01-01.",
                                    "previous_memory": "Alice adopted a cat yesterday.",
                                },
                            ],
                        }
                    ],
                    "update_chain": [
                        {"id": "m1", "event": "ADD", "memory": "Alice adopted a cat yesterday."},
                        {
                            "id": "m1",
                            "event": "UPDATE",
                            "memory": "Alice adopted a cat on 2024-01-01.",
                            "previous_memory": "Alice adopted a cat yesterday.",
                        },
                    ],
                }
            ],
        )


class LlmConfigTest(unittest.TestCase):
    def test_build_llm_config_uses_api_key_env_and_base_url(self):
        environ = {"DASHSCOPE_API_KEY": "secret-key"}

        config = run_locomo_mem0_trace.build_llm_config(
            "qwen3.5-35b-a3b",
            api_key_env="DASHSCOPE_API_KEY",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            environ=environ,
        )

        self.assertEqual(
            config,
            {
                "model": "qwen3.5-35b-a3b",
                "temperature": 0.0,
                "api_key": "secret-key",
                "openai_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            },
        )


class EmbedderConfigTest(unittest.TestCase):
    def test_build_embedder_config_uses_api_key_env_and_base_url(self):
        environ = {"DASHSCOPE_API_KEY": "secret-key"}

        config = run_locomo_mem0_trace.build_embedder_config(
            "text-embedding-v4",
            api_key_env="DASHSCOPE_API_KEY",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            environ=environ,
        )

        self.assertEqual(
            config,
            {
                "model": "text-embedding-v4",
                "api_key": "secret-key",
                "openai_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            },
        )


class VectorStoreConfigTest(unittest.TestCase):
    def test_text_embedding_v4_uses_1024_vector_dimensions(self):
        config = run_locomo_mem0_trace.build_vector_store_config(
            Path("/tmp/mem0-store"),
            "text-embedding-v4",
        )

        self.assertEqual(config["config"]["embedding_model_dims"], 1024)


class ResumeCompletionTest(unittest.TestCase):
    def test_error_record_is_not_completed_for_resume(self):
        records = [
            {
                "qa_question": "",
                "qa_answer": "",
                "qa_response": "",
                "qa_category": -1,
                "question_id": "conv-26_error",
                "error": "Connection error.",
            }
        ]

        self.assertFalse(run_locomo_mem0_trace.records_completed(records))


class ConcurrentAddTest(unittest.TestCase):
    def test_adds_two_speaker_perspectives_in_same_session_concurrently(self):
        class FakeClient:
            def __init__(self):
                self.barrier = threading.Barrier(2, timeout=1)
                self.max_active_adds = 0
                self.active_adds = 0
                self.lock = threading.Lock()

            def delete_all(self, user_id):
                return None

            def add(self, messages, user_id, metadata):
                with self.lock:
                    self.active_adds += 1
                    self.max_active_adds = max(self.max_active_adds, self.active_adds)
                try:
                    self.barrier.wait()
                    return [{"id": user_id, "event": "ADD", "memory": f"memory for {user_id}"}]
                finally:
                    with self.lock:
                        self.active_adds -= 1

        item = {
            "sample_id": "sample-1",
            "conversation": {
                "speaker_a": "Alice",
                "speaker_b": "Bob",
                "session_1": [
                    {"dia_id": "d1", "speaker": "Alice", "text": "I adopted a cat."},
                    {"dia_id": "d2", "speaker": "Bob", "text": "That is wonderful."},
                ],
                "session_1_date_time": "2024-01-02",
            },
        }
        client = FakeClient()

        person1, person2 = run_locomo_mem0_trace.add_conversation_memories(client, item, 0)

        self.assertEqual(client.max_active_adds, 2)
        self.assertEqual(person1["memories"][0]["initial_results"][0]["id"], "Alice_0")
        self.assertEqual(person2["memories"][0]["initial_results"][0]["id"], "Bob_0")


if __name__ == "__main__":
    unittest.main()
