from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[2] / "eval" / "openclaw-memory" / "run_locomo_openclaw_trace.py"
SPEC = importlib.util.spec_from_file_location("run_locomo_openclaw_trace", MODULE_PATH)
run_locomo_openclaw_trace = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(run_locomo_openclaw_trace)


class OpenClawNativeMemoryTest(unittest.TestCase):
    def test_memory_events_from_diff_records_native_memory_file_update(self):
        before = {"MEMORY.md": "Known facts\n"}
        after = {"MEMORY.md": "Known facts\n- Alice adopted a cat yesterday.\n"}

        events = run_locomo_openclaw_trace.memory_events_from_diff(
            before,
            after,
            user_id="Alice_0",
            session_id="session_1",
            timestamp="2024-01-02",
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event"], "UPDATE")
        self.assertEqual(events[0]["file"], "MEMORY.md")
        self.assertEqual(events[0]["memory"], "- Alice adopted a cat yesterday.")
        self.assertEqual(events[0]["timestamp"], "2024-01-02")

    def test_normalize_search_results_accepts_openclaw_json_shapes(self):
        raw = {
            "results": [
                {
                    "id": "m1",
                    "text": "Alice adopted a cat yesterday.",
                    "score": 0.8123,
                    "path": "MEMORY.md",
                    "created_at": "2024-01-02",
                }
            ]
        }

        results = run_locomo_openclaw_trace.normalize_search_results(raw)

        self.assertEqual(
            results,
            [
                {
                    "memory": "Alice adopted a cat yesterday.",
                    "timestamp": "2024-01-02",
                    "score": 0.81,
                    "id": "m1",
                    "file": "MEMORY.md",
                }
            ],
        )

    def test_memory_prompt_uses_native_openclaw_not_openclaw_mem0_command(self):
        prompt = run_locomo_openclaw_trace.MEMORY_WRITE_PROMPT.lower()
        self.assertIn("native memory", prompt)
        self.assertNotIn("openclaw mem0", prompt)

    def test_command_env_sets_openclaw_profile_without_rewriting_provider_credentials(self):
        with patch.dict(
            os.environ,
            {
                "DEEPSEEK_API_KEY": "sk-deepseek",
                "DEEPSEEK_API_URL": "https://api.deepseek.com",
                "OPENAI_API_KEY": "sk-openai",
            },
            clear=True,
        ):
            env = run_locomo_openclaw_trace.command_env("memeval-profile")

        self.assertEqual(env["OPENCLAW_PROFILE"], "memeval-profile")
        self.assertEqual(env["DEEPSEEK_API_KEY"], "sk-deepseek")
        self.assertEqual(env["OPENAI_API_KEY"], "sk-openai")
        self.assertNotIn("OPENAI_BASE_URL", env)

    def test_openclaw_agent_command_uses_native_deepseek_model_override(self):
        command = run_locomo_openclaw_trace.build_openclaw_agent_command(
            "openclaw",
            "main",
            "Save memory",
            "sample-session",
            "deepseek/deepseek-chat",
            300.0,
        )

        self.assertIn("--model", command)
        self.assertIn("deepseek/deepseek-chat", command)
        self.assertIn("--timeout", command)
        self.assertIn("300", command)

    def test_parse_args_uses_openclaw_agent_model_loaded_from_env_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text("OPENCLAW_AGENT_MODEL=openai/gpt-4o-mini\n", encoding="utf-8")

            with patch.dict(os.environ, {}, clear=True):
                run_locomo_openclaw_trace.load_cli_env_file(["--env-file", str(env_file), "--dry-run"])
                args = run_locomo_openclaw_trace.parse_args(["--env-file", str(env_file), "--dry-run"])

        self.assertEqual(args.agent_model, "openai/gpt-4o-mini")


if __name__ == "__main__":
    unittest.main()
