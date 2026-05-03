from __future__ import annotations

import importlib.util
import os
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
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

    def test_parse_json_output_ignores_openclaw_log_prefixes(self):
        stdout = """[proxy] routing process HTTP traffic through external proxy http://127.0.0.1:7890
[secrets] memory search: gateway secrets.resolve unavailable; resolved command secrets locally.
{
  "results": [
    {
      "text": "Alice adopted a cat yesterday.",
      "score": 0.9
    }
  ]
}
"""

        parsed = run_locomo_openclaw_trace.parse_json_output(stdout)

        self.assertEqual(parsed["results"][0]["text"], "Alice adopted a cat yesterday.")

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

    def test_parse_args_does_not_use_openclaw_agent_model_loaded_from_env_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text("OPENCLAW_AGENT_MODEL=openai/gpt-4o-mini\n", encoding="utf-8")

            with patch.dict(os.environ, {}, clear=True):
                run_locomo_openclaw_trace.load_cli_env_file(["--env-file", str(env_file), "--dry-run"])
                args = run_locomo_openclaw_trace.parse_args(["--env-file", str(env_file), "--dry-run"])

        self.assertEqual(args.agent_model, "")

    def test_with_openclaw_profile_inserts_global_profile_option(self):
        command = run_locomo_openclaw_trace.with_openclaw_profile(
            ["openclaw", "agent", "--agent", "main"],
            "memeval-worker-0",
        )

        self.assertEqual(command[:3], ["openclaw", "--profile", "memeval-worker-0"])
        self.assertEqual(command[3:], ["agent", "--agent", "main"])

    def test_openclaw_lock_path_isolated_by_profile(self):
        lock_a = run_locomo_openclaw_trace.openclaw_agent_session_lock_path("main", "worker-a")
        lock_b = run_locomo_openclaw_trace.openclaw_agent_session_lock_path("main", "worker-b")

        self.assertNotEqual(lock_a, lock_b)
        self.assertIn("worker-a", str(lock_a))
        self.assertIn("worker-b", str(lock_b))

    def test_answer_question_serializes_openclaw_agent_session_writes(self):
        active_commands = 0
        max_active_commands = 0

        def fake_run_command(*args, **kwargs):
            nonlocal active_commands, max_active_commands
            active_commands += 1
            max_active_commands = max(max_active_commands, active_commands)
            time.sleep(0.02)
            active_commands -= 1
            return SimpleNamespace(returncode=0, stdout="answer", stderr="")

        with patch.object(run_locomo_openclaw_trace, "run_command", side_effect=fake_run_command):
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(
                        run_locomo_openclaw_trace.answer_question,
                        "openclaw",
                        "",
                        "main",
                        "",
                        "run",
                        "Alice",
                        Path("."),
                        f"question {index}",
                        [],
                        [],
                        300.0,
                    )
                    for index in range(2)
                ]
                for future in futures:
                    self.assertEqual(future.result(), "answer")

        self.assertEqual(max_active_commands, 1)


if __name__ == "__main__":
    unittest.main()
