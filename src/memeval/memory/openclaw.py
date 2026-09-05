"""OpenCLAW native-memory backend driven through the ``openclaw`` CLI."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from memeval.memory.base import MemoryEvent, RetrievedMemory
from memeval.memory.locking import default_lock_root, interprocess_lock
from memeval.trace.collector import TraceCollector


MEMORY_WRITE_PROMPT = """Use OpenCLAW's native memory system only. Do not use Mem0 or any Mem0 plugin.

From the perspective of {speaker}, read this dated conversation session and write durable facts,
preferences, events, plans, relationships, and time-sensitive details into OpenCLAW memory files
such as MEMORY.md or memory/YYYY-MM-DD.md.

Preserve useful names, dates, and who did what. Do not answer questions from the conversation.
After updating memory, reply with a short summary of what you saved.

Timestamp: {timestamp}
Session: {session_id}

Conversation:
{conversation}
"""

MEMORY_FILE_CANDIDATES = ("MEMORY.md", "memory.md", "DREAMS.md")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-") or "unnamed"


def parse_json_output(stdout: str) -> Any:
    stdout = stdout.strip()
    if not stdout:
        return {}
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, char in enumerate(stdout):
            if char not in "[{":
                continue
            try:
                parsed, _ = decoder.raw_decode(stdout[index:])
            except json.JSONDecodeError:
                continue
            return parsed
        return {"raw": stdout}


def normalize_search_results(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        candidates = raw.get("results") or raw.get("memories") or raw.get("items") or []
    elif isinstance(raw, list):
        candidates = raw
    else:
        candidates = []
    results = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        memory = item.get("memory") or item.get("text") or item.get("content") or item.get("snippet") or ""
        results.append(
            {
                "memory": memory,
                "timestamp": item.get("timestamp") or item.get("created_at"),
                "score": round(item.get("score", item.get("rank_score", 0)) or 0, 2),
                "id": item.get("id") or item.get("key"),
                "file": item.get("file") or item.get("path"),
            }
        )
    return results


def message_content(message: dict[str, Any]) -> str:
    speaker = message.get("speaker") or message.get("role") or ""
    text = message.get("content", message.get("text", ""))
    pieces = [f"{speaker}: {text}".strip(": ").strip()]
    caption = message.get("blip_caption")
    if caption:
        pieces.append(f"Image caption: {caption}")
    query = message.get("query")
    if query:
        pieces.append(f"Image query: {query}")
    return "\n".join(piece for piece in pieces if piece)


def conversation_text(messages: list[dict[str, Any]]) -> str:
    return "\n".join(message_content(item) for item in messages if message_content(item).strip())


def speaker_of(messages: list[dict[str, Any]], fallback: str) -> str:
    for message in messages:
        speaker = message.get("speaker")
        if speaker:
            return str(speaker)
    return fallback


def event_id(user_id: str, session_id: str, relpath: str, content: str) -> str:
    raw = f"{user_id}\0{session_id}\0{relpath}\0{content}".encode("utf-8", errors="replace")
    return hashlib.sha1(raw).hexdigest()[:16]


def memory_files(workspace: Path) -> list[Path]:
    files = []
    for rel in MEMORY_FILE_CANDIDATES:
        path = workspace / rel
        if path.exists() and path.is_file():
            files.append(path)
    memory_dir = workspace / "memory"
    if memory_dir.exists():
        files.extend(path for path in memory_dir.rglob("*.md") if path.is_file())
    return sorted(files)


def snapshot_memory_files(workspace: Path) -> dict[str, str]:
    return {
        str(path.relative_to(workspace)): path.read_text(encoding="utf-8", errors="replace")
        for path in memory_files(workspace)
    }


def memory_events_from_diff(
    before: dict[str, str],
    after: dict[str, str],
    *,
    user_id: str,
    session_id: str,
    timestamp: str,
) -> list[dict[str, Any]]:
    events = []
    for relpath in sorted(set(before) | set(after)):
        old = before.get(relpath, "")
        new = after.get(relpath, "")
        if old == new:
            continue
        event = "ADD" if not old else "UPDATE"
        added = new[len(old) :].strip() if new.startswith(old) else new.strip()
        memory = added or new.strip()
        events.append(
            {
                "id": event_id(user_id, session_id, relpath, memory),
                "event": event,
                "memory": memory,
                "file": relpath,
                "timestamp": timestamp,
            }
        )
    if not events:
        events.append({"error": "No native OpenCLAW memory file changes"})
    return events


class OpenClawBackend:
    """Drive OpenCLAW native memory and emit MemEval trace events.

    Memory writes are performed by an agent turn and observed by diffing the
    workspace memory files, since the CLI has no structured write-event output.
    """

    name = "openclaw"

    def __init__(
        self,
        *,
        workspace_root: Path,
        openclaw_bin: str = "openclaw",
        agent: str = "main",
        agent_model: str = "",
        openclaw_profile: str = "",
        session_prefix: str = "",
        timeout: float = 300.0,
    ):
        self.workspace_root = Path(workspace_root)
        self.openclaw_bin = openclaw_bin
        self.agent = agent
        self.agent_model = agent_model
        self.openclaw_profile = openclaw_profile
        self.session_prefix = session_prefix
        self.timeout = timeout

    def workspace_for(self, subject: str) -> Path:
        return self.workspace_root / safe_name(subject)

    def _command_profile(self, subject: str) -> str:
        return self.openclaw_profile or f"memeval-openclaw-{safe_name(subject)}"

    def _command_env(self, profile: str) -> dict[str, str]:
        env = os.environ.copy()
        env["OPENCLAW_PROFILE"] = profile
        return env

    def _with_profile(self, cmd: list[str]) -> list[str]:
        if not self.openclaw_profile:
            return cmd
        if len(cmd) > 2 and cmd[1] == "--profile":
            return cmd
        return [cmd[0], "--profile", self.openclaw_profile, *cmd[1:]]

    def _lock_path(self) -> Path:
        profile_part = safe_name(self.openclaw_profile) if self.openclaw_profile else "default"
        return default_lock_root() / f"memeval-openclaw-{profile_part}-{safe_name(self.agent)}-sessions.lock"

    def _run(self, cmd: list[str], *, profile: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=self._command_env(profile),
            text=True,
            capture_output=True,
            timeout=self.timeout,
            check=False,
        )

    def _run_in_workspace(self, cmd: list[str], *, subject: str, workspace: Path) -> subprocess.CompletedProcess[str]:
        """Run a CLI command under the shared agent lock, after ensuring setup."""
        profile = self._command_profile(subject)
        with interprocess_lock(self._lock_path()):
            setup = self._run(
                self._with_profile([cmd[0], "setup", "--workspace", str(workspace)]),
                profile=profile,
                cwd=workspace,
            )
            if setup.returncode != 0:
                return setup
            return self._run(self._with_profile(cmd), profile=profile, cwd=workspace)

    def _agent_command(self, prompt: str, session_id: str) -> list[str]:
        cmd = [
            self.openclaw_bin,
            "agent",
            "--agent",
            self.agent,
            "--local",
            "-m",
            prompt,
            "--session-id",
            session_id,
            "--timeout",
            str(int(self.timeout)),
        ]
        if self.agent_model:
            cmd.extend(["--model", self.agent_model])
        return cmd

    def reset(self, subject: str) -> None:
        workspace = self.workspace_for(subject)
        if workspace.exists():
            shutil.rmtree(workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        result = self._run(
            self._with_profile([self.openclaw_bin, "setup", "--workspace", str(workspace)]),
            profile=self._command_profile(subject),
        )
        if result.returncode != 0:
            raise RuntimeError(f"openclaw setup failed: {result.stderr or result.stdout}")

    def add_session(
        self,
        subject: str,
        messages: list[dict[str, Any]],
        metadata: dict[str, Any],
        *,
        collector: TraceCollector | None = None,
        turn_id: str | None = None,
        parent_event_id: str | None = None,
    ) -> list[MemoryEvent]:
        session_id = str(metadata.get("session_id", ""))
        timestamp = str(metadata.get("timestamp", ""))
        workspace = self.workspace_for(subject)
        event_ref = collector.start_event(
            "memory_update", "add", turn_id=turn_id, parent_event_id=parent_event_id,
            input={"subject_id": subject, "messages": messages, "metadata": metadata},
        ) if collector is not None else None
        try:
            before = snapshot_memory_files(workspace)
            prompt = MEMORY_WRITE_PROMPT.format(
                speaker=speaker_of(messages, subject),
                timestamp=timestamp,
                session_id=session_id,
                conversation=conversation_text(messages),
            )
            cli_session = "-".join(
                part for part in [
                    safe_name(self.session_prefix) if self.session_prefix else "",
                    safe_name(subject),
                    safe_name(session_id),
                ] if part
            )
            result = self._run_in_workspace(
                self._agent_command(prompt, cli_session), subject=subject, workspace=workspace
            )
            if result.returncode != 0:
                raise RuntimeError(f"openclaw agent memory write failed: {result.stderr or result.stdout}")
            after = snapshot_memory_files(workspace)
            raw_events = memory_events_from_diff(
                before, after, user_id=subject, session_id=session_id, timestamp=timestamp
            )
        except Exception as exc:
            if collector is not None and event_ref is not None:
                collector.fail_event(event_ref, exc)
            raise
        normalized = [
            MemoryEvent(
                event=str(item.get("event", "ADD")), memory=item.get("memory", ""),
                memory_id=item.get("id"), session_id=session_id,
                timestamp=timestamp, metadata=dict(item),
            )
            for item in raw_events
            if "error" not in item
        ]
        if collector is not None and event_ref is not None:
            collector.finish_event(
                event_ref,
                output={"events": [
                    {"memory_id": item.memory_id, "event": item.event, "memory": item.memory,
                     "session_id": item.session_id, "timestamp": item.timestamp}
                    for item in normalized
                ]},
                raw={"response": {"results": raw_events}, "stdout": result.stdout.strip()},
            )
        return normalized

    def search(
        self,
        subject: str,
        query: str,
        top_k: int,
        *,
        collector: TraceCollector | None = None,
        turn_id: str | None = None,
        parent_event_id: str | None = None,
    ) -> list[RetrievedMemory]:
        workspace = self.workspace_for(subject)
        event_ref = collector.start_event(
            "memory_retrieval", "search", turn_id=turn_id, parent_event_id=parent_event_id,
            input={"subject_id": subject, "query": query, "top_k": top_k},
        ) if collector is not None else None
        try:
            result = self._run_in_workspace(
                [
                    self.openclaw_bin, "memory", "search",
                    "--query", query,
                    "--max-results", str(top_k),
                    "--agent", self.agent,
                    "--json",
                ],
                subject=subject,
                workspace=workspace,
            )
            if result.returncode != 0:
                raise RuntimeError(f"openclaw memory search failed: {result.stderr or result.stdout}")
            raw = parse_json_output(result.stdout)
            items = normalize_search_results(raw)
        except Exception as exc:
            if collector is not None and event_ref is not None:
                collector.fail_event(event_ref, exc)
            raise
        normalized = [
            RetrievedMemory(
                memory=item.get("memory", ""), score=round(item.get("score", 0) or 0, 2),
                memory_id=item.get("id"), session_id=None,
                timestamp=item.get("timestamp"), metadata=dict(item),
            )
            for item in items
        ]
        if collector is not None and event_ref is not None:
            collector.finish_event(
                event_ref,
                output={"candidates": [
                    {"memory_id": item.memory_id, "memory": item.memory, "score": item.score,
                     "rank": rank, "selected": False, "session_id": item.session_id,
                     "timestamp": item.timestamp}
                    for rank, item in enumerate(normalized, start=1)
                ]},
                raw={"response": raw if isinstance(raw, dict) else {"results": raw}},
            )
        return normalized
