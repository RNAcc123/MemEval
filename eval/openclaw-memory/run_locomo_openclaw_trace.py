#!/usr/bin/env python3
"""Run OpenCLAW native memory over LoCoMo and write MemEval-compatible traces."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_DATASET = Path("/share/project/chenchen/data/locomo/locomo10.json")
DEFAULT_OUTPUT_DIR = Path("/share/project/chenchen/code/MemEval/data/input/openclaw_mem/locomo10")
DEFAULT_ENV_FILE = Path("/share/project/chenchen/code/MemEval/.env")
DEFAULT_WORKSPACE_ROOT = Path("/share/project/chenchen/code/MemEval/data/input/openclaw_mem/locomo10/workspaces")
DEFAULT_OPENCLAW_BIN = "openclaw"
DEFAULT_AGENT = "main"
DEFAULT_AGENT_MODEL = "deepseek/deepseek-chat"


ANSWER_PROMPT = """Answer the question using only OpenCLAW native memories and the retrieved memory snippets below.

Rules:
1. Use only the provided memories.
2. If memories conflict, prefer the more recent timestamp or memory file content.
3. If the answer requires date arithmetic, compute it from timestamps.
4. Answer concisely.

Speaker 1 memories:
{speaker_1_memories}

Speaker 2 memories:
{speaker_2_memories}

Question:
{question}
"""


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE_ROOT)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--openclaw-bin", default=DEFAULT_OPENCLAW_BIN)
    parser.add_argument("--agent", default=DEFAULT_AGENT)
    parser.add_argument("--agent-model", default=os.getenv("OPENCLAW_AGENT_MODEL", DEFAULT_AGENT_MODEL))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=10)
    parser.add_argument("--part-size", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    return parser.parse_args(argv)


def load_cli_env_file(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    args, _ = parser.parse_known_args(argv)
    load_env_file(args.env_file)


def load_env_file(env_file: Path) -> None:
    if not env_file.exists():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def load_dataset(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("LoCoMo dataset must be a list")
    for idx, item in enumerate(data):
        for key in ["sample_id", "conversation", "qa"]:
            if key not in item:
                raise ValueError(f"Sample {idx} missing key: {key}")
        conversation = item["conversation"]
        if "speaker_a" not in conversation or "speaker_b" not in conversation:
            raise ValueError(f"Sample {idx} missing speaker_a/speaker_b")
        if not session_keys(conversation):
            raise ValueError(f"Sample {idx} has no session_N entries")
    return data


def session_keys(conversation: dict[str, Any]) -> list[str]:
    keys = [
        key
        for key, value in conversation.items()
        if re.fullmatch(r"session_\d+", key) and isinstance(value, list)
    ]
    return sorted(keys, key=lambda key: int(key.split("_")[1]))


def validate_range(start: int, end: int, total: int) -> None:
    if start < 0 or end < start or end > total:
        raise ValueError(f"Invalid range start={start}, end={end}, dataset size={total}")


def safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return safe or "user"


def profile_for(user_id: str) -> str:
    return f"memeval-openclaw-{safe_name(user_id)}"


def command_env(profile: str) -> dict[str, str]:
    env = os.environ.copy()
    env["OPENCLAW_PROFILE"] = profile
    return env


def build_openclaw_agent_command(
    openclaw_bin: str,
    agent: str,
    prompt: str,
    session_id: str,
    agent_model: str,
    timeout: float,
) -> list[str]:
    return [
        openclaw_bin,
        "agent",
        "--agent",
        agent,
        "--local",
        "-m",
        prompt,
        "--session-id",
        session_id,
        "--model",
        agent_model,
        "--timeout",
        str(int(timeout)),
    ]


def run_command(
    cmd: list[str],
    *,
    profile: str,
    timeout: float,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=command_env(profile),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def ensure_openclaw_workspace(openclaw_bin: str, profile: str, workspace: Path, timeout: float) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    result = run_command(
        [openclaw_bin, "setup", "--workspace", str(workspace)],
        profile=profile,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"openclaw setup failed: {result.stderr or result.stdout}")


def reset_workspace(workspace: Path) -> None:
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)


def memory_files(workspace: Path) -> list[Path]:
    files = []
    for rel in ["MEMORY.md", "memory.md", "DREAMS.md"]:
        path = workspace / rel
        if path.exists() and path.is_file():
            files.append(path)
    memory_dir = workspace / "memory"
    if memory_dir.exists():
        files.extend(path for path in memory_dir.rglob("*.md") if path.is_file())
    return sorted(files)


def snapshot_memory_files(workspace: Path) -> dict[str, str]:
    snapshot = {}
    for path in memory_files(workspace):
        snapshot[str(path.relative_to(workspace))] = path.read_text(encoding="utf-8", errors="replace")
    return snapshot


def event_id(user_id: str, session_id: str, relpath: str, content: str) -> str:
    raw = f"{user_id}\0{session_id}\0{relpath}\0{content}".encode("utf-8", errors="replace")
    return hashlib.sha1(raw).hexdigest()[:16]


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


def message_content(chat: dict[str, Any]) -> str:
    pieces = [f"{chat.get('speaker', '')}: {chat.get('text', '')}".strip()]
    caption = chat.get("blip_caption")
    if caption:
        pieces.append(f"Image caption: {caption}")
    query = chat.get("query")
    if query:
        pieces.append(f"Image query: {query}")
    return "\n".join(piece for piece in pieces if piece)


def conversation_text(chats: list[dict[str, Any]]) -> str:
    return "\n".join(message_content(chat) for chat in chats if message_content(chat).strip())


def normalize_add_events(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    initial_results = [event for event in events if event.get("event") == "ADD"]
    if not initial_results:
        initial_results = events
    return initial_results, events


def add_session_native_memory(
    openclaw_bin: str,
    agent: str,
    agent_model: str,
    user_id: str,
    workspace: Path,
    session_id: str,
    timestamp: str,
    speaker: str,
    chats: list[dict[str, Any]],
    timeout: float,
) -> dict[str, Any]:
    before = snapshot_memory_files(workspace)
    prompt = MEMORY_WRITE_PROMPT.format(
        speaker=speaker,
        timestamp=timestamp,
        session_id=session_id,
        conversation=conversation_text(chats),
    )
    result = run_command(
        build_openclaw_agent_command(
            openclaw_bin,
            agent,
            prompt,
            f"{safe_name(user_id)}-{safe_name(session_id)}",
            agent_model,
            timeout,
        ),
        profile=profile_for(user_id),
        timeout=timeout,
        cwd=workspace,
    )
    if result.returncode != 0:
        raise RuntimeError(f"openclaw agent memory write failed: {result.stderr or result.stdout}")
    after = snapshot_memory_files(workspace)
    events = memory_events_from_diff(before, after, user_id=user_id, session_id=session_id, timestamp=timestamp)
    initial_results, update_chain = normalize_add_events(events)
    return {
        "session_id": session_id,
        "time_stamp": timestamp,
        "evidence_sentence": "",
        "initial_results": initial_results,
        "update_chain": update_chain,
        "openclaw_response": result.stdout.strip(),
    }


def add_conversation_memories(
    openclaw_bin: str,
    agent: str,
    agent_model: str,
    workspace_root: Path,
    item: dict[str, Any],
    global_index: int,
    timeout: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    conversation = item["conversation"]
    speaker_a = conversation["speaker_a"]
    speaker_b = conversation["speaker_b"]
    user_a = f"{speaker_a}_{global_index}"
    user_b = f"{speaker_b}_{global_index}"
    workspace_a = workspace_root / safe_name(user_a)
    workspace_b = workspace_root / safe_name(user_b)

    for user_id, workspace in [(user_a, workspace_a), (user_b, workspace_b)]:
        reset_workspace(workspace)
        ensure_openclaw_workspace(openclaw_bin, profile_for(user_id), workspace, timeout)

    memories_a = {"name": user_a, "workspace": str(workspace_a), "memories": []}
    memories_b = {"name": user_b, "workspace": str(workspace_b), "memories": []}

    keys = session_keys(conversation)
    for session_index, session_key in enumerate(keys, start=1):
        chats = conversation[session_key]
        timestamp = conversation.get(f"{session_key}_date_time", "")
        print(f"Sample {global_index}: OpenCLAW adding {session_key} {session_index}/{len(keys)}", flush=True)
        memories_a["memories"].append(
            add_session_native_memory(
                openclaw_bin, agent, agent_model, user_a, workspace_a, session_key, timestamp, speaker_a, chats, timeout
            )
        )
        memories_b["memories"].append(
            add_session_native_memory(
                openclaw_bin, agent, agent_model, user_b, workspace_b, session_key, timestamp, speaker_b, chats, timeout
            )
        )
    return memories_a, memories_b


def parse_json_output(stdout: str) -> Any:
    stdout = stdout.strip()
    if not stdout:
        return {}
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", stdout, flags=re.DOTALL)
        if not match:
            return {"raw": stdout}
        return json.loads(match.group(1))


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


def search_native_memory(
    openclaw_bin: str,
    agent: str,
    user_id: str,
    workspace: Path,
    question: str,
    top_k: int,
    timeout: float,
) -> list[dict[str, Any]]:
    result = run_command(
        [
            openclaw_bin,
            "memory",
            "search",
            "--query",
            question,
            "--max-results",
            str(top_k),
            "--agent",
            agent,
            "--json",
        ],
        profile=profile_for(user_id),
        timeout=timeout,
        cwd=workspace,
    )
    if result.returncode != 0:
        raise RuntimeError(f"openclaw memory search failed: {result.stderr or result.stdout}")
    return normalize_search_results(parse_json_output(result.stdout))


def answer_question(
    openclaw_bin: str,
    agent: str,
    agent_model: str,
    user_id: str,
    workspace: Path,
    question: str,
    speaker_1_memories: list[dict[str, Any]],
    speaker_2_memories: list[dict[str, Any]],
    timeout: float,
) -> str:
    prompt = ANSWER_PROMPT.format(
        speaker_1_memories=json.dumps(speaker_1_memories, ensure_ascii=False, indent=2),
        speaker_2_memories=json.dumps(speaker_2_memories, ensure_ascii=False, indent=2),
        question=question,
    )
    result = run_command(
        build_openclaw_agent_command(
            openclaw_bin,
            agent,
            prompt,
            f"{safe_name(user_id)}-answer-{hashlib.sha1(question.encode()).hexdigest()[:8]}",
            agent_model,
            timeout,
        ),
        profile=profile_for(user_id),
        timeout=timeout,
        cwd=workspace,
    )
    if result.returncode != 0:
        raise RuntimeError(f"openclaw agent answer failed: {result.stderr or result.stdout}")
    return result.stdout.strip()


def evidence_text(chats: list[dict[str, Any]], evidence_ids: set[str]) -> str:
    matches = [
        f"{chat.get('dia_id')}: {chat.get('speaker')}: {chat.get('text', '')}"
        for chat in chats
        if chat.get("dia_id") in evidence_ids
    ]
    return "\n".join(matches)


def mark_evidence_records(conversation: dict[str, Any], person: dict[str, Any], evidence_ids: set[str]) -> None:
    for record in person["memories"]:
        chats = conversation.get(record["session_id"], [])
        record["evidence_sentence"] = evidence_text(chats, evidence_ids)


def reference_person_shape(person: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": person["name"],
        "workspace": person.get("workspace", ""),
        "memories": [
            {
                "evidence_sentence": record.get("evidence_sentence", ""),
                "time_stamp": record.get("time_stamp", ""),
                "initial_results": record.get("initial_results", []),
                "update_chain": record.get("update_chain", []),
            }
            for record in person["memories"]
        ],
    }


def qa_trace(
    openclaw_bin: str,
    agent: str,
    agent_model: str,
    item: dict[str, Any],
    qa: dict[str, Any],
    person1: dict[str, Any],
    person2: dict[str, Any],
    top_k: int,
    timeout: float,
) -> dict[str, Any]:
    question = qa.get("question", "")
    evidence_ids = set(qa.get("evidence") or [])
    speaker_1_memories = search_native_memory(
        openclaw_bin, agent, person1["name"], Path(person1["workspace"]), question, top_k, timeout
    )
    speaker_2_memories = search_native_memory(
        openclaw_bin, agent, person2["name"], Path(person2["workspace"]), question, top_k, timeout
    )
    qa_response = answer_question(
        openclaw_bin,
        agent,
        agent_model,
        person1["name"],
        Path(person1["workspace"]),
        question,
        speaker_1_memories,
        speaker_2_memories,
        timeout,
    )

    enriched_person1 = json.loads(json.dumps(person1, ensure_ascii=False))
    enriched_person2 = json.loads(json.dumps(person2, ensure_ascii=False))
    mark_evidence_records(item["conversation"], enriched_person1, evidence_ids)
    mark_evidence_records(item["conversation"], enriched_person2, evidence_ids)

    return {
        "qa_question": question,
        "qa_answer": qa.get("answer", ""),
        "qa_response": qa_response,
        "qa_category": qa.get("category", -1),
        "person1": reference_person_shape(enriched_person1),
        "person2": reference_person_shape(enriched_person2),
        "speaker_1_memories": speaker_1_memories,
        "speaker_2_memories": speaker_2_memories,
    }


def output_location(output_dir: Path, global_index: int, part_size: int) -> tuple[int, str, Path]:
    part_id = global_index // part_size + 1
    part_local_key = str(global_index % part_size)
    return part_id, part_local_key, output_dir / f"openclaw_locomo10_part{part_id}.json"


def load_part(output_file: Path) -> dict[str, list[dict[str, Any]]]:
    if not output_file.exists():
        return {}
    with output_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def part_key_completed(output_file: Path, part_local_key: str) -> bool:
    records = load_part(output_file).get(part_local_key) or []
    return bool(records) and not records[0].get("error")


def save_part_record(output_file: Path, part_local_key: str, traces: list[dict[str, Any]]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    lock_file = output_file.with_suffix(output_file.suffix + ".lock")
    tmp_file = output_file.with_suffix(output_file.suffix + f".{os.getpid()}.tmp")
    with lock_file.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        results = load_part(output_file)
        results[part_local_key] = traces
        with tmp_file.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        os.replace(tmp_file, output_file)
        fcntl.flock(lock, fcntl.LOCK_UN)


def build_error_trace(item: dict[str, Any], error: Exception) -> list[dict[str, Any]]:
    return [
        {
            "qa_question": "",
            "qa_answer": "",
            "qa_response": "",
            "qa_category": -1,
            "question_id": f"{item.get('sample_id', '')}_error",
            "error": str(error),
        }
    ]


def run_dry_run(data: list[dict[str, Any]], start: int, end: int) -> None:
    print(f"Loaded {len(data)} samples")
    print(f"Selected samples: {start}..{end - 1}")
    for global_index in range(start, end):
        item = data[global_index]
        conversation = item["conversation"]
        print(f"sample_id={item.get('sample_id')}")
        print(f"speakers={conversation['speaker_a']},{conversation['speaker_b']}")
        print(f"sessions={len(session_keys(conversation))}")
        print(f"qa={len(item['qa'])}")
    print("Dry run complete")


def main() -> int:
    load_cli_env_file(sys.argv[1:])
    args = parse_args()
    data = load_dataset(args.dataset)
    validate_range(args.start, args.end, len(data))
    if args.part_size <= 0:
        raise ValueError(f"Invalid part size: {args.part_size}")

    if args.dry_run:
        run_dry_run(data, args.start, args.end)
        return 0

    if shutil.which(args.openclaw_bin) is None:
        raise FileNotFoundError(f"OpenCLAW CLI not found: {args.openclaw_bin}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.workspace_root.mkdir(parents=True, exist_ok=True)

    for global_index in range(args.start, args.end):
        _, part_local_key, output_file = output_location(args.output_dir, global_index, args.part_size)
        if args.resume and part_key_completed(output_file, part_local_key):
            print(f"Skipping sample {global_index}: already in {output_file.name}", flush=True)
            continue

        item = data[global_index]
        print(f"Processing sample {global_index} ({item.get('sample_id')})", flush=True)
        try:
            person1, person2 = add_conversation_memories(
                args.openclaw_bin,
                args.agent,
                args.agent_model,
                args.workspace_root,
                item,
                global_index,
                args.request_timeout,
            )
            traces = []
            for qa_index, qa in enumerate(item["qa"]):
                print(f"Sample {global_index}: OpenCLAW answering QA {qa_index + 1}/{len(item['qa'])}", flush=True)
                traces.append(
                    qa_trace(
                        args.openclaw_bin,
                        args.agent,
                        args.agent_model,
                        item,
                        qa,
                        person1,
                        person2,
                        args.top_k,
                        args.request_timeout,
                    )
                )
            save_part_record(output_file, part_local_key, traces)
            print(f"Saved {output_file.name}", flush=True)
        except Exception as exc:  # noqa: BLE001
            if args.fail_fast:
                raise
            save_part_record(output_file, part_local_key, build_error_trace(item, exc))
            print(f"Saved error for sample {global_index}: {exc}", flush=True)

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
