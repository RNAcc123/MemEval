#!/usr/bin/env python3
"""Run Mem0 over LongMemEval-S and write MemEval-compatible traces."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_DATASET = Path("/share/project/chenchen/data/longmemeval-cleaned/longmemeval_s_cleaned.json")
DEFAULT_OUTPUT_DIR = Path("/share/project/chenchen/code/MemEval/data/input/mem0_mem/longmemeval_s")
DEFAULT_ENV_FILE = Path("/share/project/chenchen/code/MemEval/.env")
DEFAULT_MEM0_REPO = Path("/share/project/chenchen/code/mem0")
DEFAULT_MEM0_STORE = Path("/share/project/chenchen/code/MemEval/data/input/mem0_mem/longmemeval_s/local_mem0")
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


ANSWER_PROMPT = """You are answering a question using retrieved memories from a user's past conversations.

Rules:
1. Use only the provided memories.
2. If memories conflict, prefer the more recent timestamp.
3. If the answer requires date arithmetic, compute it from timestamps.
4. Answer concisely.

Memories:
{memories_json}

Question:
{question}

Answer:
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=500)
    parser.add_argument("--part-size", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--model", default="env:MODEL")
    parser.add_argument("--embedding-model", default="env:EMBEDDING_MODEL")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--mem0-repo", type=Path, default=DEFAULT_MEM0_REPO)
    parser.add_argument("--mem0-store-dir", type=Path, default=DEFAULT_MEM0_STORE)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    return parser.parse_args()


def load_dataset(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    for idx, item in enumerate(data):
        lengths = (
            len(item["haystack_dates"]),
            len(item["haystack_session_ids"]),
            len(item["haystack_sessions"]),
        )
        if len(set(lengths)) != 1:
            raise ValueError(f"Sample {idx} has misaligned haystack fields: {lengths}")
    return data


def normalize_session_messages(session: list[dict[str, Any]]) -> list[dict[str, str]]:
    messages = []
    for message in session:
        role = message.get("role")
        content = message.get("content", "")
        if role not in {"user", "assistant", "system"}:
            raise ValueError(f"Unsupported role: {role}")
        if content.strip():
            messages.append({"role": role, "content": content})
    return messages


def session_has_answer(session_id: str, session: list[dict[str, Any]], answer_session_ids: set[str]) -> bool:
    return session_id in answer_session_ids or any(bool(message.get("has_answer")) for message in session)


def build_metadata(item: dict[str, Any], haystack_date: str, haystack_session_id: str) -> dict[str, str]:
    return {
        "timestamp": haystack_date,
        "session_id": haystack_session_id,
        "question_id": item.get("question_id", ""),
        "question_type": item.get("question_type", ""),
        "source": "longmemeval_s",
    }


def normalize_add_events(result: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if isinstance(result, list):
        events = result
    elif isinstance(result, dict):
        events = result.get("results", [])
    else:
        events = []

    if not events:
        events = [{"error": "No memory in that message"}]

    initial_results = [event for event in events if event.get("event") == "ADD"]
    if not initial_results and events == [{"error": "No memory in that message"}]:
        initial_results = events
    return initial_results, events


def retry_call(fn, *, retries: int = 3, delay_seconds: float = 2.0):
    last_error = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - preserve API exception messages in trace.
            last_error = exc
            if attempt < retries - 1:
                time.sleep(delay_seconds)
    raise last_error


def load_env_file(env_file: Path = DEFAULT_ENV_FILE) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_env_file_plain(env_file)
        return
    load_dotenv(env_file)


def load_env_file_plain(env_file: Path) -> None:
    if not env_file.exists():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        os.environ.setdefault(key, value)


def add_mem0_repo_to_path(mem0_repo: Path) -> None:
    repo = str(mem0_repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)


def create_mem0_client(mem0_repo: Path, store_dir: Path, env_file: Path, model: str, embedding_model: str):
    load_env_file(env_file)
    add_mem0_repo_to_path(mem0_repo)

    from mem0 import Memory

    vector_store_dir = store_dir / "qdrant"
    history_db = store_dir / "history.db"
    vector_store_dir.mkdir(parents=True, exist_ok=True)
    history_db.parent.mkdir(parents=True, exist_ok=True)

    config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "path": str(vector_store_dir),
                "collection_name": "longmemeval_s_memories",
            },
        },
        "history_db_path": str(history_db),
        "llm": {
            "provider": "openai",
            "config": {
                "model": model,
                "temperature": 0.0,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": embedding_model,
            },
        },
    }
    return Memory.from_config(config)


def resolve_model(model_arg: str, *, default: str | None = None) -> str:
    if model_arg.startswith("env:"):
        env_name = model_arg.split(":", 1)[1]
        model = os.getenv(env_name)
        if not model and default:
            model = default
        if not model:
            raise ValueError(f"Environment variable {env_name} is not set")
        return model
    return model_arg


def normalize_search_results(raw_memories: Any) -> list[dict[str, Any]]:
    if isinstance(raw_memories, dict):
        raw_memories = raw_memories.get("results", [])

    normalized = []
    for memory in raw_memories or []:
        metadata = memory.get("metadata") or {}
        normalized.append(
            {
                "memory": memory.get("memory", ""),
                "timestamp": metadata.get("timestamp"),
                "session_id": metadata.get("session_id"),
                "score": round(memory.get("score", 0) or 0, 2),
            }
        )
    return normalized


def search_memories(client: Any, question: str, user_id: str, top_k: int) -> list[dict[str, Any]]:
    memories = retry_call(lambda: client.search(question, filters={"user_id": user_id}, top_k=top_k))
    return normalize_search_results(memories)


def answer_question(question: str, memories: list[dict[str, Any]], model: str, request_timeout: float) -> str:
    from openai import OpenAI

    prompt = ANSWER_PROMPT.format(
        memories_json=json.dumps(memories, ensure_ascii=False, indent=2),
        question=question,
    )
    response = retry_call(
        lambda: OpenAI(timeout=request_timeout).chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": prompt}],
            temperature=0.0,
        )
    )
    return response.choices[0].message.content or ""


def output_location(output_dir: Path, global_index: int, part_size: int) -> tuple[int, str, Path]:
    part_id = global_index // part_size + 1
    part_local_key = str(global_index % part_size)
    output_file = output_dir / f"mem0_longmemeval_s_part{part_id}.json"
    return part_id, part_local_key, output_file


def load_part(output_file: Path) -> dict[str, list[dict[str, Any]]]:
    if not output_file.exists():
        return {}
    with output_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_part(output_file: Path, results: dict[str, list[dict[str, Any]]]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def part_key_completed(output_file: Path, part_local_key: str) -> bool:
    results = load_part(output_file)
    records = results.get(part_local_key) or []
    if not records:
        return False
    return not records[0].get("error")


def save_part_record(output_file: Path, part_local_key: str, trace: dict[str, Any]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    lock_file = output_file.with_suffix(output_file.suffix + ".lock")
    tmp_file = output_file.with_suffix(output_file.suffix + f".{os.getpid()}.tmp")

    with lock_file.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        results = load_part(output_file)
        results[part_local_key] = [trace]
        with tmp_file.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        os.replace(tmp_file, output_file)
        fcntl.flock(lock, fcntl.LOCK_UN)


def build_error_trace(item: dict[str, Any], error: Exception) -> dict[str, Any]:
    return {
        "qa_question": item.get("question", ""),
        "qa_answer": item.get("answer", ""),
        "qa_response": "",
        "qa_category": item.get("question_type", ""),
        "question_id": item.get("question_id", ""),
        "error": str(error),
    }


def process_sample(
    client: Any,
    item: dict[str, Any],
    global_index: int,
    top_k: int,
    model: str,
    request_timeout: float,
) -> dict[str, Any]:
    user_id = f"longmemeval_s_{global_index}"
    answer_session_ids = set(item.get("answer_session_ids") or [])
    memory_records = []

    retry_call(lambda: client.delete_all(user_id=user_id))

    sessions = list(zip(item["haystack_dates"], item["haystack_session_ids"], item["haystack_sessions"], strict=True))
    for session_index, (haystack_date, session_id, session) in enumerate(sessions, start=1):
        print(
            f"Sample {global_index}: adding session {session_index}/{len(sessions)} ({session_id})",
            flush=True,
        )
        normalized_messages = normalize_session_messages(session)
        metadata = build_metadata(item, haystack_date, session_id)
        result = retry_call(
            lambda messages=normalized_messages, md=metadata: client.add(
                messages,
                user_id=user_id,
                metadata=md,
            )
        )
        initial_results, update_chain = normalize_add_events(result)
        memory_records.append(
            {
                "session_id": session_id,
                "time_stamp": haystack_date,
                "has_answer": session_has_answer(session_id, session, answer_session_ids),
                "evidence_sentence": f"session_id={session_id}",
                "initial_results": initial_results,
                "update_chain": update_chain,
            }
        )

    print(f"Sample {global_index}: searching memories", flush=True)
    speaker_1_memories = search_memories(client, item.get("question", ""), user_id, top_k)
    retrieved_answer_session_ids = sorted(
        {
            memory["session_id"]
            for memory in speaker_1_memories
            if memory.get("session_id") in answer_session_ids
        }
    )
    print(f"Sample {global_index}: generating QA response", flush=True)
    qa_response = answer_question(item.get("question", ""), speaker_1_memories, model, request_timeout)

    return {
        "qa_question": item.get("question", ""),
        "qa_answer": item.get("answer", ""),
        "qa_response": qa_response,
        "qa_category": item.get("question_type", ""),
        "question_id": item.get("question_id", ""),
        "question_date": item.get("question_date", ""),
        "answer_session_ids": item.get("answer_session_ids") or [],
        "person1": {
            "name": user_id,
            "memories": memory_records,
        },
        "person2": {
            "name": "",
            "memories": [],
        },
        "speaker_1_memories": speaker_1_memories,
        "speaker_2_memories": [],
        "retrieved_answer_session_ids": retrieved_answer_session_ids,
        "retrieval_hit": bool(retrieved_answer_session_ids),
    }


def validate_range(start: int, end: int, total: int) -> None:
    if start < 0 or end < start or end > total:
        raise ValueError(f"Invalid range start={start}, end={end}, dataset size={total}")


def validate_part_size(part_size: int) -> None:
    if part_size <= 0:
        raise ValueError(f"Invalid part size: {part_size}")


def run_dry_run(data: list[dict[str, Any]], start: int, end: int) -> None:
    print(f"Loaded {len(data)} samples")
    print(f"Selected samples: {start}..{end - 1}")
    for global_index in range(start, end):
        item = data[global_index]
        first_session = item["haystack_sessions"][0] if item["haystack_sessions"] else []
        normalize_session_messages(first_session)
        print(f"user_id=longmemeval_s_{global_index}")
        print(f"sessions={len(item['haystack_sessions'])}")
    print("Dry run complete")


def main() -> int:
    args = parse_args()
    load_env_file(args.env_file)
    data = load_dataset(args.dataset)
    validate_range(args.start, args.end, len(data))
    validate_part_size(args.part_size)

    if args.dry_run:
        run_dry_run(data, args.start, args.end)
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = resolve_model(args.model, default=DEFAULT_MODEL)
    embedding_model = resolve_model(args.embedding_model, default=DEFAULT_EMBEDDING_MODEL)
    client = create_mem0_client(args.mem0_repo, args.mem0_store_dir, args.env_file, model, embedding_model)

    for global_index in range(args.start, args.end):
        _, part_local_key, output_file = output_location(args.output_dir, global_index, args.part_size)

        if args.resume and part_key_completed(output_file, part_local_key):
            print(f"Skipping sample {global_index}: already in {output_file.name}")
            continue

        print(f"Processing sample {global_index}", flush=True)
        item = data[global_index]
        try:
            trace = process_sample(client, item, global_index, args.top_k, model, args.request_timeout)
        except Exception as exc:  # noqa: BLE001 - save per-sample failures as requested.
            if args.fail_fast:
                raise
            trace = build_error_trace(item, exc)

        save_part_record(output_file, part_local_key, trace)
        print(f"Saved {output_file.name}", flush=True)

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
