#!/usr/bin/env python3
"""Run local Mem0 over LoCoMo and write MemEval-compatible traces."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


DEFAULT_DATASET = Path("/share/project/chenchen/data/locomo/locomo10.json")
DEFAULT_OUTPUT_DIR = Path("/share/project/chenchen/code/MemEval/data/input/mem0_mem/locomo10")
DEFAULT_ENV_FILE = Path("/share/project/chenchen/code/MemEval/.env")
DEFAULT_MEM0_REPO = Path("/share/project/chenchen/code/mem0")
DEFAULT_MEM0_STORE = Path("/share/project/chenchen/code/MemEval/data/input/mem0_mem/locomo10/local_mem0")
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


ANSWER_PROMPT = """You are answering a question using retrieved memories from two people's past conversations.

Rules:
1. Use only the provided memories.
2. If memories conflict, prefer the more recent timestamp.
3. If the answer requires date arithmetic, compute it from timestamps.
4. Answer concisely.

Speaker 1 memories:
{speaker_1_memories}

Speaker 2 memories:
{speaker_2_memories}

Question:
{question}

Answer:
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=10)
    parser.add_argument("--part-size", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--model", default="env:MODEL")
    parser.add_argument("--embedding-model", default="env:EMBEDDING_MODEL")
    parser.add_argument("--llm-api-key-env", default="")
    parser.add_argument("--llm-base-url", default="")
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
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def add_mem0_repo_to_path(mem0_repo: Path) -> None:
    repo = str(mem0_repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)


def build_llm_config(
    model: str,
    *,
    api_key_env: str = "",
    base_url: str = "",
    environ: dict[str, str] | None = None,
) -> dict[str, Any]:
    environ = environ or os.environ
    config: dict[str, Any] = {"model": model, "temperature": 0.0}
    if api_key_env:
        api_key = environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"Environment variable {api_key_env} is not set")
        config["api_key"] = api_key
    if base_url:
        config["openai_base_url"] = base_url
    return config


def create_mem0_client(
    mem0_repo: Path,
    store_dir: Path,
    env_file: Path,
    model: str,
    embedding_model: str,
    llm_api_key_env: str = "",
    llm_base_url: str = "",
):
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
                "collection_name": "locomo10_memories",
            },
        },
        "history_db_path": str(history_db),
        "llm": {
            "provider": "openai",
            "config": build_llm_config(model, api_key_env=llm_api_key_env, base_url=llm_base_url),
        },
        "embedder": {
            "provider": "openai",
            "config": {"model": embedding_model},
        },
    }
    return Memory.from_config(config)


def resolve_model(model_arg: str, *, default: str | None = None) -> str:
    if model_arg.startswith("env:"):
        env_name = model_arg.split(":", 1)[1]
        model = os.getenv(env_name) or default
        if not model:
            raise ValueError(f"Environment variable {env_name} is not set")
        return model
    return model_arg


def retry_call(fn, *, retries: int = 3, delay_seconds: float = 2.0):
    last_error = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries - 1:
                time.sleep(delay_seconds)
    raise last_error


def message_content(chat: dict[str, Any]) -> str:
    pieces = [f"{chat.get('speaker', '')}: {chat.get('text', '')}".strip()]
    caption = chat.get("blip_caption")
    if caption:
        pieces.append(f"Image caption: {caption}")
    query = chat.get("query")
    if query:
        pieces.append(f"Image query: {query}")
    return "\n".join(piece for piece in pieces if piece)


def normalize_session_messages(
    chats: list[dict[str, Any]],
    *,
    speaker_a: str,
    speaker_b: str,
    perspective: str,
) -> list[dict[str, str]]:
    messages = []
    for chat in chats:
        speaker = chat.get("speaker")
        content = message_content(chat)
        if not content.strip():
            continue
        if perspective == speaker_a:
            role = "user" if speaker == speaker_a else "assistant"
        elif perspective == speaker_b:
            role = "user" if speaker == speaker_b else "assistant"
        else:
            raise ValueError(f"Unknown perspective: {perspective}")
        if speaker not in {speaker_a, speaker_b}:
            raise ValueError(f"Unknown speaker: {speaker}")
        messages.append({"role": role, "content": content})
    return messages


def build_metadata(item: dict[str, Any], session_key: str, timestamp: str, perspective: str) -> dict[str, str]:
    return {
        "timestamp": timestamp,
        "session_id": session_key,
        "sample_id": item.get("sample_id", ""),
        "speaker": perspective,
        "source": "locomo10",
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


def evidence_text(chats: list[dict[str, Any]], evidence_ids: set[str]) -> str:
    matches = [
        f"{chat.get('dia_id')}: {chat.get('speaker')}: {chat.get('text', '')}"
        for chat in chats
        if chat.get("dia_id") in evidence_ids
    ]
    return "\n".join(matches)


def session_has_evidence(chats: list[dict[str, Any]], evidence_ids: set[str]) -> bool:
    return any(chat.get("dia_id") in evidence_ids for chat in chats)


def add_conversation_memories(client: Any, item: dict[str, Any], global_index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    conversation = item["conversation"]
    speaker_a = conversation["speaker_a"]
    speaker_b = conversation["speaker_b"]
    user_a = f"{speaker_a}_{global_index}"
    user_b = f"{speaker_b}_{global_index}"
    memories_a = {"name": user_a, "memories": []}
    memories_b = {"name": user_b, "memories": []}

    retry_call(lambda: client.delete_all(user_id=user_a))
    retry_call(lambda: client.delete_all(user_id=user_b))

    keys = session_keys(conversation)
    for session_index, session_key in enumerate(keys, start=1):
        chats = conversation[session_key]
        timestamp = conversation.get(f"{session_key}_date_time", "")
        print(f"Sample {global_index}: adding {session_key} {session_index}/{len(keys)}", flush=True)

        add_jobs = []
        for user_id, speaker, memory_bucket in [
            (user_a, speaker_a, memories_a),
            (user_b, speaker_b, memories_b),
        ]:
            messages = normalize_session_messages(
                chats,
                speaker_a=speaker_a,
                speaker_b=speaker_b,
                perspective=speaker,
            )
            metadata = build_metadata(item, session_key, timestamp, speaker)
            add_jobs.append((user_id, messages, metadata, memory_bucket))

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    retry_call,
                    lambda messages=messages, md=metadata, uid=user_id: client.add(
                        messages, user_id=uid, metadata=md
                    ),
                )
                for user_id, messages, metadata, _memory_bucket in add_jobs
            ]

        for future, (_user_id, _messages, _metadata, memory_bucket) in zip(futures, add_jobs, strict=True):
            result = future.result()
            initial_results, update_chain = normalize_add_events(result)
            memory_bucket["memories"].append(
                {
                    "session_id": session_key,
                    "time_stamp": timestamp,
                    "evidence_sentence": "",
                    "initial_results": initial_results,
                    "update_chain": update_chain,
                }
            )
    return memories_a, memories_b


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
                "score": round(memory.get("score", 0) or 0, 2),
            }
        )
    return normalized


def search_memories(client: Any, question: str, user_id: str, top_k: int) -> list[dict[str, Any]]:
    raw = retry_call(lambda: client.search(question, filters={"user_id": user_id}, top_k=top_k))
    return normalize_search_results(raw)


def answer_question(
    question: str,
    speaker_1_memories: list[dict[str, Any]],
    speaker_2_memories: list[dict[str, Any]],
    model: str,
    request_timeout: float,
    llm_api_key_env: str = "",
    llm_base_url: str = "",
) -> str:
    from openai import OpenAI

    prompt = ANSWER_PROMPT.format(
        speaker_1_memories=json.dumps(speaker_1_memories, ensure_ascii=False, indent=2),
        speaker_2_memories=json.dumps(speaker_2_memories, ensure_ascii=False, indent=2),
        question=question,
    )
    client_kwargs: dict[str, Any] = {"timeout": request_timeout}
    if llm_api_key_env:
        api_key = os.getenv(llm_api_key_env)
        if not api_key:
            raise ValueError(f"Environment variable {llm_api_key_env} is not set")
        client_kwargs["api_key"] = api_key
    if llm_base_url:
        client_kwargs["base_url"] = llm_base_url
    response = retry_call(
        lambda: OpenAI(**client_kwargs).chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": prompt}],
            temperature=0.0,
        )
    )
    return response.choices[0].message.content or ""


def qa_trace(
    client: Any,
    item: dict[str, Any],
    qa: dict[str, Any],
    qa_index: int,
    global_index: int,
    person1: dict[str, Any],
    person2: dict[str, Any],
    top_k: int,
    model: str,
    request_timeout: float,
    llm_api_key_env: str = "",
    llm_base_url: str = "",
) -> dict[str, Any]:
    question = qa.get("question", "")
    evidence_ids = set(qa.get("evidence") or [])
    speaker_1_memories = search_memories(client, question, person1["name"], top_k)
    speaker_2_memories = search_memories(client, question, person2["name"], top_k)
    qa_response = answer_question(
        question,
        speaker_1_memories,
        speaker_2_memories,
        model,
        request_timeout,
        llm_api_key_env,
        llm_base_url,
    )

    enriched_person1 = json.loads(json.dumps(person1, ensure_ascii=False))
    enriched_person2 = json.loads(json.dumps(person2, ensure_ascii=False))
    mark_evidence_records(item["conversation"], enriched_person1, evidence_ids)
    mark_evidence_records(item["conversation"], enriched_person2, evidence_ids)
    evidence_memory_traces = {
        "person1": build_evidence_memory_traces(item["conversation"], enriched_person1, evidence_ids),
        "person2": build_evidence_memory_traces(item["conversation"], enriched_person2, evidence_ids),
    }
    enriched_person1 = reference_person_shape(enriched_person1)
    enriched_person2 = reference_person_shape(enriched_person2)

    return {
        "qa_question": question,
        "qa_answer": qa.get("answer", ""),
        "qa_response": qa_response,
        "qa_category": qa.get("category", -1),
        "person1": enriched_person1,
        "person2": enriched_person2,
        "evidence_memory_traces": evidence_memory_traces,
        "speaker_1_memories": speaker_1_memories,
        "speaker_2_memories": speaker_2_memories,
    }


def mark_evidence_records(conversation: dict[str, Any], person: dict[str, Any], evidence_ids: set[str]) -> None:
    for record in person["memories"]:
        chats = conversation.get(record["session_id"], [])
        record["evidence_sentence"] = evidence_text(chats, evidence_ids)


def format_evidence_sentence(chat: dict[str, Any]) -> str:
    return f"{chat.get('speaker', '')}: {chat.get('text', '')}".strip()


def memory_ids(events: list[dict[str, Any]]) -> list[str]:
    ids = []
    for event in events:
        memory_id = event.get("id")
        if memory_id and memory_id not in ids:
            ids.append(memory_id)
    return ids


def memory_update_paths(
    initial_results: list[dict[str, Any]], update_chain: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    paths = []
    for memory_id in memory_ids(initial_results):
        paths.append(
            {
                "memory_id": memory_id,
                "initial_results": [event for event in initial_results if event.get("id") == memory_id],
                "update_chain": [event for event in update_chain if event.get("id") == memory_id],
            }
        )
    return paths


def build_evidence_memory_traces(
    conversation: dict[str, Any], person: dict[str, Any], evidence_ids: set[str]
) -> list[dict[str, Any]]:
    records_by_session = {record.get("session_id"): record for record in person["memories"]}
    traces = []
    for session_id in session_keys(conversation):
        record = records_by_session.get(session_id)
        if not record:
            continue
        for chat in conversation.get(session_id, []):
            dia_id = chat.get("dia_id")
            if dia_id not in evidence_ids:
                continue
            initial_results = record.get("initial_results", [])
            update_chain = record.get("update_chain", [])
            traces.append(
                {
                    "dia_id": dia_id,
                    "speaker": chat.get("speaker", ""),
                    "evidence_sentence": format_evidence_sentence(chat),
                    "session_id": session_id,
                    "time_stamp": record.get("time_stamp", ""),
                    "initial_results": initial_results,
                    "initial_memory_ids": memory_ids(initial_results),
                    "memory_update_paths": memory_update_paths(initial_results, update_chain),
                    "update_chain": update_chain,
                }
            )
    return traces


def reference_person_shape(person: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": person["name"],
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


def output_location(output_dir: Path, global_index: int, part_size: int) -> tuple[int, str, Path]:
    part_id = global_index // part_size + 1
    part_local_key = str(global_index % part_size)
    return part_id, part_local_key, output_dir / f"mem0_locomo10_part{part_id}.json"


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


def validate_range(start: int, end: int, total: int) -> None:
    if start < 0 or end < start or end > total:
        raise ValueError(f"Invalid range start={start}, end={end}, dataset size={total}")


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
    args = parse_args()
    load_env_file(args.env_file)
    data = load_dataset(args.dataset)
    validate_range(args.start, args.end, len(data))
    if args.part_size <= 0:
        raise ValueError(f"Invalid part size: {args.part_size}")

    if args.dry_run:
        run_dry_run(data, args.start, args.end)
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = resolve_model(args.model, default=DEFAULT_MODEL)
    embedding_model = resolve_model(args.embedding_model, default=DEFAULT_EMBEDDING_MODEL)
    client = create_mem0_client(
        args.mem0_repo,
        args.mem0_store_dir,
        args.env_file,
        model,
        embedding_model,
        args.llm_api_key_env,
        args.llm_base_url,
    )

    for global_index in range(args.start, args.end):
        _, part_local_key, output_file = output_location(args.output_dir, global_index, args.part_size)
        if args.resume and part_key_completed(output_file, part_local_key):
            print(f"Skipping sample {global_index}: already in {output_file.name}", flush=True)
            continue

        item = data[global_index]
        print(f"Processing sample {global_index} ({item.get('sample_id')})", flush=True)
        try:
            person1, person2 = add_conversation_memories(client, item, global_index)
            traces = []
            for qa_index, qa in enumerate(item["qa"]):
                print(f"Sample {global_index}: answering QA {qa_index + 1}/{len(item['qa'])}", flush=True)
                traces.append(
                    qa_trace(
                        client,
                        item,
                        qa,
                        qa_index,
                        global_index,
                        person1,
                        person2,
                        args.top_k,
                        model,
                        args.request_timeout,
                        args.llm_api_key_env,
                        args.llm_base_url,
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
