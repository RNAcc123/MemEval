#!/usr/bin/env python3
"""Concurrent LongMemEval-S diagnosis runner.

This script adapts ``run_diagnosis.py`` to the LongMemEval-S Mem0 trace files
under ``data/input/mem0_mem/longmemeval_s``. It keeps the staged diagnosis logic
unchanged and adds item-level concurrency plus LongMemEval-specific metadata in
the output.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Set, Tuple

# ``memeval.diagnosis`` imports python-dotenv at module import time. Keep this
# adapter importable in lean test environments where dotenv is not installed.
DEFAULT_ENV_FILE = Path(os.getenv("MEMEVAL_ENV_FILE", ".env"))
try:
    from dotenv import load_dotenv

    load_dotenv(DEFAULT_ENV_FILE, override=True)
except ImportError:
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *args, **kwargs: False
    sys.modules["dotenv"] = dotenv_stub

from memeval.diagnosis import (
    analyze_qa_pair,
    analyze_qa_pair_with_voting,
    load_json_file,
)
from memeval.schema import (
    DIAGNOSIS_SCHEMA_VERSION,
    DiagnosisStage,
    MemoryData,
    QAData,
    UsageStats,
    validate_trace_dataset,
)


DEFAULT_INPUT = Path(os.getenv(
    "MEMEVAL_LONGMEMEVAL_TRACE",
    "data/input/mem0_mem/longmemeval_s/mem0_longmemeval_s_part1.json",
))
DEFAULT_OUTPUT_DIR = Path(os.getenv(
    "MEMEVAL_LONGMEMEVAL_DIAGNOSIS_OUTPUT",
    "data/output/llm_annotation_longmemeval_s",
))


class DiagnosisTask(NamedTuple):
    """One LongMemEval-S QA item ready for diagnosis."""

    item_id: str
    sample_key: str
    question_index: int
    qa_item: Dict
    qa_data: QAData
    memory_data: MemoryData
    output_metadata: Dict


_print_lock = threading.Lock()
_write_lock = threading.Lock()


def thread_print(*args, **kwargs) -> None:
    """Print without interleaving lines from multiple workers."""
    with _print_lock:
        print(*args, **kwargs)


def make_item_id(sample_key: str, question_index: int, qa_item: Dict) -> str:
    """Build a stable output id for one LongMemEval-S QA item."""
    question_id = qa_item.get("question_id")
    if question_id:
        return f"longmemeval_s_{sample_key}_{question_id}"
    return f"longmemeval_s_{sample_key}_{question_index}"


def iter_diagnosis_tasks(data: Dict, processed_ids: Set[str]) -> Iterable[DiagnosisTask]:
    """Yield unprocessed diagnosis tasks from a LongMemEval-S part file."""
    for sample_key, qa_list in data.items():
        if not isinstance(qa_list, list):
            logging.warning("Skip sample %s: expected a list, got %s", sample_key, type(qa_list).__name__)
            continue

        for question_index, qa_item in enumerate(qa_list):
            if not isinstance(qa_item, dict):
                logging.warning("Skip sample %s question %s: expected an object", sample_key, question_index)
                continue

            item_id = make_item_id(sample_key, question_index, qa_item)
            if item_id in processed_ids:
                continue

            qa_data = QAData(
                question=qa_item.get("qa_question", ""),
                answer=qa_item.get("qa_answer", ""),
                response=qa_item.get("qa_response", ""),
                category=qa_item.get("qa_category", ""),
            )
            memory_data = MemoryData.from_qa_item(qa_item)
            output_metadata = {
                "sample_key": sample_key,
                "question_index": question_index,
                "question_id": qa_item.get("question_id", ""),
                "question_date": qa_item.get("question_date", ""),
                "answer_session_ids": qa_item.get("answer_session_ids", []),
                "retrieved_answer_session_ids": qa_item.get("retrieved_answer_session_ids", []),
                "retrieval_hit": qa_item.get("retrieval_hit", False),
            }

            yield DiagnosisTask(
                item_id=item_id,
                sample_key=sample_key,
                question_index=question_index,
                qa_item=qa_item,
                qa_data=qa_data,
                memory_data=memory_data,
                output_metadata=output_metadata,
            )


def load_existing_results(output_file: Path) -> List[Dict]:
    """Load existing output records for resume support."""
    if not output_file.exists():
        return []
    try:
        with output_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        logging.warning("Ignore existing output %s: expected a list", output_file)
    except (json.JSONDecodeError, OSError) as exc:
        logging.warning("Ignore existing output %s: %s", output_file, exc)
    return []


def processed_item_ids(results: List[Dict]) -> Set[str]:
    """Return item ids already present in an output file."""
    return {
        item["conv_id_question_id"]
        for item in results
        if isinstance(item, dict) and item.get("conv_id_question_id")
    }


def save_results(output_file: Path, results: List[Dict]) -> None:
    """Write all results atomically enough for a single local process."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = output_file.with_suffix(output_file.suffix + f".{os.getpid()}.tmp")
    with tmp_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    os.replace(tmp_file, output_file)


def append_result(output_file: Path, results: List[Dict], result: Dict) -> None:
    """Append one result and persist it under a process-local lock."""
    with _write_lock:
        results.append(result)
        save_results(output_file, results)


def usage_stats_from_dict(stats: Optional[Dict]) -> UsageStats:
    """Convert serialized usage stats back into ``UsageStats`` for aggregation."""
    usage = UsageStats()
    if not stats:
        return usage
    usage.total_calls = stats.get("total_calls", 0)
    usage.total_latency = stats.get("total_latency_seconds", 0)
    usage.total_prompt_tokens = stats.get("total_prompt_tokens", 0)
    usage.total_completion_tokens = stats.get("total_completion_tokens", 0)
    usage.total_tokens = stats.get("total_tokens", 0)
    usage.call_details = stats.get("call_details", [])
    return usage


def build_output_record(task: DiagnosisTask, analysis: Dict, mode: str) -> Dict:
    """Build the JSON output record for one completed diagnosis."""
    result = {
        "schema_version": DIAGNOSIS_SCHEMA_VERSION,
        "conv_id_question_id": task.item_id,
        "qa_question": task.qa_item.get("qa_question", ""),
        "qa_answer": task.qa_item.get("qa_answer", ""),
        "qa_response": task.qa_item.get("qa_response", ""),
        "qa_category": task.qa_item.get("qa_category", ""),
        "label": analysis.get("label"),
        "reason": analysis.get("reason", ""),
        "status": analysis.get("status", "completed"),
        "answer_correct": analysis.get("answer_correct", analysis.get("label") is None),
        "diagnosis_mode": mode,
    }
    result.update(task.output_metadata)

    if analysis.get("stage") is not None:
        result["stage"] = analysis["stage"]
    if analysis.get("voting_details") is not None:
        result["voting_details"] = analysis["voting_details"]
    if analysis.get("usage_stats") is not None:
        result["usage_stats"] = analysis["usage_stats"]
    return result


def run_task(task: DiagnosisTask, model: str, use_voting: bool, num_votes: int) -> Tuple[Dict, UsageStats]:
    """Run diagnosis for one task and return an output record plus usage stats."""
    thread_print(f"🔍 Processing {task.item_id}")
    if use_voting:
        subjects = [
            {
                "subject_id": subject.subject_id,
                "memories": subject.memories,
                "retrieval": subject.retrieval,
            }
            for subject in task.memory_data.subjects
        ]
        analysis = analyze_qa_pair_with_voting(
            qa_question=task.qa_data.question,
            qa_answer=task.qa_data.answer,
            qa_response=task.qa_data.response,
            subjects=subjects,
            model=model,
            num_votes=num_votes,
        )
        record = build_output_record(task, analysis, f"voting_{num_votes}rounds")
        return record, usage_stats_from_dict(analysis.get("usage_stats"))

    diagnosis = analyze_qa_pair(task.qa_data, task.memory_data, model=model)
    stage = diagnosis.stage.value if isinstance(diagnosis.stage, DiagnosisStage) else diagnosis.stage
    analysis = {
        "label": diagnosis.label,
        "reason": diagnosis.reason,
        "stage": stage,
        "status": diagnosis.status.value,
        "answer_correct": diagnosis.answer_correct,
        "usage_stats": diagnosis.usage_stats.to_dict() if diagnosis.usage_stats else None,
    }
    record = build_output_record(task, analysis, f"single_model_{model}")
    return record, diagnosis.usage_stats or UsageStats()


def generate_output_file(input_file: Path, output_dir: Path, model: str, use_voting: bool, num_votes: int) -> Path:
    """Generate a timestamped LongMemEval-S diagnosis output path."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = f"voting_{num_votes}rounds" if use_voting else "single"
    filename = f"{input_file.stem}_{mode}_{model.replace('-', '_')}_{timestamp}.json"
    return output_dir / filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    model_map = {
        "deepseek": "deepseek",
        "gpt4.1": "gpt-4.1",
        "gpt5": "gpt-5",
    }
    parser.add_argument("model", nargs="?", default="deepseek", choices=sorted(model_map))
    parser.add_argument("-i", "--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("-o", "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("-f", "--output-file", type=Path, default=None)
    parser.add_argument("-t", "--threads", type=int, default=4)
    parser.add_argument("--voting", action="store_true", default=True)
    parser.add_argument("--no-voting", action="store_true")
    parser.add_argument("--num-votes", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="Process at most N unprocessed QA items.")
    args = parser.parse_args()
    args.model = model_map[args.model]
    args.use_voting = args.voting and not args.no_voting
    args.threads = max(1, args.threads)
    return args


def main() -> int:
    args = parse_args()
    input_file = args.input
    if not input_file.exists():
        print(f"❌ Error: input file does not exist: {input_file}")
        return 1

    output_file = args.output_file
    if output_file is None:
        output_file = generate_output_file(input_file, args.output_dir, args.model, args.use_voting, args.num_votes)
    elif not output_file.is_absolute():
        output_file = args.output_dir / output_file

    data = validate_trace_dataset(load_json_file(str(input_file)))
    results = load_existing_results(output_file)
    processed_ids = processed_item_ids(results)
    tasks = list(iter_diagnosis_tasks(data, processed_ids))
    if args.limit is not None:
        tasks = tasks[: args.limit]

    print("\n" + "=" * 70)
    print("🚀 LongMemEval-S memory diagnosis started")
    print("=" * 70)
    print(f"🤖 Model: {args.model}")
    print(f"📊 Diagnosis mode: {'Voting (' + str(args.num_votes) + ' rounds)' if args.use_voting else 'Single-model diagnosis'}")
    print(f"📁 Input: {input_file}")
    print(f"📄 Output: {output_file}")
    print(f"🧵 Item-level threads: {args.threads}")
    print(f"⏭️  Resume skipped: {len(processed_ids)}")
    print(f"📝 Pending items: {len(tasks)}")
    print("=" * 70 + "\n")

    if not tasks:
        save_results(output_file, results)
        print("No pending items.")
        return 0

    global_stats = UsageStats()
    completed = 0
    failed = 0
    effective_threads = min(args.threads, len(tasks))

    with ThreadPoolExecutor(max_workers=effective_threads) as executor:
        future_to_task = {
            executor.submit(run_task, task, args.model, args.use_voting, args.num_votes): task
            for task in tasks
        }
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                record, stats = future.result()
                append_result(output_file, results, record)
                global_stats.merge(stats)
                completed += 1
                thread_print(f"✅ Completed {task.item_id} ({completed}/{len(tasks)})")
            except Exception as exc:  # noqa: BLE001 - keep processing independent items.
                failed += 1
                logging.exception("Diagnosis failed for %s", task.item_id)
                error_record = {
                    "schema_version": DIAGNOSIS_SCHEMA_VERSION,
                    "conv_id_question_id": task.item_id,
                    "qa_question": task.qa_item.get("qa_question", ""),
                    "qa_answer": task.qa_item.get("qa_answer", ""),
                    "qa_response": task.qa_item.get("qa_response", ""),
                    "qa_category": task.qa_item.get("qa_category", ""),
                    "label": None,
                    "reason": f"Diagnosis failed: {exc}",
                    "stage": "error",
                    "diagnosis_mode": "error",
                }
                error_record.update(task.output_metadata)
                append_result(output_file, results, error_record)
                thread_print(f"❌ Failed {task.item_id}: {exc}")

    print("\n" + "=" * 70)
    print("🎉 LongMemEval-S diagnosis completed")
    print("=" * 70)
    print(f"✅ Completed: {completed}")
    print(f"❌ Failed: {failed}")
    print(f"📄 Output: {output_file}")
    print("\n📊 API call summary")
    global_stats.print_summary()
    print("=" * 70 + "\n")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
