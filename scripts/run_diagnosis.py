"""
Memory diagnosis system - staged diagnosis for issues in QA pairs.

This script is the thin CLI/driver-loop entrypoint. The stage logic, LLM
helpers, and diagnosis algorithms live in the `memeval.diagnosis` package:
- Stage 0: Consistency check
- Stage 1: Memory extraction diagnosis
- Stage 2: Memory update diagnosis
- Stage 3: Memory retrieval diagnosis
- Stage 4: Reasoning diagnosis
"""

# Standard library imports
import glob as glob_module
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from memeval.config import DiagnosisConfig
from memeval.diagnosis import (
    analyze_qa_pair,
    analyze_qa_pair_with_voting,
    load_json_file,
)
from memeval.schema import (
    DIAGNOSIS_SCHEMA_VERSION,
    DiagnosisStage,
    DiagnosisStatus,
    MemoryData,
    QAData,
    UsageStats,
    validate_trace_dataset,
)

# ============================================================================
# Single-file processing (thread-safe)
# ============================================================================

# Thread-safe print lock
_print_lock = threading.Lock()


def _thread_print(*args, **kwargs):
    """Thread-safe print helper."""
    with _print_lock:
        print(*args, **kwargs)


def process_single_file(
    input_file: str,
    output_file: str,
    model: str,
    use_voting: bool,
    num_votes: int,
    qa_threads: int = 1,
    thread_label: str = "",
    min_valid_votes: Optional[int] = None,
) -> Tuple[int, UsageStats]:
    """Run diagnosis for a single input file.

    This function is safe to call from multiple threads; each thread processes
    an independent input/output file pair.

    Args:
        input_file: input JSON file path
        output_file: output JSON file path
        model: model name to use
        use_voting: whether to use voting
        num_votes: number of voting rounds
        qa_threads: number of worker threads for QA items within this file
        thread_label: thread label (for log prefix)

    Returns:
        Tuple of (num_processed_items, file_level_usage_stats)
    """
    prefix = f"[{thread_label}] " if thread_label else ""

    # Validate input file
    if not os.path.exists(input_file):
        _thread_print(f"{prefix}❌ Error: Input file does not exist: {input_file}")
        return 0, UsageStats()

    # Load input data
    try:
        data = load_json_file(input_file)
        data = validate_trace_dataset(data)
        _thread_print(f"{prefix}✅ Loaded {input_file} successfully, total conversations: {len(data)}")
    except Exception as e:
        logging.error(f"{prefix}Failed to load input file: {str(e)}")
        _thread_print(f"{prefix}❌ Error: Failed to parse input file: {str(e)}")
        return 0, UsageStats()

    # Load previously processed results (resume support)
    results = []
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                results = json.load(f)
                _thread_print(f"{prefix}📂 Loaded {len(results)} historical results (resume enabled)")
        except (json.JSONDecodeError, FileNotFoundError):
            results = []

    processed_items = {item["conv_id_question_id"] for item in results}

    # File-level usage tracker
    file_stats = UsageStats()

    try:
        total_convs = len(data)
        _thread_print(f"{prefix}📊 Start processing, total conversations: {total_convs}\n")

        pending_items = []
        for conv_idx, (conv_id, qa_list) in enumerate(data.items(), 1):
            _thread_print(f"\n{prefix}{'='*60}")
            _thread_print(f"{prefix}📝 Processing conversation {conv_id} ({conv_idx}/{total_convs})")
            _thread_print(f"{prefix}{'='*60}\n")

            for qa_idx, qa_item in enumerate(qa_list, 1):
                item_id = f"{conv_id}_{qa_idx-1}"

                if item_id in processed_items:
                    _thread_print(f"{prefix}⏭️  Skip already processed item: {item_id}")
                    continue

                pending_items.append((conv_id, qa_idx, len(qa_list), qa_item, item_id))

        def process_qa_item(task: Tuple[str, int, int, Dict, str]) -> Optional[Tuple[Dict, UsageStats]]:
            _conv_id, qa_idx, qa_count, qa_item, item_id = task
            _thread_print(f"{prefix}🔍 Processing question {qa_idx}/{qa_count}: {item_id}")

            try:
                memory_data = MemoryData.from_qa_item(qa_item)
                subjects = [
                    {
                        "subject_id": subject.subject_id,
                        "memories": subject.memories,
                        "retrieval": subject.retrieval,
                    }
                    for subject in memory_data.subjects
                ]
                item_stats = UsageStats()

                if use_voting:
                    analysis = analyze_qa_pair_with_voting(
                        qa_question=qa_item["qa_question"],
                        qa_answer=qa_item["qa_answer"],
                        qa_response=qa_item["qa_response"],
                        subjects=subjects,
                        model=model,
                        num_votes=num_votes,
                        min_valid_votes=min_valid_votes,
                    )

                    result = {
                        "schema_version": DIAGNOSIS_SCHEMA_VERSION,
                        "conv_id_question_id": item_id,
                        "qa_question": qa_item["qa_question"],
                        "qa_answer": qa_item["qa_answer"],
                        "qa_response": qa_item["qa_response"],
                        "qa_category": qa_item.get("qa_category", ""),
                        "label": analysis["label"],
                        "reason": analysis["reason"],
                        "stage": analysis.get("stage"),
                        "status": analysis.get("status", DiagnosisStatus.COMPLETED.value),
                        "answer_correct": analysis.get("answer_correct", analysis.get("label") is None),
                        "diagnosis_mode": f"voting_{num_votes}rounds",
                    }

                    if "voting_details" in analysis:
                        result["voting_details"] = analysis["voting_details"]

                    if "usage_stats" in analysis:
                        result["usage_stats"] = analysis["usage_stats"]
                        item_stats.total_calls = analysis["usage_stats"]["total_calls"]
                        item_stats.total_latency = analysis["usage_stats"]["total_latency_seconds"]
                        item_stats.total_prompt_tokens = analysis["usage_stats"]["total_prompt_tokens"]
                        item_stats.total_completion_tokens = analysis["usage_stats"]["total_completion_tokens"]
                        item_stats.total_tokens = analysis["usage_stats"]["total_tokens"]
                        item_stats.call_details = analysis["usage_stats"].get("call_details", [])
                else:
                    qa_data = QAData(
                        question=qa_item["qa_question"],
                        answer=qa_item["qa_answer"],
                        response=qa_item["qa_response"],
                    )

                    diagnosis = analyze_qa_pair(qa_data, memory_data, model=model)

                    result = {
                        "schema_version": DIAGNOSIS_SCHEMA_VERSION,
                        "conv_id_question_id": item_id,
                        "qa_question": qa_item["qa_question"],
                        "qa_answer": qa_item["qa_answer"],
                        "qa_response": qa_item["qa_response"],
                        "qa_category": qa_item.get("qa_category", ""),
                        "label": diagnosis.label,
                        "reason": diagnosis.reason,
                        "stage": diagnosis.stage.value if isinstance(diagnosis.stage, DiagnosisStage) else diagnosis.stage,
                        "status": diagnosis.status.value,
                        "answer_correct": diagnosis.answer_correct,
                        "diagnosis_mode": f"single_model_{model}",
                    }

                    if diagnosis.usage_stats is not None:
                        result["usage_stats"] = diagnosis.usage_stats.to_dict()
                        item_stats.merge(diagnosis.usage_stats)

                return result, item_stats

            except Exception as e:
                logging.error(f"{prefix}Error while processing {item_id}: {str(e)}")
                _thread_print(f"{prefix}❌ {item_id} failed: {str(e)}\n")
                return ({
                    "schema_version": DIAGNOSIS_SCHEMA_VERSION,
                    "conv_id_question_id": item_id,
                    "qa_question": qa_item.get("qa_question", ""),
                    "qa_answer": qa_item.get("qa_answer", ""),
                    "qa_response": qa_item.get("qa_response", ""),
                    "qa_category": qa_item.get("qa_category", ""),
                    "label": None,
                    "reason": f"Diagnosis failed: {str(e)}",
                    "stage": DiagnosisStage.ERROR.value,
                    "status": DiagnosisStatus.ERROR.value,
                    "answer_correct": False,
                    "diagnosis_mode": "error",
                }, UsageStats())

        def save_processed_item(processed: Optional[Tuple[Dict, UsageStats]]) -> None:
            if processed is None:
                return

            result, item_stats = processed
            results.append(result)
            file_stats.merge(item_stats)

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            _thread_print(f"{prefix}✅ {result['conv_id_question_id']} completed and saved\n")

        effective_qa_threads = max(1, qa_threads)
        if effective_qa_threads <= 1 or len(pending_items) <= 1:
            for task in pending_items:
                save_processed_item(process_qa_item(task))
        else:
            effective_qa_threads = min(effective_qa_threads, len(pending_items))
            _thread_print(f"{prefix}🧵 Start {effective_qa_threads} QA threads inside this file...\n")
            with ThreadPoolExecutor(max_workers=effective_qa_threads) as executor:
                futures = [executor.submit(process_qa_item, task) for task in pending_items]
                for future in as_completed(futures):
                    save_processed_item(future.result())

    except KeyboardInterrupt:
        _thread_print(f"\n{prefix}⚠️  Processing interrupted, saving...\n")
    except Exception as e:
        logging.error(f"{prefix}Error occurred during processing: {str(e)}")
        _thread_print(f"\n{prefix}❌ Error occurred during processing: {str(e)}\n")
    finally:
        if results:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    error_count = sum(1 for result in results if result.get("status") == DiagnosisStatus.ERROR.value)
    completed_count = len(results) - error_count
    _thread_print(
        f"{prefix}🎉 File processing completed: {completed_count} completed, "
        f"{error_count} failed -> {output_file}"
    )
    file_stats.print_summary()
    return len(results), file_stats


def _resolve_input_files(input_args: List[str]) -> List[str]:
    """Resolve input arguments (file paths, directory paths, and glob patterns).

    Args:
        input_args: list of input args

    Returns:
        De-duplicated list of JSON file paths
    """
    files = []
    for path in input_args:
        if os.path.isdir(path):
            files.extend(sorted(glob_module.glob(os.path.join(path, "*.json"))))
        elif os.path.isfile(path):
            files.append(path)
        else:
            expanded = sorted(glob_module.glob(path))
            if expanded:
                files.extend(expanded)
            else:
                logging.warning(f"Path does not exist or no files matched: {path}")
    seen = set()
    unique = []
    for f in files:
        real = os.path.realpath(f)
        if real not in seen:
            seen.add(real)
            unique.append(f)
    return unique


def _generate_output_path(
    input_file: str,
    model: str,
    use_voting: bool,
    num_votes: int,
    output_dir: Optional[str],
    timestamp: str,
) -> str:
    """Generate the output file path for a given input file."""
    input_basename = os.path.splitext(os.path.basename(input_file))[0]
    input_identifier = input_basename.replace(" ", "_").replace("(", "").replace(")", "")

    if output_dir is None:
        output_dir = "data/output/llm_annotation_voting" if use_voting else "data/output/llm_annotation_single"

    os.makedirs(output_dir, exist_ok=True)

    if use_voting:
        filename = f"{input_identifier}_voting_{num_votes}rounds_{model.replace('-', '_')}_{timestamp}.json"
    else:
        filename = f"{input_identifier}_single_{model.replace('-', '_')}_{timestamp}.json"

    return os.path.join(output_dir, filename)


def _print_stage_summary(global_stats: UsageStats):
    """Print per-stage aggregated statistics."""
    stage_stats: Dict[str, Dict] = {}
    for detail in global_stats.call_details:
        stage = detail.get("stage", "unknown")
        if stage not in stage_stats:
            stage_stats[stage] = {
                "calls": 0, "latency": 0.0,
                "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            }
        stage_stats[stage]["calls"] += 1
        stage_stats[stage]["latency"] += detail.get("latency_seconds", 0)
        stage_stats[stage]["prompt_tokens"] += detail.get("prompt_tokens", 0)
        stage_stats[stage]["completion_tokens"] += detail.get("completion_tokens", 0)
        stage_stats[stage]["total_tokens"] += detail.get("total_tokens", 0)

    if stage_stats:
        print(f"\n  📋 Per-stage summary:")
        for stage_name, s in sorted(stage_stats.items()):
            avg_lat = round(s["latency"] / s["calls"], 3) if s["calls"] > 0 else 0
            print(f"     {stage_name}: {s['calls']} calls, "
                  f"total latency {round(s['latency'], 3)}s (avg {avg_lat}s), "
                  f"tokens {s['total_tokens']}")


# ============================================================================
# Main entrypoint
# ============================================================================

def main():
    """Main entrypoint.

    Supports CLI usage:
        python run_diagnosis.py [model] [options]

    Arguments:
        model: model alias (deepseek, gpt4.1, gpt5), default: deepseek
        --voting: enable voting (default)
        --no-voting: disable voting and use a single model
        --num-votes N: voting rounds, default: 3
        -i, --input: input file/dir/glob paths (multiple supported)
        -o, --output-dir: output directory
        -f, --output-file: output filename (single-file mode only)
        -t, --threads: number of worker threads, default: 1
        --qa-threads: number of worker threads for QA items within each file

    Examples:
        python run_diagnosis.py deepseek --no-voting -i file.json
        python run_diagnosis.py deepseek -i data/input/mem0_mem/gpt4omini/ -t 5
        python run_diagnosis.py deepseek -i part1.json part2.json part3.json -t 3
        python run_diagnosis.py deepseek --num-votes 5 -i dir/ -t 5
    """
    import argparse
    import datetime

    parser = argparse.ArgumentParser(
        description="Memory diagnosis system - staged diagnosis for issues in QA pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    model_map = {
        "deepseek": "deepseek",
        "gpt4.1": "gpt-4.1",
        "gpt5": "gpt-5",
    }

    parser.add_argument(
        "model",
        nargs="?",
        default="deepseek",
        choices=list(model_map.keys()),
        help="Model to use (default: deepseek)",
    )
    parser.add_argument(
        "--voting",
        action="store_true",
        default=True,
        help="Enable voting mode (enabled by default)",
    )
    parser.add_argument(
        "--no-voting",
        action="store_true",
        help="Disable voting and use single-model diagnosis",
    )
    parser.add_argument(
        "--num-votes",
        type=int,
        default=3,
        help="Number of voting rounds (default: 3)",
    )
    parser.add_argument(
        "--min-valid-votes",
        type=int,
        default=None,
        help="Minimum successful judgments required in voting mode (default: strict majority)",
    )
    parser.add_argument(
        "-i", "--input",
        nargs="+",
        default=["data/input/mem0_mem/gpt4omini/mem0_dataset_part1.json"],
        help="Input file path(s), directory path(s), or glob pattern(s) (supports multiple)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Output directory path (default: auto-selected by diagnosis mode)",
    )
    parser.add_argument(
        "-f", "--output-file",
        type=str,
        default=None,
        help="Output filename (single-file mode only)",
    )
    parser.add_argument(
        "-t", "--threads",
        type=int,
        default=1,
        help="Number of parallel threads (default: 1, recommended to match input file count)",
    )
    parser.add_argument(
        "--qa-threads",
        type=int,
        default=1,
        help="Number of QA worker threads inside each input file (default: 1)",
    )

    args = parser.parse_args()

    use_voting = args.voting and not args.no_voting
    model = model_map[args.model]
    num_threads = max(1, args.threads)
    qa_threads = max(1, args.qa_threads)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Resolve input files
    input_files = _resolve_input_files(args.input)
    if not input_files:
        print("❌ Error: No valid input files found")
        print(f"💡 Tip: Please check input paths {args.input}")
        return

    # Print startup info
    print("\n" + "=" * 70)
    print("🚀 Memory diagnosis system started")
    print("=" * 70)
    print(f"🤖 Model: {model}")
    print(f"📊 Diagnosis mode: {'Voting (' + str(args.num_votes) + ' rounds)' if use_voting else 'Single-model diagnosis'}")
    print(f"📁 Input files: {len(input_files)}")
    for f in input_files:
        print(f"   - {f}")
    print(f"🧵 File parallel threads: {num_threads}")
    print(f"🧵 QA parallel threads per file: {qa_threads}")
    print(f"⚙️  Config: {DiagnosisConfig()}")
    print("=" * 70 + "\n")

    # Generate output paths for each input file
    file_pairs: List[Tuple[str, str]] = []
    for idx, inp in enumerate(input_files):
        if len(input_files) == 1 and args.output_file:
            out_dir = args.output_dir or ("data/output/llm_annotation_voting" if use_voting else "data/output/llm_annotation_single")
            os.makedirs(out_dir, exist_ok=True)
            out = os.path.join(out_dir, args.output_file)
        else:
            out = _generate_output_path(inp, model, use_voting, args.num_votes, args.output_dir, timestamp)
        file_pairs.append((inp, out))
        print(f"📄 [{idx+1}] {inp}")
        print(f"   → {out}")
    print()

    # ---------------------------------------------------------------
    # Run diagnosis
    # ---------------------------------------------------------------
    global_stats = UsageStats()
    total_processed = 0

    if num_threads <= 1 or len(file_pairs) <= 1:
        # Single-threaded sequential processing
        for inp, out in file_pairs:
            count, stats = process_single_file(
                input_file=inp,
                output_file=out,
                model=model,
                use_voting=use_voting,
                num_votes=args.num_votes,
                min_valid_votes=args.min_valid_votes,
                qa_threads=qa_threads,
                thread_label=os.path.basename(inp),
            )
            total_processed += count
            global_stats.merge(stats)
    else:
        # Multi-threaded parallel processing
        effective_threads = min(num_threads, len(file_pairs))
        print(f"🧵 Start {effective_threads} threads to process {len(file_pairs)} files in parallel...\n")

        futures_map = {}
        with ThreadPoolExecutor(max_workers=effective_threads) as executor:
            for inp, out in file_pairs:
                future = executor.submit(
                    process_single_file,
                    input_file=inp,
                    output_file=out,
                    model=model,
                    use_voting=use_voting,
                    num_votes=args.num_votes,
                    min_valid_votes=args.min_valid_votes,
                    qa_threads=qa_threads,
                    thread_label=os.path.basename(inp),
                )
                futures_map[future] = inp

            for future in as_completed(futures_map):
                inp = futures_map[future]
                try:
                    count, stats = future.result()
                    total_processed += count
                    global_stats.merge(stats)
                    print(f"✅ Thread completed: {os.path.basename(inp)} ({count} questions)")
                except Exception as e:
                    logging.error(f"Thread error while processing {inp}: {str(e)}")
                    print(f"❌ Thread failed: {os.path.basename(inp)}: {str(e)}")

    # ---------------------------------------------------------------
    # Global summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("🎉 All processing completed")
    print("=" * 70)
    print(f"✅ Processed {total_processed} questions in total ({len(file_pairs)} files)")
    for _, out in file_pairs:
        print(f"   📁 {out}")

    print(f"\n{'=' * 70}")
    print(f"📊 Global API call summary")
    print(f"{'=' * 70}")
    global_stats.print_summary()
    _print_stage_summary(global_stats)
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
