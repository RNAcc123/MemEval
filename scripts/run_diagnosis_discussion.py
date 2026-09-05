"""
Memory diagnosis system - multi-model discussion version

This script is the thin CLI/driver-loop entrypoint. The discussion prompts,
consensus/voting engine, and dataclasses live in `memeval.diagnosis.discussion`:
- Stage 0: Consistency check (3-model discussion)
- Stage 1: Memory extraction diagnosis (3-model discussion)
- Stage 2: Memory update diagnosis (3-model discussion)
- Stage 3: Memory retrieval diagnosis (3-model discussion)
- Stage 4: Reasoning diagnosis (3-model discussion)

Within each stage, three models first make independent judgments, then run
multiple discussion rounds to reach consensus or decide by voting.
"""

# Standard library imports
import argparse
import datetime
import json
import logging
import os

from memeval.config import DiagnosisConfig
from memeval.diagnosis import analyze_qa_pair_with_discussion, load_json_file
from memeval.schema import (
    DIAGNOSIS_SCHEMA_VERSION,
    DiagnosisStage,
    DiagnosisStatus,
    MemoryData,
    validate_trace_dataset,
)


# ============================================================================
# Main entrypoint
# ============================================================================

def main():
    """Main entrypoint.

    Supports CLI arguments:
        python run_diagnosis_discussion.py [options]

    Arguments:
        --max-rounds N: max discussion rounds per stage (default: 3)
        --models: models participating in discussions (default: deepseek gpt-4.1 gpt-5)
        -i, --input: input file path
        -o, --output-dir: output directory
        -f, --output-file: output filename

    Examples:
        python run_diagnosis_discussion.py --max-rounds 3
        python run_diagnosis_discussion.py --models deepseek gpt-4.1 gpt-5
        python run_diagnosis_discussion.py -i data/input.json -o results/
    """
    parser = argparse.ArgumentParser(
        description="Memory diagnosis system - multi-model staged discussion edition",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help="Maximum discussion rounds per stage (default: 3)"
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["deepseek", "gpt-4.1", "gpt-5"],
        help="List of models participating in discussion (default: deepseek gpt-4.1 gpt-5)"
    )

    parser.add_argument(
        "-i", "--input",
        type=str,
        default="data/input/mem0_mem/sample/sampled_qa_50.json",
        help="Input file path"
    )

    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="data/output/llm_annotation_discussion",
        help="Output directory path"
    )

    parser.add_argument(
        "-f", "--output-file",
        type=str,
        default=None,
        help="Output filename (default: auto-generated)"
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("🚀 Memory diagnosis system - multi-model staged discussion edition started")
    print("="*70)
    print(f"🤖 Participating models: {', '.join(args.models)}")
    print(f"🔄 Max rounds per stage: {args.max_rounds}")
    print(f"⚙️  Config: {DiagnosisConfig()}")
    print("="*70 + "\n")

    input_file = args.input
    input_basename = os.path.splitext(os.path.basename(input_file))[0]
    input_identifier = input_basename.replace(" ", "_").replace("(", "").replace(")", "")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.output_file:
        output_filename = args.output_file
    else:
        models_str = "_".join([m.replace("-", "").replace(".", "") for m in args.models])
        output_filename = f"{input_identifier}_discussion_{args.max_rounds}rounds_{models_str}_{timestamp}.json"

    output_file = os.path.join(output_dir, output_filename)

    print(f"📁 Input file: {input_file}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📁 Output file: {output_file}\n")

    if not os.path.exists(input_file):
        print(f"❌ Error: Input file does not exist: {input_file}")
        return

    try:
        data = validate_trace_dataset(load_json_file(input_file))
        print(f"✅ Loaded {len(data)} conversations successfully\n")
    except Exception as e:
        logging.error(f"Failed to load input file: {str(e)}")
        print(f"❌ Error: Failed to parse input file: {str(e)}")
        return

    # Load previously processed results (supports resume)
    results = []
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                results = json.load(f)
                logging.info(f"Loaded {len(results)} historical results")
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logging.warning(f"Failed to load historical results: {str(e)}, starting from scratch")
            results = []

    processed_items = {item["conv_id_question_id"] for item in results}

    try:
        total_convs = len(data)
        print(f"📊 Start processing, total conversations to analyze: {total_convs}\n")

        for conv_idx, (conv_id, qa_list) in enumerate(data.items(), 1):
            print(f"\n{'='*70}")
            print(f"📝 Processing conversation {conv_id} ({conv_idx}/{total_convs})")
            print(f"{'='*70}\n")

            for qa_idx, qa_item in enumerate(qa_list, 1):
                # Prefer the index from original_source; otherwise use list index
                original_source = qa_item.get("original_source", {})
                if original_source and original_source.get("qa_index") is not None:
                    item_id = f"{conv_id}_{original_source['qa_index']}"
                else:
                    item_id = f"{conv_id}_{qa_idx-1}"

                if item_id in processed_items:
                    print(f"⏭️  Skip already processed question: {item_id}\n")
                    continue

                print(f"🔍 Start processing question {qa_idx}/{len(qa_list)}: {item_id}")

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

                    analysis = analyze_qa_pair_with_discussion(
                        qa_question=qa_item["qa_question"],
                        qa_answer=qa_item["qa_answer"],
                        qa_response=qa_item["qa_response"],
                        subjects=subjects,
                        models=args.models,
                        max_rounds=args.max_rounds
                    )

                    # Capture original location info
                    original_source = qa_item.get("original_source", {})
                    if original_source:
                        original_id = f"{original_source.get('file', '')}_{original_source.get('key', '')}_{original_source.get('qa_index', '')}"
                    else:
                        original_id = item_id

                    result = {
                        "schema_version": DIAGNOSIS_SCHEMA_VERSION,
                        "conv_id_question_id": item_id,
                        "original_id": original_id,
                        "original_source": original_source,
                        "qa_question": qa_item["qa_question"],
                        "qa_answer": qa_item["qa_answer"],
                        "qa_response": qa_item["qa_response"],
                        "qa_category": qa_item.get("qa_category", ""),
                        "label": analysis["label"],
                        "reason": analysis["reason"],
                        "stage": analysis["stage"],
                        "status": analysis["status"],
                        "answer_correct": analysis["answer_correct"],
                        "diagnosis_mode": f"discussion_{args.max_rounds}rounds_per_stage",
                        "consensus_reached": analysis["consensus_reached"],
                        "total_stage_rounds": analysis["total_stage_rounds"],
                        "stage_results": analysis["stage_results"]
                    }

                    results.append(result)

                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)

                    print(f"✅ Question {item_id} processed and saved\n")

                except Exception as e:
                    logging.error(f"Error while processing question {item_id}: {str(e)}")
                    print(f"❌ Failed to process question {item_id}: {str(e)}\n")
                    results.append({
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
                    })
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                    continue

    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted, saving...\n")
    except Exception as e:
        logging.error(f"Unexpected error during processing: {str(e)}")
        print(f"\n❌ Error occurred during processing: {str(e)}\n")
    finally:
        if results:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            consensus_count = sum(1 for r in results if r.get("consensus_reached", False))
            error_count = sum(1 for r in results if r.get("status") == DiagnosisStatus.ERROR.value)

            print("\n" + "="*70)
            print("🎉 Processing completed")
            print("="*70)
            print(f"✅ Total processed questions: {len(results)}")
            print(f"🤝 Consensus reached: {consensus_count}/{len(results)} ({100*consensus_count/len(results):.1f}%)")
            print(f"❌ Failed diagnoses: {error_count}/{len(results)}")
            print(f"📁 Results saved to: {output_file}")
            print("="*70 + "\n")


if __name__ == "__main__":
    main()
