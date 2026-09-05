#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize overall statistics for LLM annotation files.

This script is the thin CLI entrypoint. The stats-collection and report
formatting logic lives in `memeval.analysis` (see `llm_stats.py`).

Features:
- By default, read voting annotation files under `data/output/llm_annotation_voting`.
- Optionally, use `--input-file` to analyze one or more explicit JSON files,
  including single-model diagnosis outputs under `llm_annotation_single`.
- Only count qa_category ∈ {1,2,3,4} (exclude 5)
- Output tables for the top-level result and each individual `used_model`:
  - counts for each label type (1.1, 1.2, ..., 4.3, 5, …)
  - total samples
  - samples with a label
  - total labels
  - samples with no label
  - accuracy = (no-label samples / total samples) * 100
"""

import argparse
import os
from datetime import datetime

from memeval.analysis import collect_stats, find_merged_files, format_and_save

# Default output directory (can be overridden by CLI args)
DEFAULT_OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "output", "evalresult")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize annotation statistics in llm_annotation_voting "
            "(based on merged_*_gpt4omini_voting_3rounds_gpt_5.json).\n"
            "By default it reads from ../llm_annotation_voting, and you can also "
            "specify a subdirectory with --input-dir."
        )
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        default=os.path.join("data", "output", "llm_annotation_voting"),
        help=(
            "Input directory. Can be an absolute path, or a path relative to the "
            "project root / parent directory of this script. JSON files matching "
            "merged_*_gpt4omini_voting_3rounds_gpt_5.json will be searched."
        ),
    )
    parser.add_argument(
        "--input-file",
        nargs="+",
        default=None,
        help=(
            "Explicit JSON file path(s) to analyze. Can be absolute or relative "
            "to the project root. Use this for llm_annotation_single outputs."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="evalresult",
        help=(
            "Output directory. Can be an absolute path, or a path relative to the "
            "project root / parent directory of this script. "
            "The output filename is llm_annotation_voting_stats_full.txt."
        ),
    )
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    if args.input_file:
        files = [
            path if os.path.isabs(path) else os.path.join(project_root, path)
            for path in args.input_file
        ]
        missing = [path for path in files if not os.path.isfile(path)]
        if missing:
            print("Input JSON file not found:", ", ".join(missing))
            return
        source_label = ", ".join(os.path.basename(path) for path in files)
    else:
        # Resolve input dir: absolute, or relative to project root
        if os.path.isabs(args.input_dir):
            base_dir = args.input_dir
        else:
            base_dir = os.path.join(project_root, args.input_dir)

        if not os.path.isdir(base_dir):
            print("llm_annotation_voting directory not found:", base_dir)
            return

        files = find_merged_files(base_dir)
        if not files:
            print("No .json files found in", base_dir)
            return
        source_label = base_dir

    final_stats, model_stats, label_list, coverage = collect_stats(files)

    # Resolve output dir: absolute, or relative to this script's parent directory
    if os.path.isabs(args.output_dir):
        out_dir = args.output_dir
    else:
        out_dir = os.path.join(project_root, args.output_dir)

    # If user didn't change the default, keep backward-compatible default directory
    if args.output_dir == "evalresult":
        out_dir = DEFAULT_OUT_DIR

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"llm_annotation_stats_{timestamp}.txt")

    format_and_save(final_stats, model_stats, label_list, out_path, source_label, coverage)
    print("Saved results to:", out_path)


if __name__ == "__main__":
    main()
