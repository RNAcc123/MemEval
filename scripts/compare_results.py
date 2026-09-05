# -*- coding: utf-8 -*-
"""
Compare human annotations against LLM diagnosis results.

This script is the thin CLI entrypoint. All comparison/matching logic lives
in `memeval.analysis` (see `label_matching.py::run_compare`), which is also
what `memeval.cli`'s `compare` command calls directly.
"""

import os
import argparse

from memeval.analysis import run_compare


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare human annotations (human_annotation) and LLM annotations "
            "(llm_annotation_voting), and report match statistics by phase and "
            "by full label."
        )
    )
    parser.add_argument(
        "-H",
        "--human-dir",
        type=str,
        default=os.path.join("data", "input", "human_annotation"),
        help=(
            "Directory containing human-annotation JSON files. Can be an absolute "
            "path, or a path relative to the project root / parent directory of "
            "this script."
        ),
    )
    parser.add_argument(
        "-L",
        "--llm-dir",
        type=str,
        default=os.path.join("data", "output", "llm_annotation_voting", "20251205"),
        help=(
            "Directory containing LLM voting-result JSON files. Can be an absolute "
            "path, or a path relative to the project root / parent directory of "
            "this script."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=os.path.join("data", "output", "evalresult"),
        help=(
            "Output directory. Can be an absolute path, or a path relative to the "
            "project root / parent directory of this script. By default, results "
            "are written under evalresult."
        ),
    )
    args = parser.parse_args()

    # Treat the parent directory of this script as the "project root"
    project_root = os.path.join(os.path.dirname(__file__), "..")

    def resolve(path: str) -> str:
        return path if os.path.isabs(path) else os.path.join(project_root, path)

    run_compare(resolve(args.human_dir), resolve(args.llm_dir), resolve(args.output_dir))


if __name__ == "__main__":
    main()
