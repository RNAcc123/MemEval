#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize overall statistics for LLM annotations under the `llm_annotation_voting` directory
(based on merged_*_voting_3rounds_gpt_5.json).

Features:
- By default, read `merged_*_gpt4omini_voting_3rounds_gpt_5.json` under `../llm_annotation_voting`
- Optionally, use `--input-dir` to restrict stats to a specific subfolder
- Only count qa_category ∈ {1,2,3,4} (exclude 5)
- Output tables for both the "final voting" result and each individual `used_model`:
  - counts for each label type (1.1, 1.2, ..., 4.3, 5, …)
  - total samples
  - samples with a label
  - total labels (multi-labels summed; final voting has 0/1)
  - samples with no label
  - accuracy = (no-label samples / total samples) * 100
- Save results to: `evalresult/llm_annotation_voting_stats_full.txt`
"""

import argparse
import json
import os
from datetime import datetime
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple


# Default output directory (can be overridden by CLI args)
DEFAULT_OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "output", "evalresult")


def find_merged_files(dirpath: str) -> List[str]:
    """
    Find JSON files to process under the given directory.
    Strategy: select all `.json` files whose filename contains 'voting_annotation_'.
    """
    files: List[str] = []
    for fn in sorted(os.listdir(dirpath)):
        full_path = os.path.join(dirpath, fn)
        if os.path.isfile(full_path) and "voting_annotation_" in fn and fn.lower().endswith(".json"):
            files.append(full_path)
    return files


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _init_cat_stats() -> Dict[int, dict]:
    """Initialize the per-category statistics structure."""
    stats: Dict[int, dict] = {}
    for cat in (1, 2, 3, 4):
        stats[cat] = {
            "total_items": 0,
            "labeled_items": 0,
            "total_labels": 0,
            "no_label_items": 0,
            "label_counts": defaultdict(int),
        }
    return stats


def collect_stats(files: List[str]) -> Tuple[Dict[int, dict], Dict[str, Dict[int, dict]], List[str]]:
    """
    Compute label distributions and aggregate counts for the final voting result
    and for each `used_model`.

    Returns:
        final_stats: per-category stats for the final voting result
        model_stats: model -> per-category stats
        label_list: all labels observed (sorted)
    """
    final_stats: Dict[int, dict] = _init_cat_stats()
    model_stats: Dict[str, Dict[int, dict]] = {}
    labels_seen: Set[str] = set()

    for fp in files:
        data = load_json(fp)
        if not isinstance(data, list):
            continue

        for item in data:
            cat = item.get("qa_category")
            if cat is None:
                continue
            try:
                cat_int = int(cat)
            except Exception:
                continue
            if cat_int == 5 or cat_int not in final_stats:
                continue

            # ========================
            # 1) Final voting stats
            # ========================
            fs = final_stats[cat_int]
            fs["total_items"] += 1

            final_label = item.get("label")
            if final_label is None:
                fs["no_label_items"] += 1
            else:
                lb_str = str(final_label).strip()
                if lb_str:
                    fs["labeled_items"] += 1
                    fs["total_labels"] += 1
                    fs["label_counts"][lb_str] += 1
                    labels_seen.add(lb_str)
                else:
                    fs["no_label_items"] += 1

            # ========================
            # 2) Per-used_model stats
            # ========================
            voting = item.get("voting_details", {})
            individual = []
            if isinstance(voting, dict):
                individual = voting.get("individual_results", []) or []

            for res in individual:
                used_model = str(res.get("used_model") or "unknown")
                if used_model not in model_stats:
                    model_stats[used_model] = _init_cat_stats()

                ms = model_stats[used_model][cat_int]
                ms["total_items"] += 1

                lb = res.get("label")
                if lb is None:
                    ms["no_label_items"] += 1
                else:
                    lb_str = str(lb).strip()
                    if lb_str:
                        ms["labeled_items"] += 1
                        ms["total_labels"] += 1
                        ms["label_counts"][lb_str] += 1
                        labels_seen.add(lb_str)
                    else:
                        ms["no_label_items"] += 1

    label_list = sorted(labels_seen)
    return final_stats, model_stats, label_list


def _format_single_table(
    title: str,
    stats: Dict[int, dict],
    label_list: List[str],
) -> List[str]:
    """Generate table text lines for one entry (voting_final or a specific used_model)."""
    lines: List[str] = []
    lines.append(f"Model: {title}")

    header = ["Cat"] + label_list + [
        "Total samples",
        "Samples with label",
        "Total label count",
        "Samples without label",
        "Accuracy (%)",
    ]

    # Build rows (one row per category)
    rows: List[List[str]] = []
    for cat in (1, 2, 3, 4):
        s = stats[cat]
        total = s["total_items"]
        labeled = s["labeled_items"]
        total_labels = s["total_labels"]
        no_label = s["no_label_items"]
        acc = (no_label / total * 100) if total else 0.0

        row: List[str] = [str(cat)]
        for lb in label_list:
            row.append(str(s["label_counts"].get(lb, 0)))
        row.extend(
            [
                str(total),
                str(labeled),
                str(total_labels),
                str(no_label),
                f"{acc:.2f}",
            ]
        )
        rows.append(row)

    # Compute an overall row (aggregate categories 1-4) as the last table row
    total_items_all = sum(stats[c]["total_items"] for c in stats)
    labeled_items_all = sum(stats[c]["labeled_items"] for c in stats)
    total_labels_all = sum(stats[c]["total_labels"] for c in stats)
    no_label_all = sum(stats[c]["no_label_items"] for c in stats)
    acc_all = (no_label_all / total_items_all * 100) if total_items_all else 0.0

    # overall per-label counts
    overall_label_counts: Dict[str, int] = defaultdict(int)
    for c in stats:
        for lb, cnt in stats[c]["label_counts"].items():
            overall_label_counts[lb] += cnt

    overall_row: List[str] = ["Overall"]
    for lb in label_list:
        overall_row.append(str(overall_label_counts.get(lb, 0)))
    overall_row.extend(
        [
            str(total_items_all),
            str(labeled_items_all),
            str(total_labels_all),
            str(no_label_all),
            f"{acc_all:.2f}",
        ]
    )
    rows.append(overall_row)

    # Compute column widths (including the overall row)
    cols = list(zip(header, *rows))
    col_widths = [max(len(x) for x in col) for col in cols]

    def make_border() -> str:
        return "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    border = make_border()

    lines.append(border)
    header_row = (
        "|" + "|".join(f" {header[i].ljust(col_widths[i])} " for i in range(len(header))) + "|"
    )
    lines.append(header_row)
    lines.append(border)

    for r in rows:
        row_str = "|" + "|".join(
            f" {r[i].rjust(col_widths[i])} " for i in range(len(r))
        ) + "|"
        lines.append(row_str)
        lines.append(border)

    return lines


def format_and_save(
    final_stats: Dict[int, dict],
    model_stats: Dict[str, Dict[int, dict]],
    label_list: List[str],
    out_path: str,
) -> None:
    """Generate the final report text and write it to a file."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    lines: List[str] = []
    lines.append(
        "Statistics for llm_annotation_voting (merged_*_gpt4omini_voting_3rounds_gpt_5.json)"
    )
    lines.append("Note: Accuracy = (samples without label / total samples) * 100")
    lines.append("")

    # First write final voting results
    lines.extend(_format_single_table("voting_final", final_stats, label_list))

    # Then write tables for each used_model
    for model_name in sorted(model_stats.keys()):
        lines.extend(_format_single_table(model_name, model_stats[model_name], label_list))

    content = "\n".join(lines)
    print(content)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)


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

    # Resolve input dir: absolute, or relative to this script's parent directory
    if os.path.isabs(args.input_dir):
        base_dir = args.input_dir
    else:
        base_dir = os.path.join(os.path.dirname(__file__), "..", args.input_dir)

    if not os.path.isdir(base_dir):
        print("llm_annotation_voting directory not found:", base_dir)
        return

    files = find_merged_files(base_dir)
    if not files:
        print("No .json files found in", base_dir)
        return

    final_stats, model_stats, label_list = collect_stats(files)

    # Resolve output dir: absolute, or relative to this script's parent directory
    if os.path.isabs(args.output_dir):
        out_dir = args.output_dir
    else:
        out_dir = os.path.join(os.path.dirname(__file__), "..", args.output_dir)

    # If user didn't change the default, keep backward-compatible default directory
    if args.output_dir == "evalresult":
        out_dir = DEFAULT_OUT_DIR

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"llm_annotation_voting_stats_{timestamp}.txt")

    format_and_save(final_stats, model_stats, label_list, out_path)
    print("Saved results to:", out_path)


if __name__ == "__main__":
    main()


