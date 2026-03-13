#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize overall statistics for human annotations under the `human_annotation` directory.

Features:
- Read `../human_annotation/annotation_*_gpt4omini_fixed.json`
- Only count qa_category ∈ {1,2,3,4} (exclude 5)
- For each qa_category, compute:
  - counts for each label type (1.1, 1.2, ..., 4.3, 5)
  - total samples
  - samples with at least one label
  - total number of labels (multi-labels summed)
  - samples with no label
  - accuracy = (no-label samples / total samples) * 100
- Output an ASCII table and include overall label counts
- Save results to: `evalresult/human_annotation_stats_full.txt`
"""

import json
import os
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Set


BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "input", "human_annotation")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "output", "evalresult")


def find_annotation_files(dirpath: str) -> List[str]:
    """Find `human_dataset_part*.json` files under the human_annotation directory."""
    files: List[str] = []
    for fn in sorted(os.listdir(dirpath)):
        if fn.startswith("human_dataset_part") and fn.endswith(".json"):
            files.append(os.path.join(dirpath, fn))
    return files


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_stats(files: List[str]) -> Tuple[Dict[int, dict], List[str]]:
    """
    Compute label distribution and aggregate counts for each qa_category.

    Returns:
        stats: aggregated statistics per category
        label_list: all labels observed (sorted)
    """
    # stats[cat] structure:
    # {
    #   'total_items': int,
    #   'labeled_items': int,
    #   'total_labels': int,
    #   'no_label_items': int,
    #   'label_counts': defaultdict(int),
    # }
    stats: Dict[int, dict] = {}
    for cat in (1, 2, 3, 4):
        stats[cat] = {
            "total_items": 0,
            "labeled_items": 0,
            "total_labels": 0,
            "no_label_items": 0,
            "label_counts": defaultdict(int),
        }

    labels_seen: Set[str] = set()

    for fp in files:
        data = load_json(fp)
        # human_annotation is typically a dict: key -> item
        if isinstance(data, dict):
            it = data.items()
        else:
            # Fallback: if it becomes a list in the future, iterate items one by one
            it = enumerate(data)

        for _, h in it:
            cat = h.get("qa_category")
            if cat is None:
                continue
            try:
                cat_int = int(cat)
            except Exception:
                continue
            if cat_int == 5 or cat_int not in stats:
                continue

            s = stats[cat_int]
            s["total_items"] += 1

            human_labels = h.get("labels") or []
            if human_labels:
                s["labeled_items"] += 1
                s["total_labels"] += len(human_labels)
                for lb in human_labels:
                    if lb is None:
                        continue
                    lb_str = str(lb).strip()
                    if not lb_str:
                        continue
                    labels_seen.add(lb_str)
                    s["label_counts"][lb_str] += 1
            else:
                s["no_label_items"] += 1

    label_list = sorted(labels_seen)
    return stats, label_list


def format_and_save(stats: Dict[int, dict], label_list: List[str], out_path: str) -> None:
    """Generate the ASCII table in the predefined format and save to out_path."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    lines: List[str] = []
    lines.append("Human annotation label statistics (by qa_category, excluding 5, with per-label counts)")
    lines.append("Note: Accuracy = (samples without label / total samples) * 100")
    lines.append("")

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

    # Compute column widths
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

    content = "\n".join(lines)
    print(content)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)


def main() -> None:
    if not os.path.isdir(BASE_DIR):
        print("human_annotation directory not found:", BASE_DIR)
        return

    files = find_annotation_files(BASE_DIR)
    if not files:
        print("No annotation_*_gpt4omini_fixed.json files found in", BASE_DIR)
        return

    stats, label_list = collect_stats(files)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUT_DIR, f"human_annotation_stats_{timestamp}.txt")
    
    format_and_save(stats, label_list, out_path)
    print("Saved results to:", out_path)


if __name__ == "__main__":
    main()


