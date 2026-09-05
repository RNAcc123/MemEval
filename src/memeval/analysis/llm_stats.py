"""Label-distribution statistics for LLM diagnosis output files.

This module holds the analysis previously embedded in
``scripts/analyze_llm_results.py``: summarizing per-category label
distributions for the top-level (voting_final / single-model) result and
for each individual judge model, plus writing both a human-readable report
and a structured ``metrics.json`` sidecar.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

__all__ = [
    "find_merged_files",
    "load_json",
    "collect_stats",
    "format_and_save",
    "run_analyze",
]


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


def _record_label(stats: Dict[int, dict], cat_int: int, label: Any, labels_seen: Set[str]) -> None:
    s = stats[cat_int]
    s["total_items"] += 1
    if label is None:
        s["no_label_items"] += 1
        return
    lb_str = str(label).strip()
    if not lb_str:
        s["no_label_items"] += 1
        return
    s["labeled_items"] += 1
    s["total_labels"] += 1
    s["label_counts"][lb_str] += 1
    labels_seen.add(lb_str)


def _single_model_name(item: Dict[str, Any]) -> str:
    if item.get("used_model"):
        return str(item["used_model"])
    mode = str(item.get("diagnosis_mode") or "")
    prefix = "single_model_"
    if mode.startswith(prefix):
        return mode[len(prefix):]
    return "single_model"


def collect_stats(files: List[str]) -> Tuple[Dict[int, dict], Dict[str, Dict[int, dict]], List[str], Dict[str, int]]:
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
    coverage = {"completed_records": 0, "excluded_error_records": 0, "excluded_error_votes": 0}

    for fp in files:
        data = load_json(fp)
        if not isinstance(data, list):
            continue

        for item in data:
            if item.get("status", "completed") != "completed":
                coverage["excluded_error_records"] += 1
                continue
            cat = item.get("qa_category")
            if cat is None:
                continue
            try:
                cat_int = int(cat)
            except Exception:
                continue
            if cat_int == 5 or cat_int not in final_stats:
                continue
            coverage["completed_records"] += 1

            # ========================
            # 1) Top-level result stats
            # ========================
            _record_label(final_stats, cat_int, item.get("label"), labels_seen)

            # ========================
            # 2) Per-used_model stats
            # ========================
            voting = item.get("voting_details", {})
            individual = []
            if isinstance(voting, dict):
                individual = voting.get("individual_results", []) or []

            if individual:
                individual_results = individual
            else:
                individual_results = [
                    {
                        "used_model": _single_model_name(item),
                        "label": item.get("label"),
                    }
                ]

            for res in individual_results:
                if res.get("status", "completed") != "completed":
                    coverage["excluded_error_votes"] += 1
                    continue
                used_model = str(res.get("used_model") or "unknown")
                if used_model not in model_stats:
                    model_stats[used_model] = _init_cat_stats()

                _record_label(model_stats[used_model], cat_int, res.get("label"), labels_seen)

    label_list = sorted(labels_seen)
    return final_stats, model_stats, label_list, coverage


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
    source_label: str,
    coverage: Dict[str, int],
) -> None:
    """Generate the final report text and write it to a file."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    lines: List[str] = []
    lines.append(f"Statistics for {source_label}")
    lines.append("Note: Accuracy = (samples without label / total samples) * 100")
    lines.append(
        "Coverage: "
        f"{coverage['completed_records']} completed records; "
        f"{coverage['excluded_error_records']} error records excluded; "
        f"{coverage['excluded_error_votes']} error votes excluded"
    )
    lines.append("")

    # First write top-level results.
    lines.extend(_format_single_table("top_level", final_stats, label_list))

    # Then write tables for each used_model
    for model_name in sorted(model_stats.keys()):
        lines.extend(_format_single_table(model_name, model_stats[model_name], label_list))

    content = "\n".join(lines)
    print(content)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)

    # Keep the human-readable report for compatibility, but make structured
    # metrics the canonical input for future analysis and plotting tools.
    def plain(value: Any) -> Any:
        if isinstance(value, defaultdict):
            return {str(key): plain(item) for key, item in value.items()}
        if isinstance(value, dict):
            return {str(key): plain(item) for key, item in value.items()}
        return value

    metrics_path = os.path.join(os.path.dirname(out_path), "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        json.dump(
            {
                "schema_version": "1.0",
                "source": source_label,
                "coverage": coverage,
                "labels": label_list,
                "final_stats": plain(final_stats),
                "model_stats": plain(model_stats),
            },
            metrics_file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )


def run_analyze(input_dir: str, output_dir: str) -> Optional[str]:
    """Find voting-annotation files under `input_dir`, collect stats, and write
    the report + metrics.json under `output_dir`. Shared by the CLI `analyze`
    command and the `scripts/analyze_llm_results.py` entrypoint.

    Returns:
        The written report path, or None if no input files were found.
    """
    if not os.path.isdir(input_dir):
        print("llm_annotation_voting directory not found:", input_dir)
        return None

    files = find_merged_files(input_dir)
    if not files:
        print("No .json files found in", input_dir)
        return None

    final_stats, model_stats, label_list, coverage = collect_stats(files)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"llm_annotation_stats_{timestamp}.txt")

    format_and_save(final_stats, model_stats, label_list, out_path, input_dir, coverage)
    print("Saved results to:", out_path)
    return out_path
