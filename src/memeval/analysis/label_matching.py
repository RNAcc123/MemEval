"""Human-vs-LLM label/phase matching and confusion-matrix logic.

This module holds the analysis previously embedded in
``scripts/compare_results.py``: comparing human annotations against LLM
voting results by phase (first segment of the dotted label) and by exact
label, plus a phase-level confusion matrix for the voting_final prediction.

It is distinct from :mod:`memeval.analysis.metrics`/``matching`` (generic
record-ID coverage comparisons) -- this module compares *label values*
between a human annotation source and an LLM diagnosis source.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, IO, Optional

from memeval.diagnosis import load_json_file

__all__ = [
    "is_completed_result",
    "analyze_model_label_matching_strict",
    "analyze_model_label_matching_exact",
    "collect_phase_confusion_voting_final",
    "write_phase_confusion_matrix",
    "print_model_matching_results",
    "print_model_label_matching_results",
    "run_compare",
]


def is_completed_result(item: Any) -> bool:
    """Treat legacy records as completed and exclude explicit execution errors."""
    return isinstance(item, dict) and item.get("status", "completed") == "completed"


def analyze_model_label_matching_strict(human_file: str, llm_file: str) -> Dict[str, Any]:
    human_data = load_json_file(human_file)
    llm_data = load_json_file(llm_file)
    llm_dict = {}
    for item in llm_data:
        if is_completed_result(item):
            llm_dict[item["conv_id_question_id"]] = item

    # stats[model][qa_category][phase] = {"total": 0, "matched": 0}
    stats = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: {"total": 0, "matched": 0}))
    )
    # Track all model names (reset per conversation to avoid cross-conversation leakage)
    for conv_id, human_item in human_data.items():
        human_category = str(human_item.get("qa_category", ""))
        human_label = None
        if "labels" in human_item and human_item["labels"]:
            human_label = human_item["labels"][0]
        # Keep the sample even if human_label is empty; treat empty/None/null as special phase "EMPTY"
        if not (human_category and human_category != "5"):
            continue
        if human_label is None or str(human_label).strip().lower() in ("none", "null", ""):
            human_phase = "EMPTY"
        else:
            try:
                human_phase = str(human_label).split('.')[0]
            except Exception:
                human_phase = "EMPTY"
        if conv_id not in llm_dict:
            continue
        llm_item = llm_dict[conv_id]
        voting = llm_item.get("voting_details", {})
        # Collect all models (per-conversation to avoid model_set accumulating across conversations)
        model_set = set()
        for r in voting.get("individual_results", []):
            if not is_completed_result(r):
                continue
            model = r.get("used_model", "unknown")
            model_set.add(model)
        # For each model, add 1 sample (even if its label is missing in this conversation),
        # and also create a pseudo-model name "voting_final" for the outer voting result.
        for model in model_set:
            stats[model][human_category][human_phase]["total"] += 1
        # Treat the outer voting result as a "model" too (for comparison with human annotations).
        # Do not require final_label to be valid/non-empty: always count it in total (per user request).
        final_label = llm_item.get("label", None)
        # Count the outer voting result by phase; treat empty model label as "EMPTY"
        if final_label is None or str(final_label).strip().lower() in ("none", "null", ""):
            final_phase = "EMPTY"
        else:
            try:
                final_phase = str(final_label).split('.')[0]
            except Exception:
                final_phase = "EMPTY"
        stats["voting_final"][human_category][human_phase]["total"] += 1
        # Matching stats
        for r in voting.get("individual_results", []):
            if not is_completed_result(r):
                continue
            model = r.get("used_model", "unknown")
            model_label = r.get("label", None)
            # Treat empty model label as "EMPTY" and compare only by phase
            if model_label is None or str(model_label).strip().lower() in ("none", "null", ""):
                model_phase = "EMPTY"
            else:
                try:
                    model_phase = str(model_label).split('.')[0]
                except Exception:
                    model_phase = "EMPTY"
            if model_phase == human_phase:
                stats[model][human_category][human_phase]["matched"] += 1
        # Whether the outer voting result matches the human annotation (match if equal)
        if final_phase == human_phase:
            stats["voting_final"][human_category][human_phase]["matched"] += 1
    return stats


def analyze_model_label_matching_exact(human_file: str, llm_file: str) -> Dict[str, Any]:
    """
    Exact-match statistics based on the full label (not just the phase number).
    stats_exact[model][qa_category][label] = {"total": x, "matched": y}
    Here, label is the full string (empty/None normalized to "EMPTY").
    """
    human_data = load_json_file(human_file)
    llm_data = load_json_file(llm_file)
    llm_dict = {}
    for item in llm_data:
        if is_completed_result(item):
            llm_dict[item["conv_id_question_id"]] = item

    # stats_exact[model][category][label] = {"total", "matched"}
    stats_exact = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: {"total": 0, "matched": 0}))
    )

    for conv_id, human_item in human_data.items():
        human_category = str(human_item.get("qa_category", ""))
        human_label = None
        if "labels" in human_item and human_item["labels"]:
            human_label = human_item["labels"][0]

        # Only count categories 1-4
        if not (human_category and human_category != "5"):
            continue

        # Normalize human label (empty treated as "EMPTY")
        if human_label is None or str(human_label).strip().lower() in (
            "none",
            "null",
            "",
        ):
            human_label_norm = "EMPTY"
        else:
            human_label_norm = str(human_label).strip()

        if conv_id not in llm_dict:
            continue
        llm_item = llm_dict[conv_id]
        voting = llm_item.get("voting_details", {})

        # Collect all models participating in voting (per conversation)
        model_set = set()
        for r in voting.get("individual_results", []):
            if not is_completed_result(r):
                continue
            model = r.get("used_model", "unknown")
            model_set.add(model)

        # For each model, add 1 sample under this category for this human label
        for model in model_set:
            stats_exact[model][human_category][human_label_norm]["total"] += 1

        # Treat the outer voting result as a "model" too (for comparison with human annotations)
        final_label = llm_item.get("label", None)
        if final_label is None or str(final_label).strip().lower() in (
            "none",
            "null",
            "",
        ):
            final_label_norm = "EMPTY"
        else:
            final_label_norm = str(final_label).strip()
        stats_exact["voting_final"][human_category][human_label_norm]["total"] += 1

        # Match stats (per model)
        for r in voting.get("individual_results", []):
            if not is_completed_result(r):
                continue
            model = r.get("used_model", "unknown")
            model_label = r.get("label", None)
            if model_label is None or str(model_label).strip().lower() in (
                "none",
                "null",
                "",
            ):
                model_label_norm = "EMPTY"
            else:
                model_label_norm = str(model_label).strip()
            if model_label_norm == human_label_norm:
                stats_exact[model][human_category][human_label_norm]["matched"] += 1

        # Match stats (outer voting result)
        if final_label_norm == human_label_norm:
            stats_exact["voting_final"][human_category][human_label_norm]["matched"] += 1

    return stats_exact


def collect_phase_confusion_voting_final(human_file: str, llm_file: str) -> Dict[str, Dict[str, int]]:
    """
    Compute the confusion matrix between human annotations (true phase)
    and voting_final (predicted phase).
    Human phase is the ground truth (rows), voting_final phase is the prediction (columns).
    Returns:
        conf[true_phase][pred_phase] = count
    """
    human_data = load_json_file(human_file)
    llm_data = load_json_file(llm_file)

    llm_dict = {}
    for item in llm_data:
        if is_completed_result(item):
            llm_dict[item["conv_id_question_id"]] = item

    phases = ["1", "2", "3", "4", "EMPTY"]
    # Initialize 5x5 matrix
    conf = {
        tp: {pp: 0 for pp in phases}
        for tp in phases
    }

    for conv_id, human_item in human_data.items():
        human_category = str(human_item.get("qa_category", ""))
        if not (human_category and human_category != "5"):
            # Only count categories 1-4
            continue

        human_label = None
        if "labels" in human_item and human_item["labels"]:
            human_label = human_item["labels"][0]

        # Normalize human phase (empty treated as "EMPTY")
        if human_label is None or str(human_label).strip().lower() in (
            "none",
            "null",
            "",
        ):
            human_phase = "EMPTY"
        else:
            try:
                human_phase = str(human_label).split(".")[0]
            except Exception:
                human_phase = "EMPTY"

        if conv_id not in llm_dict:
            continue
        llm_item = llm_dict[conv_id]

        # voting_final phase
        final_label = llm_item.get("label", None)
        if final_label is None or str(final_label).strip().lower() in (
            "none",
            "null",
            "",
        ):
            final_phase = "EMPTY"
        else:
            try:
                final_phase = str(final_label).split(".")[0]
            except Exception:
                final_phase = "EMPTY"

        if human_phase not in conf:
            conf[human_phase] = {pp: 0 for pp in phases}
        if final_phase not in conf[human_phase]:
            conf[human_phase][final_phase] = 0
        conf[human_phase][final_phase] += 1

    return conf


def write_phase_confusion_matrix(conf_matrix: Dict[str, Dict[str, int]], out_path: str) -> None:
    """
    Write the phase confusion matrix as an ASCII table.
    Each cell contains the row-normalized percentage and the count: "xx.xx% (n)".
    """
    phases = ["1", "2", "3", "4", "EMPTY"]
    true_labels = [f"Phase {p}" if p != "EMPTY" else "EMPTY" for p in phases]
    pred_labels = [f"Phase {p}" if p != "EMPTY" else "EMPTY" for p in phases]

    # First compute each cell's string value and row totals
    rows_cells = []  # Each row is [true_label, cell1, cell2, ...]
    row_totals = {}
    for tp, tp_label in zip(phases, true_labels):
        row = [tp_label]
        total = sum(conf_matrix.get(tp, {}).get(pp, 0) for pp in phases)
        row_totals[tp] = total
        for pp in phases:
            cnt = conf_matrix.get(tp, {}).get(pp, 0)
            rate = (cnt / total * 100) if total > 0 else 0.0
            cell = f"{rate:.2f}% ({cnt})"
            row.append(cell)
        rows_cells.append(row)

    # Compute column widths
    headers = ["True phase"] + pred_labels
    num_cols = len(headers)
    col_widths = [0] * num_cols
    for i, h in enumerate(headers):
        col_widths[i] = max(col_widths[i], len(h))
    for row in rows_cells:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def make_border():
        return "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    border = make_border()

    lines = []
    lines.append(
        "Confusion matrix between human (true phase) and voting_final (predicted phase)"
    )
    lines.append("Rows = human (true) phase, Columns = voting_final (predicted) phase")
    lines.append("")
    lines.append(border)

    # Header row
    header_row = "|" + "|".join(
        f" {headers[i].center(col_widths[i])} " for i in range(num_cols)
    ) + "|"
    lines.append(header_row)
    lines.append(border)

    # Each row
    for row in rows_cells:
        line = "|" + "|".join(
            f" {row[i].center(col_widths[i])} " for i in range(num_cols)
        ) + "|"
        lines.append(line)
        lines.append(border)

    # Append overall info: per-true-phase totals + overall accuracy
    total_samples = 0
    total_correct = 0
    for p in phases:
        total_samples += row_totals.get(p, 0)
        total_correct += conf_matrix.get(p, {}).get(p, 0)

    lines.append("")
    lines.append("Row totals (per true phase):")
    for tp, tp_label in zip(phases, true_labels):
        lines.append(f"- {tp_label}: {row_totals.get(tp, 0)} samples")

    overall_acc = (total_correct / total_samples * 100) if total_samples > 0 else 0.0
    lines.append(f"Overall accuracy (diagonal / all): {overall_acc:.2f}%")

    content = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)


def print_model_matching_results(stats: Dict[str, Any], file: Optional[IO[str]] = None) -> None:
    # Display by phase (1-4) and include an EMPTY column
    phases = ["1", "2", "3", "4", "EMPTY"]
    phase_width = 10
    cat_width = 6
    def make_separator():
        # Reserve an extra column for the overall match rate
        return (
            "+"
            + "-" * (cat_width)
            + ("+" + "-" * phase_width) * (len(phases) + 1)
            + "+"
        )
    def write_and_print(s):
        print(s)
        if file:
            file.write(s + "\n")
    for model in stats:
        write_and_print(f"\nModel: {model}")
        header = f"|{'Cat':^{cat_width}}|"
        for p in phases:
            label = ('Phase '+p) if p != 'EMPTY' else 'EMPTY'
            header += f"{label:^{phase_width}}|"
        # Last column: overall match rate per category (including EMPTY)
        header += f"{'Overall':^{phase_width}}|"
        write_and_print(make_separator())
        write_and_print(header)
        write_and_print(make_separator())
        for category in ["1", "2", "3", "4"]:
            row = f"|{category:^{cat_width}}|"
            for p in phases:
                if p in stats[model][category]:
                    total = stats[model][category][p]["total"]
                    matched = stats[model][category][p]["matched"]
                    rate = (matched / total * 100) if total > 0 else 0
                    row += f"{rate:^{phase_width}.1f}|"
                else:
                    row += f"{'---':^{phase_width}}|"
            # Overall match rate for this category (including EMPTY) in the last column
            if category in stats[model]:
                cat_total_all = sum(
                    stats[model][category][label]["total"]
                    for label in stats[model][category]
                )
                cat_matched_all = sum(
                    stats[model][category][label]["matched"]
                    for label in stats[model][category]
                )
                if cat_total_all > 0:
                    cat_rate_all = cat_matched_all / cat_total_all * 100
                    row += f"{cat_rate_all:^{phase_width}.2f}|"
                else:
                    row += f"{'N/A':^{phase_width}}|"
            else:
                row += f"{'N/A':^{phase_width}}|"
            write_and_print(row)
            write_and_print(make_separator())

        # Overall stats
        all_total = sum(
            stats[model][c][l]["total"] for c in stats[model] for l in stats[model][c]
        )
        all_matched = sum(
            stats[model][c][l]["matched"]
            for c in stats[model]
            for l in stats[model][c]
        )
        all_rate = (all_matched / all_total * 100) if all_total > 0 else 0
        write_and_print(f"Total samples: {all_total}")
        write_and_print(f"Total matches: {all_matched}")
        write_and_print(f"Overall match rate: {all_rate:.2f}%")


def print_model_label_matching_results(stats: Dict[str, Any], file: Optional[IO[str]] = None) -> None:
    """
    Print statistics for exact matching on the full label:
    - Similar to phase-level reporting, for each model draw a Cat × Label match-rate table.
    """

    def write_and_print(s):
        print(s)
        if file:
            file.write(s + "\n")

    for model in stats:
        # Collect all labels observed for this model as columns
        label_set = set()
        for cat in stats[model]:
            for lbl in stats[model][cat]:
                label_set.add(lbl)
        if not label_set:
            continue
        # Put EMPTY at the end
        labels = sorted(
            label_set, key=lambda x: (x == "EMPTY", x)
        )

        label_width = max(12, max(len(l) for l in labels) + 2)
        cat_width = 6

        def make_separator():
            return (
                "+"
                + "-" * cat_width
                + ("+" + "-" * label_width) * len(labels)
                + "+"
            )

        write_and_print(f"\n[Exact label match] Model: {model}")
        # Header row
        header = f"|{'Cat':^{cat_width}}|"
        for lbl in labels:
            header += f"{lbl:^{label_width}}|"
        write_and_print(make_separator())
        write_and_print(header)
        write_and_print(make_separator())

        # One row per category: match rate for each label
        for category in ["1", "2", "3", "4"]:
            row = f"|{category:^{cat_width}}|"
            for lbl in labels:
                if (
                    category in stats[model]
                    and lbl in stats[model][category]
                    and stats[model][category][lbl]["total"] > 0
                ):
                    total = stats[model][category][lbl]["total"]
                    matched = stats[model][category][lbl]["matched"]
                    rate = matched / total * 100 if total > 0 else 0.0
                    row += f"{rate:^{label_width}.1f}|"
                else:
                    row += f"{'---':^{label_width}}|"
            write_and_print(row)
            write_and_print(make_separator())

        # Overall exact match rate per category (including EMPTY)
        for category in ["1", "2", "3", "4"]:
            if category not in stats[model]:
                continue
            # Overall exact match rate including EMPTY
            cat_total_all = sum(
                stats[model][category][lbl]["total"]
                for lbl in stats[model][category]
            )
            cat_matched_all = sum(
                stats[model][category][lbl]["matched"]
                for lbl in stats[model][category]
            )
            if cat_total_all > 0:
                cat_rate_all = cat_matched_all / cat_total_all * 100
                write_and_print(
                    f"Category {category} exact label match rate (including EMPTY): {cat_rate_all:.2f}%"
                )
            else:
                write_and_print(
                    f"Category {category} exact label match rate (including EMPTY): N/A"
                )

        # Overall stats
        all_total = sum(
            stats[model][c][lbl]["total"]
            for c in stats[model]
            for lbl in stats[model][c]
        )
        all_matched = sum(
            stats[model][c][lbl]["matched"]
            for c in stats[model]
            for lbl in stats[model][c]
        )
        overall_rate = (all_matched / all_total * 100) if all_total > 0 else 0
        write_and_print(f"Total samples: {all_total}")
        write_and_print(f"Total matches: {all_matched}")
        write_and_print(f"Overall exact match rate: {overall_rate:.2f}%")


def run_compare(human_base: str, llm_base: str, out_dir: str) -> Dict[str, str]:
    """Auto-pair human/LLM files under the given directories, merge stats across
    all pairs, and write the three report files. Shared by the CLI `compare`
    command and the `scripts/compare_results.py` entrypoint.

    Returns:
        Dict with keys "phase", "label", "confusion" mapping to the written
        file paths.
    """
    os.makedirs(out_dir, exist_ok=True)

    human_files = sorted(
        f for f in os.listdir(human_base) if f.startswith("human_dataset_part") and f.endswith(".json")
    )
    llm_files = sorted(f for f in os.listdir(llm_base) if f.endswith(".json"))

    file_pairs = []
    print("Matching files...")
    print(f"Human dir: {human_base}")
    print(f"LLM dir: {llm_base}")

    for h_file in human_files:
        match = re.search(r"part(\d+)", h_file)
        if not match:
            continue
        part_num = match.group(1)
        matching_llm_files = [f for f in llm_files if f"part{part_num}" in f]
        if not matching_llm_files:
            print(f"⚠️  Warning: No matching LLM output file found for {h_file} (part {part_num})")
            continue
        best_match = matching_llm_files[-1]
        file_pairs.append((h_file, best_match))
        print(f"✅ Matched: {h_file} <-> {best_match}")

    if not file_pairs:
        print("❌ No matched file pairs found")
        return {}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    merged_stats_phase: Dict[str, Any] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: {"total": 0, "matched": 0}))
    )
    merged_stats_label: Dict[str, Any] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: {"total": 0, "matched": 0}))
    )
    merged_confusion: Dict[str, Any] = defaultdict(lambda: defaultdict(int))

    for human_file, llm_file in file_pairs:
        human_path = os.path.join(human_base, human_file)
        llm_path = os.path.join(llm_base, llm_file)

        stats_phase = analyze_model_label_matching_strict(human_path, llm_path)
        for model in stats_phase:
            for cat in stats_phase[model]:
                for label in stats_phase[model][cat]:
                    merged_stats_phase[model][cat][label]["total"] += stats_phase[model][cat][label]["total"]
                    merged_stats_phase[model][cat][label]["matched"] += stats_phase[model][cat][label]["matched"]

        stats_label = analyze_model_label_matching_exact(human_path, llm_path)
        for model in stats_label:
            for cat in stats_label[model]:
                for lbl in stats_label[model][cat]:
                    merged_stats_label[model][cat][lbl]["total"] += stats_label[model][cat][lbl]["total"]
                    merged_stats_label[model][cat][lbl]["matched"] += stats_label[model][cat][lbl]["matched"]

        conf = collect_phase_confusion_voting_final(human_path, llm_path)
        for tp in conf:
            for pp in conf[tp]:
                merged_confusion[tp][pp] += conf[tp][pp]

    phase_out_path = os.path.join(out_dir, f"model_phase_{timestamp}.txt")
    with open(phase_out_path, "w", encoding="utf-8") as f:
        print_model_matching_results(merged_stats_phase, file=f)
    print_model_matching_results(merged_stats_phase)

    label_out_path = os.path.join(out_dir, f"model_label_exact_{timestamp}.txt")
    with open(label_out_path, "w", encoding="utf-8") as f:
        print_model_label_matching_results(merged_stats_label, file=f)
    print_model_label_matching_results(merged_stats_label)

    print(f"Phase-level results saved to: {phase_out_path}")
    print(f"Exact-label results saved to: {label_out_path}")

    conf_out_path = os.path.join(out_dir, f"human_vs_voting_final_phase_confusion_{timestamp}.txt")
    write_phase_confusion_matrix(merged_confusion, conf_out_path)
    print(f"Human vs voting_final phase confusion matrix saved to: {conf_out_path}")

    return {"phase": phase_out_path, "label": label_out_path, "confusion": conf_out_path}
