# -*- coding: utf-8 -*-
import json
import os
import argparse
from collections import defaultdict

def load_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def analyze_model_label_matching_strict(human_file, llm_file):
    human_data = load_json_file(human_file)
    llm_data = load_json_file(llm_file)
    llm_dict = {}
    for item in llm_data:
        if isinstance(item, dict):
            llm_dict[item["conv_id_question_id"]] = item

    # stats[model][qa_category][phase] = {"total": 0, "matched": 0}
    stats = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: {"total": 0, "matched": 0}))
    )
    # 记录所有模型名（注意：按对话重置，避免跨对话污染）
    for conv_id, human_item in human_data.items():
        human_category = str(human_item.get("qa_category", ""))
        human_label = None
        if "labels" in human_item and human_item["labels"]:
            human_label = human_item["labels"][0]
        # 即使 human_label 为空也保留样本；把空/None/null 视为特殊阶段 'EMPTY'
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
        # 统计所有模型（每个对话单独统计，防止之前的 model_set 跨对话累积）
        model_set = set()
        for r in voting.get("individual_results", []):
            model = r.get("used_model", "unknown")
            model_set.add(model)
        # 对每个模型都+1（即使该模型在本对话没有可用 label 也算样本），并且同时为最外层投票结果创建一个伪模型名 'voting_final'
        for model in model_set:
            stats[model][human_category][human_phase]["total"] += 1
        # 最外层投票结果也作为一个“模型”统计（用于对比人工标注）
        # 不再要求 final_label 有效或非空：无论 final_label 是什么，都计入 total（按照用户要求）
        final_label = llm_item.get("label", None)
        # 统计最外层投票结果按阶段；模型空 label 视为 'EMPTY'
        if final_label is None or str(final_label).strip().lower() in ("none", "null", ""):
            final_phase = "EMPTY"
        else:
            try:
                final_phase = str(final_label).split('.')[0]
            except Exception:
                final_phase = "EMPTY"
        stats["voting_final"][human_category][human_phase]["total"] += 1
        # 匹配统计
        for r in voting.get("individual_results", []):
            model = r.get("used_model", "unknown")
            model_label = r.get("label", None)
            # 将模型空 label 也视为 'EMPTY'，只按阶段比较
            if model_label is None or str(model_label).strip().lower() in ("none", "null", ""):
                model_phase = "EMPTY"
            else:
                try:
                    model_phase = str(model_label).split('.')[0]
                except Exception:
                    model_phase = "EMPTY"
            if model_phase == human_phase:
                stats[model][human_category][human_phase]["matched"] += 1
        # 统计最外层投票结果是否与人工标注匹配（仅在字符串相等时计为匹配）
        if final_phase == human_phase:
            stats["voting_final"][human_category][human_phase]["matched"] += 1
    return stats


def analyze_model_label_matching_exact(human_file, llm_file):
    """
    基于完整 label 的精确匹配统计（不再只看阶段号）。
    stats_exact[model][qa_category][label] = {"total": x, "matched": y}
    这里的 label 为完整字符串（空/None 归一为 'EMPTY'）。
    """
    human_data = load_json_file(human_file)
    llm_data = load_json_file(llm_file)
    llm_dict = {}
    for item in llm_data:
        if isinstance(item, dict):
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

        # 只统计类别 1-4
        if not (human_category and human_category != "5"):
            continue

        # 归一化人工 label（空视为 'EMPTY'）
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

        # 统计所有参与投票的模型（每个对话单独统计）
        model_set = set()
        for r in voting.get("individual_results", []):
            model = r.get("used_model", "unknown")
            model_set.add(model)

        # 对每个模型都+1：该类别下，这个“人工 label”是一条样本
        for model in model_set:
            stats_exact[model][human_category][human_label_norm]["total"] += 1

        # 最外层投票结果也作为一个“模型”统计（用于对比人工标注）
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

        # 匹配统计（各个模型）
        for r in voting.get("individual_results", []):
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

        # 匹配统计（最外层投票结果）
        if final_label_norm == human_label_norm:
            stats_exact["voting_final"][human_category][human_label_norm]["matched"] += 1

    return stats_exact


def collect_phase_confusion_voting_final(human_file, llm_file):
    """
    统计人工标注（true phase）与 voting_final（predicted phase）之间的混淆矩阵。
    人工 phase 作为真实值（行），voting_final phase 作为预测值（列）。
    返回:
        conf[true_phase][pred_phase] = count
    """
    human_data = load_json_file(human_file)
    llm_data = load_json_file(llm_file)

    llm_dict = {}
    for item in llm_data:
        if isinstance(item, dict):
            llm_dict[item["conv_id_question_id"]] = item

    phases = ["1", "2", "3", "4", "EMPTY"]
    # 初始化 5x5 矩阵
    conf = {
        tp: {pp: 0 for pp in phases}
        for tp in phases
    }

    for conv_id, human_item in human_data.items():
        human_category = str(human_item.get("qa_category", ""))
        if not (human_category and human_category != "5"):
            # 只统计类别 1-4
            continue

        human_label = None
        if "labels" in human_item and human_item["labels"]:
            human_label = human_item["labels"][0]

        # 归一化人工 phase（空视为 'EMPTY'）
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

        # voting_final 的 phase
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


def write_phase_confusion_matrix(conf_matrix, out_path):
    """
    将 phase 混淆矩阵写成 ASCII 表格。
    每个单元格包含：该预测 phase 在该真实 phase 下的占比（按行归一化）和样本数："xx.xx% (n)"。
    """
    phases = ["1", "2", "3", "4", "EMPTY"]
    true_labels = [f"Phase {p}" if p != "EMPTY" else "EMPTY" for p in phases]
    pred_labels = [f"Phase {p}" if p != "EMPTY" else "EMPTY" for p in phases]

    # 先计算每个单元格的字符串表示和行总数
    rows_cells = []  # 每行是 [true_label, cell1, cell2, ...]
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

    # 计算列宽
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

    # 头行
    header_row = "|" + "|".join(
        f" {headers[i].center(col_widths[i])} " for i in range(num_cols)
    ) + "|"
    lines.append(header_row)
    lines.append(border)

    # 每一行
    for row in rows_cells:
        line = "|" + "|".join(
            f" {row[i].center(col_widths[i])} " for i in range(num_cols)
        ) + "|"
        lines.append(line)
        lines.append(border)

    # 追加总体信息：每个真实 phase 的样本总数 + overall accuracy
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

def print_model_matching_results(stats, file=None):
    # 现在按阶段展示（1-4）并包含 EMPTY 列
    phases = ["1", "2", "3", "4", "EMPTY"]
    phase_width = 10
    cat_width = 6
    def make_separator():
        # 额外预留一列用于“总匹配率”
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
        write_and_print(f"\n模型: {model}")
        header = f"|{'Cat':^{cat_width}}|"
        for p in phases:
            label = ('Phase '+p) if p != 'EMPTY' else 'EMPTY'
            header += f"{label:^{phase_width}}|"
        # 最后一列为每个 Cat 的总体匹配率（含 EMPTY）
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
            # 该类别的总体匹配率（含 EMPTY），写在最后一列
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

        # 总体统计
        all_total = sum(
            stats[model][c][l]["total"] for c in stats[model] for l in stats[model][c]
        )
        all_matched = sum(
            stats[model][c][l]["matched"]
            for c in stats[model]
            for l in stats[model][c]
        )
        all_rate = (all_matched / all_total * 100) if all_total > 0 else 0
        write_and_print(f"总样本数: {all_total}")
        write_and_print(f"总匹配数: {all_matched}")
        write_and_print(f"总体匹配率: {all_rate:.2f}%")


def print_model_label_matching_results(stats, file=None):
    """
    输出“完整 label 精确匹配”的统计结果：
    - 像 Phase 一样，按模型画出 Cat × Label 的匹配率表格
    """

    def write_and_print(s):
        print(s)
        if file:
            file.write(s + "\n")

    for model in stats:
        # 收集该模型下所有出现过的 label，用作列
        label_set = set()
        for cat in stats[model]:
            for lbl in stats[model][cat]:
                label_set.add(lbl)
        if not label_set:
            continue
        # 将 EMPTY 放在最后
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

        write_and_print(f"\n[Label 精确匹配] 模型: {model}")
        # 表头
        header = f"|{'Cat':^{cat_width}}|"
        for lbl in labels:
            header += f"{lbl:^{label_width}}|"
        write_and_print(make_separator())
        write_and_print(header)
        write_and_print(make_separator())

        # 每个类别一行：各 label 的匹配率
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

        # 每一类 cat 的总体精确匹配率（包含 EMPTY）
        for category in ["1", "2", "3", "4"]:
            if category not in stats[model]:
                continue
            # 含 EMPTY 的总体精确匹配率
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
                    f"Category {category}（含 EMPTY）label 精确匹配率: {cat_rate_all:.2f}%"
                )
            else:
                write_and_print(
                    f"Category {category}（含 EMPTY）label 精确匹配率: N/A"
                )

        # 总体统计
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
        write_and_print(f"总样本数: {all_total}")
        write_and_print(f"总匹配数: {all_matched}")
        write_and_print(f"总体精确匹配率: {overall_rate:.2f}%")

def main():
    parser = argparse.ArgumentParser(
        description=(
            "对比人工标注（human_annotation）与大模型标注（llm_annotation_voting），"
            "统计按阶段（phase）和完整 label 的匹配情况。"
        )
    )
    parser.add_argument(
        "-H",
        "--human-dir",
        type=str,
        default=os.path.join("data", "input", "human_annotation"),
        help=(
            "人工标注 JSON 所在目录，可以是绝对路径，"
            "或相对于项目根目录/当前脚本上级目录的相对路径。"
        ),
    )
    parser.add_argument(
        "-L",
        "--llm-dir",
        type=str,
        default=os.path.join("data", "output", "llm_annotation_voting", "20251205"),
        help=(
            "大模型投票结果 JSON 所在目录，可以是绝对路径，"
            "或相对于项目根目录/当前脚本上级目录的相对路径。"
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=os.path.join("data", "output", "evalresult"),
        help=(
            "输出目录，可以是绝对路径，或相对于项目根目录/当前脚本上级目录的相对路径。"
            "默认写入 evalresult 目录下。"
        ),
    )
    args = parser.parse_args()

    # 统一从当前脚本上级目录视作“项目根目录”
    project_root = os.path.join(os.path.dirname(__file__), "..")

    # 解析人工标注目录
    if os.path.isabs(args.human_dir):
        human_base = args.human_dir
    else:
        human_base = os.path.join(project_root, args.human_dir)

    # 解析大模型标注目录
    if os.path.isabs(args.llm_dir):
        llm_base = args.llm_dir
    else:
        llm_base = os.path.join(project_root, args.llm_dir)

    # 解析输出目录（默认 evalresult）
    if os.path.isabs(args.output_dir):
        out_dir = args.output_dir
    else:
        out_dir = os.path.join(project_root, args.output_dir)

    os.makedirs(out_dir, exist_ok=True)

    file_pairs = [  
        ("annotation_1_gpt4omini_fixed.json", "merged_1_gpt4omini_voting_3rounds_gpt_5.json"),
        ("annotation_2_gpt4omini_fixed.json", "merged_2_gpt4omini_voting_3rounds_gpt_5.json"),
        ("annotation_3_gpt4omini_fixed.json", "merged_3_gpt4omini_voting_3rounds_gpt_5.json"),
        ("annotation_4_gpt4omini_fixed.json", "merged_4_gpt4omini_voting_3rounds_gpt_5.json"),
        ("annotation_5_gpt4omini_fixed.json", "merged_5_gpt4omini_voting_3rounds_gpt_5.json"),
    ]
    # 阶段（phase）匹配统计
    merged_stats_phase = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: {"total": 0, "matched": 0}))
    )
    # 完整 label 精确匹配统计
    merged_stats_label = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: {"total": 0, "matched": 0}))
    )
    # 人工（true phase） vs voting_final（predicted phase）的混淆矩阵
    merged_confusion = defaultdict(lambda: defaultdict(int))

    for human_file, llm_file in file_pairs:
        human_path = os.path.join(human_base, human_file)
        llm_path = os.path.join(llm_base, llm_file)

        # phase 级别
        stats_phase = analyze_model_label_matching_strict(human_path, llm_path)
        for model in stats_phase:
            for cat in stats_phase[model]:
                for label in stats_phase[model][cat]:
                    merged_stats_phase[model][cat][label]["total"] += stats_phase[
                        model
                    ][cat][label]["total"]
                    merged_stats_phase[model][cat][label]["matched"] += stats_phase[
                        model
                    ][cat][label]["matched"]

        # 完整 label 级别
        stats_label = analyze_model_label_matching_exact(human_path, llm_path)
        for model in stats_label:
            for cat in stats_label[model]:
                for lbl in stats_label[model][cat]:
                    merged_stats_label[model][cat][lbl]["total"] += stats_label[
                        model
                    ][cat][lbl]["total"]
                    merged_stats_label[model][cat][lbl]["matched"] += stats_label[
                        model
                    ][cat][lbl]["matched"]

        # voting_final phase 混淆矩阵（人工为真实值）
        conf = collect_phase_confusion_voting_final(human_path, llm_path)
        for tp in conf:
            for pp in conf[tp]:
                merged_confusion[tp][pp] += conf[tp][pp]

    # phase 级别结果
    phase_out_path = os.path.join(out_dir, "model_phase.txt")
    with open(phase_out_path, "w", encoding="utf-8") as f:
        print_model_matching_results(merged_stats_phase, file=f)
    print_model_matching_results(merged_stats_phase)

    # 完整 label 精确匹配结果
    label_out_path = os.path.join(out_dir, "model_label_exact.txt")
    with open(label_out_path, "w", encoding="utf-8") as f:
        print_model_label_matching_results(merged_stats_label, file=f)
    print_model_label_matching_results(merged_stats_label)

    print(f"Phase-level results saved to: {phase_out_path}")
    print(f"Exact-label results saved to: {label_out_path}")

    # 写出 phase 混淆矩阵（人工 vs voting_final）
    conf_out_path = os.path.join(out_dir, "human_vs_voting_final_phase_confusion.txt")
    write_phase_confusion_matrix(merged_confusion, conf_out_path)
    print(f"Human vs voting_final phase confusion matrix saved to: {conf_out_path}")

if __name__ == "__main__":
    main()