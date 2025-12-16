#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计 `llm_annotation_voting` 目录下大模型标注的整体情况（基于 merged_*_voting_3rounds_gpt_5.json）。

功能：
- 默认读取 `../llm_annotation_voting` 下的 `merged_*_gpt4omini_voting_3rounds_gpt_5.json`
- 也可以通过命令行参数 `--input-dir` 选择只统计某个子文件夹中的 merged 文件
- 只统计 qa_category ∈ {1,2,3,4}（排除 5）
- 对「最终 voting 结果」和「每个 used_model 的单独结果」分别输出表格：
  - 每种 label 类型（1.1, 1.2, ..., 4.3, 5, …）的计数
  - 总样本数
  - 有 label 的样本数
  - label 总数量（多标签累加；最终 voting 只有 0/1 个）
  - 无 label 样本数
  - 正确率 = 无 label 样本数 / 总样本数 * 100
- 结果保存为：`evalresult/llm_annotation_voting_stats_full.txt`
"""

import argparse
import json
import os
from datetime import datetime
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple


# 默认输出目录与文件名（可通过命令行参数覆盖目录）
DEFAULT_OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "output", "evalresult")


def find_merged_files(dirpath: str) -> List[str]:
    """
    查找给定目录下需要处理的 JSON 文件。
    策略：选择该目录下所有文件名包含 'voting_annotation_' 的 .json 文件。
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
    """初始化按 category 统计的结构。"""
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
    统计 voting 最终结果 & 各 used_model 结果的 label 分布及数量信息。

    返回：
        final_stats: 按 category 聚合的 voting 最终结果
        model_stats: model -> 按 category 聚合的结果
        label_list: 所有出现过的 label（排序后）
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
            # 1) voting 最终结果统计
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
            # 2) 各 used_model 统计
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
    """为一个（voting_final 或某个 used_model）生成表格文本行。"""
    lines: List[str] = []
    lines.append(f"Model: {title}")

    header = ["Cat"] + label_list + [
        "总样本数",
        "有label样本数",
        "label总数量",
        "无label样本数",
        "正确率(%)",
    ]

    # 构建行（各 category 的行）
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

    # 计算总体行（汇总 1-4 类的结果），作为表格最后一行
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

    overall_row: List[str] = ["总计"]
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

    # 计算列宽（包含总体行）
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
    """生成最终文本并写入文件。"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    lines: List[str] = []
    lines.append(
        "Statistics for llm_annotation_voting (merged_*_gpt4omini_voting_3rounds_gpt_5.json)"
    )
    lines.append("说明：正确率 = (无 label 的样本数 / 总样本数) * 100")
    lines.append("")

    # 先写 voting 最终结果
    lines.extend(_format_single_table("voting_final", final_stats, label_list))

    # 再写各个 used_model 的表格
    for model_name in sorted(model_stats.keys()):
        lines.extend(_format_single_table(model_name, model_stats[model_name], label_list))

    content = "\n".join(lines)
    print(content)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "统计 llm_annotation_voting 中的大模型标注结果（基于 merged_*_gpt4omini_voting_3rounds_gpt_5.json）。\n"
            "默认从 ../llm_annotation_voting 读取，也可通过 --input-dir 指定子目录。"
        )
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        default=os.path.join("data", "output", "llm_annotation_voting"),
        help=(
            "输入目录，可以是绝对路径，或相对于项目根目录/当前脚本上级目录的相对路径。"
            "目录下将按模式 merged_*_gpt4omini_voting_3rounds_gpt_5.json 搜索文件。"
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="evalresult",
        help=(
            "输出目录，可以是绝对路径，或相对于项目根目录/当前脚本上级目录的相对路径。"
            "文件名固定为 llm_annotation_voting_stats_full.txt。"
        ),
    )
    args = parser.parse_args()

    # 解析输入目录：支持绝对路径，或相对于当前脚本上级目录的相对路径
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

    # 解析输出目录：支持绝对路径，或相对于当前脚本上级目录的相对路径
    if os.path.isabs(args.output_dir):
        out_dir = args.output_dir
    else:
        out_dir = os.path.join(os.path.dirname(__file__), "..", args.output_dir)

    # 如果用户没改默认值，则保持与旧版本一致的默认目录
    if args.output_dir == "evalresult":
        out_dir = DEFAULT_OUT_DIR

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"llm_annotation_voting_stats_{timestamp}.txt")

    format_and_save(final_stats, model_stats, label_list, out_path)
    print("Saved results to:", out_path)


if __name__ == "__main__":
    main()


