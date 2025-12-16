#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计 `human_annotation` 目录下人工标注的整体情况。

功能：
- 读取 `../human_annotation/annotation_*_gpt4omini_fixed.json`
- 只统计 qa_category ∈ {1,2,3,4}（排除 5）
- 对每个 qa_category 统计：
  - 每种 label 类型（1.1, 1.2, ..., 4.3, 5）的计数
  - 总样本数
  - 有 label 的样本数
  - label 总数量（多标签累加）
  - 无 label 样本数
  - 正确率 = 无 label 样本数 / 总样本数 * 100
- 将结果输出为 ASCII 表格，并附 Overall label counts
- 结果保存为：`evalresult/human_annotation_stats_full.txt`
"""

import json
import os
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Set


BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "input", "human_annotation")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "output", "evalresult")


def find_annotation_files(dirpath: str) -> List[str]:
    """查找 human_annotation 目录下的 human_dataset_part*.json 文件。"""
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
    统计每个 qa_category 的 label 分布及整体数量信息。

    返回：
        stats: 按 category 聚合的统计信息
        label_list: 所有出现过的 label（排序后）
    """
    # stats[cat] 结构：
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
        # human_annotation 的结构是一个 dict: key -> item
        if isinstance(data, dict):
            it = data.items()
        else:
            # 兜底：如果将来变成 list，则逐个遍历
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
    """按照既定格式生成 ASCII 表格，并保存到 out_path。"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    lines: List[str] = []
    lines.append("人工标注 label 统计（按 qa_category，排除5，含各 label 计数）")
    lines.append("说明：正确率 = (无 label 的样本数 / 总样本数) * 100")
    lines.append("")

    header = ["Cat"] + label_list + [
        "总样本数",
        "有label样本数",
        "label总数量",
        "无label样本数",
        "正确率(%)",
    ]

    # 构建每一行（各 category 的行）
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

    # 计算列宽
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


