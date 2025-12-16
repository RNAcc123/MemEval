import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# 保持与其他图一致的全局风格（Times New Roman、粗+斜体）
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.sans-serif": ["Times New Roman"],
        "axes.unicode_minus": False,
        "font.style": "italic",
        "font.weight": "bold",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 13,
        "axes.labelweight": "bold",
        "legend.fontsize": 12,
    }
)


CAT_NAME = {
    "1": "Multi-hop",
    "2": "Temporal",
    "3": "Open-domain",
    "4": "Single-hop",
}

def load_label_level_category_rates(path: Path, model_name: str = "voting_final"):
    """
    从 model_label_exact.txt 中解析指定模型的「Category X（含 EMPTY）label 精确匹配率」。

    返回:
        categories: ['1','2','3','4']
        rates: [float, ...]  # 百分比数值
    """
    text = path.read_text(encoding="utf-8").splitlines()

    # 找到 "[Label 精确匹配] 模型: xxx" 段落
    start_idx = None
    header = f"[Label 精确匹配] 模型: {model_name}"
    for i, line in enumerate(text):
        if line.strip() == header:
            start_idx = i
            break
    if start_idx is None:
        raise ValueError(f"在 {path} 中未找到段落: {header}")

    categories = []
    rates = []

    # 使用正则匹配: Category X（含 EMPTY）label 精确匹配率: Y%
    pattern = re.compile(
        r"Category\s+(\d)（含 EMPTY）label 精确匹配率:\s*([\d.]+)%"
    )

    for line in text[start_idx + 1 :]:
        line = line.strip()
        if not line:
            # 该段落结束
            break
        m = pattern.search(line)
        if m:
            cat = m.group(1)
            rate = float(m.group(2))
            categories.append(cat)
            rates.append(rate)

    if not categories:
        raise ValueError(
            f"在 {path} 的 {header} 段落中未解析到任何 Category 一致率行。"
        )

    # 按类别编号排序
    paired = sorted(zip(categories, rates), key=lambda x: int(x[0]))
    categories_sorted = [p[0] for p in paired]
    rates_sorted = [p[1] for p in paired]
    return categories_sorted, rates_sorted


def load_phase_level_category_rates(path: Path, model_name: str = "voting_final"):
    """
    从 model_phase.txt 中解析指定模型的各 Category 的 Overall 列（phase-level 一致率）。

    返回:
        categories: ['1','2','3','4']
        rates: [float, ...]  # 百分比数值
    """
    lines = path.read_text(encoding="utf-8").splitlines()

    # 找到 "模型: xxx" 段落
    start_idx = None
    header = f"模型: {model_name}"
    for i, line in enumerate(lines):
        if line.strip() == header:
            start_idx = i
            break
    if start_idx is None:
        raise ValueError(f"在 {path} 中未找到段落: {header}")

    categories = []
    rates = []

    # 在该模型段落内部，表头之后的数据行如：
    # |  1   |   97.4   | ... |  72.34   |
    in_table = False
    for line in lines[start_idx + 1 :]:
        stripped = line.strip()
        if not stripped:
            # 段落结束
            break
        if stripped.startswith("总样本数:"):
            # 统计信息之后的不再解析
            break
        if stripped.startswith("| Cat"):
            in_table = True
            continue
        if not in_table:
            continue
        if not stripped.startswith("|"):
            continue
        if stripped.startswith("|---") or stripped.startswith("+"):
            # 分隔线
            continue

        # 去掉首尾的竖线，然后按 '|' 分列
        parts = [p.strip() for p in stripped.strip("|").split("|")]
        if not parts or parts[0] == "Cat":
            continue

        cat = parts[0]
        # Overall 列是最后一列
        try:
            overall_str = parts[-1]
            rate = float(overall_str)
        except ValueError as e:
            raise ValueError(f"解析 Overall 列失败: 行内容={stripped}") from e

        categories.append(cat)
        rates.append(rate)

    if not categories:
        raise ValueError(
            f"在 {path} 的 {header} 段落中未解析到任何 Category Overall 行。"
        )

    paired = sorted(zip(categories, rates), key=lambda x: int(x[0]))
    categories_sorted = [p[0] for p in paired]
    rates_sorted = [p[1] for p in paired]
    return categories_sorted, rates_sorted


def plot_bar_chart(
    xlabels,
    rates,
    title: str,
    ylabel: str,
    out_path: Path,
):
    """绘制单组数值的直方图。"""
    x = np.arange(len(xlabels))
    width = 0.6

    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
    bars = ax.bar(x, rates, width=width, color="#4477AA", edgecolor="black")

    ax.set_xticks(x)
    # 横轴标签水平（不再倾斜），按给定顺序正向排列
    ax.set_xticklabels(xlabels, rotation=0, ha="center")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, max(rates) * 1.15)

    # 在柱子上方标数值
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(rates) * 0.02,
            f"{rate:.2f}%",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Bar chart saved to: {out_path}")


def main():
    # 本脚本位于 MemEval/plot/，项目根目录为其上一级
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent

    label_file = project_root / "data" / "output" / "evalresult" / "model_label_exact.txt"
    phase_file = project_root / "data" / "output" / "evalresult" / "model_phase.txt"

    if not label_file.exists():
        raise FileNotFoundError(f"未找到文件: {label_file}")
    if not phase_file.exists():
        raise FileNotFoundError(f"未找到文件: {phase_file}")

    out_dir = project_root / "data" / "output" / "plot_result"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 使用 voting_final 的结果（按类别）
    categories_l, rates_l = load_label_level_category_rates(
        label_file, model_name="voting_final"
    )
    categories_p, rates_p = load_phase_level_category_rates(
        phase_file, model_name="voting_final"
    )

    # 将 Cat 编号映射为更直观的类别名称
    xlabels_l = [CAT_NAME.get(c, f"Cat {c}") for c in categories_l]
    xlabels_p = [CAT_NAME.get(c, f"Cat {c}") for c in categories_p]

    # 为 label-level 绘制「按类别」直方图
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_label = out_dir / f"model_label_exact_voting_final_bar_{timestamp}.png"
    plot_bar_chart(
        xlabels_l,
        rates_l,
        title="Label-level exact match by category",
        ylabel="Exact match accuracy (%)",
        out_path=out_label,
    )

    # 为 phase-level 绘制「按类别」直方图
    out_phase = out_dir / f"model_phase_voting_final_bar_{timestamp}.png"
    plot_bar_chart(
        xlabels_p,
        rates_p,
        title="Phase-level match by category",
        ylabel="Match accuracy (%)",
        out_path=out_phase,
    )


if __name__ == "__main__":
    main()



