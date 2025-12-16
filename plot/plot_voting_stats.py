import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


# Global plotting style (keep consistent with human_annotation_stats_plot.py)
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


def load_llm_voting_stats(path: Path, model_name: str = "voting_final"):
    """
    从 llm_annotation_voting_stats*.txt 中解析指定 Model 的 ASCII 表格（例如 voting_final）。

    返回:
        categories: ['1','2','3','4']
        label_names: ['1.1', '1.2', ..., '4.3', ...]
        counts: dict[str, list[int]]  # 每个类别对应一个 label 计数列表
    """
    lines = path.read_text(encoding="utf-8").splitlines()

    # 1) 找到目标 Model 段落的起始行
    model_line = f"Model: {model_name}"
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip() == model_line:
            start_idx = i
            break
    if start_idx is None:
        raise ValueError(f"在文件中未找到指定模型段落: {model_line}")

    # 2) 在该段落内找到表头行（包含 Cat 和 总样本数）
    header_idx = None
    header_line = None
    for j in range(start_idx + 1, len(lines)):
        line = lines[j]
        # 如果遇到下一个模型段落，说明当前模型没有表格
        if line.strip().startswith("Model:") and j > start_idx + 1:
            break
        if line.startswith("|") and "Cat" in line and "总样本数" in line:
            header_idx = j
            header_line = line
            break

    if header_line is None:
        raise ValueError(f"未在模型 {model_name} 段落中找到表头行。")

    header_parts = [p.strip() for p in header_line.split("|")[1:-1]]
    if header_parts[0] != "Cat":
        raise ValueError("解析表头失败，首列不是 'Cat'。")

    # label 列为从 'Cat' 后开始，到 '总样本数' 之前的所有列
    try:
        total_idx = header_parts.index("总样本数")
    except ValueError as e:
        raise ValueError("表头中未找到 '总样本数' 列。") from e

    label_names = header_parts[1:total_idx]

    counts: dict[str, list[int]] = {}
    categories: list[str] = []

    # 3) 解析该模型的表格数据，直到遇到下一个 Model 段或文件结束
    for k in range(header_idx + 1, len(lines)):
        line = lines[k]
        stripped = line.strip()
        if stripped.startswith("Model:") and k > header_idx + 1:
            # 下一个模型的表格开始，当前模型结束
            break

        if not line.startswith("|"):
            continue
        if "| 总计 |" in line:
            # 跳过总计行
            continue

        parts = [p.strip() for p in line.split("|")[1:-1]]
        if not parts:
            continue

        cat = parts[0]
        if cat not in {"1", "2", "3", "4"}:
            continue

        label_values_str = parts[1 : 1 + len(label_names)]
        label_values = [int(v) for v in label_values_str]

        counts[cat] = label_values
        categories.append(cat)

    if not counts:
        raise ValueError(f"在模型 {model_name} 段落中未解析到任何有效数据行。")

    categories = sorted(categories, key=lambda x: int(x))
    return categories, label_names, counts


def plot_line_chart(
    categories, label_names, counts, out_dir: Path, filename: str
):
    """绘制不同问题类别的 label 数量分布折线图。"""
    x = np.arange(len(label_names))

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    color_cycle = plt.get_cmap("tab10").colors
    markers = ["o", "s", "^", "D"]
    linestyles = ["-", "--", "-.", ":"]

    for i, cat in enumerate(categories):
        y = counts[cat]
        ax.plot(
            x,
            y,
            marker=markers[i % len(markers)],
            color=color_cycle[i % len(color_cycle)],
            linestyle=linestyles[i % len(linestyles)],
            linewidth=1.2,
            markersize=5,
            label=CAT_NAME.get(cat, f"Category {cat}"),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(label_names, rotation=0)
    ax.set_xlabel("Label type")
    ax.set_ylabel("Number of samples")
    ax.legend(frameon=False, ncol=2, prop={"weight": "bold"})
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)

    fig.tight_layout()
    save_path = out_dir / filename
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Line chart saved to: {save_path}")


def plot_pie_charts(
    categories, label_names, counts, out_dir: Path, filename: str
):
    """在一张图中绘制 4 个类别的 label 分布饼图（2x2）。"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=300)
    axes = axes.flatten()

    # 统一配色，使四个饼图颜色一致
    cmap = plt.get_cmap("tab20")
    colors = cmap(np.linspace(0, 1, len(label_names)))

    for ax, cat in zip(axes, categories):
        data = np.array(counts[cat], dtype=float)
        total = data.sum()
        if total == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
            ax.set_axis_off()
            continue
        wedges, texts, autotexts = ax.pie(
            data,
            labels=None,  # 不在扇区外侧标出 label 名称，仅显示百分比
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            textprops={"fontsize": 10, "fontweight": "bold"},
            pctdistance=0.8,
        )
        ax.set_title(CAT_NAME.get(cat, f"Category {cat}"), fontsize=14)

        for w in wedges:
            w.set_linewidth(0.5)
            w.set_edgecolor("white")

        # 只保留每个类别中前三大的百分比标注
        top3_idx = np.argsort(data)[-3:]
        for idx, autotext in enumerate(autotexts):
            if idx not in top3_idx:
                autotext.set_text("")
            else:
                autotext.set_fontsize(12)
                autotext.set_fontweight("bold")

    # 如果类别不足 4，剩余子图隐藏
    for j in range(len(categories), 4):
        axes[j].set_axis_off()

    # 为饼图添加统一图例（颜色对应 label 类型）
    legend_patches = [
        Patch(facecolor=colors[i], edgecolor="white", label=label_names[i])
        for i in range(len(label_names))
    ]
    fig.legend(
        handles=legend_patches,
        loc="center right",
        ncol=1,
        frameon=False,
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        prop={"weight": "bold"},
    )

    fig.tight_layout(rect=[0.0, 0.0, 0.85, 1.0])

    save_path = out_dir / filename
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Pie charts saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "根据 data/output/evalresult/llm_annotation_voting_stats*.txt 中指定模型 (默认 voting_final) 的表格，"
            "绘制 label 分布的折线图和饼图。"
        )
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="voting_final",
        help="要绘图的模型名称，对应文本中的 'Model: <name>'，默认为 'voting_final'",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        help="指定输入统计文件的路径。如不指定，则默认查找 data/output/evalresult/llm_annotation_voting_statsl.txt 或 llm_annotation_voting_stats.txt",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="指定输出图片的目录路径。如不指定，则默认输出到 data/output/plot_result/ 目录",
    )
    args = parser.parse_args()

    # 本脚本位于 MemEval/plot/，项目根目录为其上一级
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent

    # 如果用户指定了输入文件，则使用指定的路径
    if args.input:
        stats_path = Path(args.input)
        if not stats_path.exists():
            raise FileNotFoundError(f"指定的统计文件不存在: {stats_path}")
    else:
        # 默认优先使用 llm_annotation_voting_statsl.txt，如不存在再退回到原始文件名
        stats_path_l = project_root / "data" / "output" / "evalresult" / "llm_annotation_voting_statsl.txt"
        stats_path = stats_path_l
        if not stats_path.exists():
            stats_path_alt = project_root / "data" / "output" / "evalresult" / "llm_annotation_voting_stats.txt"
            if stats_path_alt.exists():
                stats_path = stats_path_alt
            else:
                raise FileNotFoundError(
                    f"统计文件不存在: {stats_path_l} 或 {stats_path_alt}"
                )

    # 如果用户指定了输出目录，则使用指定的路径
    if args.output:
        out_dir = Path(args.output)
    else:
        out_dir = project_root / "data" / "output" / "plot_result"
    out_dir.mkdir(parents=True, exist_ok=True)

    categories, label_names, counts = load_llm_voting_stats(
        stats_path, model_name=args.model
    )

    # 根据模型名区分输出文件
    safe_model = args.model.replace(" ", "_")
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    line_name = f"llm_voting_{safe_model}_line_{timestamp}.png"
    pies_name = f"llm_voting_{safe_model}_pies_{timestamp}.png"

    plot_line_chart(categories, label_names, counts, out_dir=out_dir, filename=line_name)
    plot_pie_charts(categories, label_names, counts, out_dir=out_dir, filename=pies_name)


if __name__ == "__main__":
    main()







