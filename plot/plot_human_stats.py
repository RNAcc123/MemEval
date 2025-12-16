import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from pathlib import Path


# Global plotting style (paper-like):
# Use Times New Roman for all visible text (English-only to avoid missing glyphs)
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

# Mapping from category id to descriptive name
CAT_NAME = {
    "1": "Multi-hop",
    "2": "Temporal",
    "3": "Open-domain",
    "4": "Single-hop",
}


def load_human_annotation_stats(path: Path):
    """
    解析 human_annotation_stats.txt 中的 ASCII 表格。

    返回:
        categories: ['1','2','3','4']
        label_names: ['1.1', '1.2', ..., '4.3']
        counts: dict[str, list[int]]  # 每个类别对应一个 label 计数列表
    """
    lines = path.read_text(encoding="utf-8").splitlines()

    # 找到表头行（包含 Cat 和 总样本数）
    header_line = None
    for line in lines:
        if line.startswith("|") and "Cat" in line and "总样本数" in line:
            header_line = line
            break
    if header_line is None:
        raise ValueError("未在文件中找到表头行（包含 'Cat' 和 '总样本数'）。")

    header_parts = [p.strip() for p in header_line.split("|")[1:-1]]
    # header_parts 结构: ['Cat', '1.1', ..., '4.3', '总样本数', '有label样本数', 'label总数量', '无label样本数', '正确率(%)']
    if header_parts[0] != "Cat":
        raise ValueError("解析表头失败，首列不是 'Cat'。")

    # 根据给定文件，label 列为 1.1 ~ 4.3，共 10 列
    label_names = header_parts[1:11]

    counts: dict[str, list[int]] = {}
    categories: list[str] = []

    for line in lines:
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

        # 取该行对应的 label 计数
        label_values_str = parts[1 : 1 + len(label_names)]
        label_values = [int(v) for v in label_values_str]

        counts[cat] = label_values
        categories.append(cat)

    # 按类别排序（1,2,3,4）
    categories = sorted(categories, key=lambda x: int(x))
    return categories, label_names, counts


def plot_line_chart(
    categories, label_names, counts, out_dir: Path, filename: str = "human_label_line.png"
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
    categories, label_names, counts, out_dir: Path, filename: str = "human_label_pies.png"
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
    # 本脚本位于 MemEval/plot/，项目根目录为其上一级
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent

    stats_path = project_root / "data" / "output" / "evalresult" / "human_annotation_stats.txt"
    out_dir = project_root / "data" / "output" / "plot_result"
    out_dir.mkdir(parents=True, exist_ok=True)

    categories, label_names, counts = load_human_annotation_stats(stats_path)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    plot_line_chart(categories, label_names, counts, out_dir=out_dir, filename=f"human_label_line_{timestamp}.png")
    plot_pie_charts(categories, label_names, counts, out_dir=out_dir, filename=f"human_label_pies_{timestamp}.png")

    # 如需交互显示，可取消注释
    # plt.show()


if __name__ == "__main__":
    main()


