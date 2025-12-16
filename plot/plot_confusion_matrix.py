import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Global plotting style (consistent with other plots)
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


PHASES = ["Phase 1", "Phase 2", "Phase 3", "Phase 4", "EMPTY"]


def load_confusion_matrix(path: Path):
    """
    解析 human_vs_voting_final_phase_confusion.txt 中的混淆矩阵。

    返回:
        true_labels: list[str]  # 行标签顺序
        pred_labels: list[str]  # 列标签顺序
        perc: np.ndarray (n_true, n_pred)  # 百分比
        counts: np.ndarray (n_true, n_pred)  # 计数
    """
    lines = path.read_text(encoding="utf-8").splitlines()

    header_line = None
    for line in lines:
        if line.startswith("| True phase"):
            header_line = line
            break
    if header_line is None:
        raise ValueError("未在文件中找到表头行 '| True phase | ... |'")

    header_cells = [c.strip() for c in header_line.strip().strip("|").split("|")]
    # header_cells: ['True phase', 'Phase 1', 'Phase 2', 'Phase 3', 'Phase 4', 'EMPTY']
    pred_labels = header_cells[1:]

    true_labels = []
    perc_rows = []
    count_rows = []

    row_pattern = re.compile(r"\|\s*Phase|\|\s*EMPTY")

    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        if not row_pattern.match(stripped):
            continue
        # 去掉首尾竖线
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if not cells:
            continue

        true_label = cells[0]
        if true_label == "True phase":
            # 已在上面解析过表头
            continue

        # 每个单元格形如 "96.04% (364)"
        row_perc = []
        row_counts = []
        for cell in cells[1:]:
            # 提取百分比和数字
            m = re.search(r"([\d.]+)%\s*\((\d+)\)", cell)
            if not m:
                row_perc.append(0.0)
                row_counts.append(0)
            else:
                row_perc.append(float(m.group(1)))
                row_counts.append(int(m.group(2)))

        true_labels.append(true_label)
        perc_rows.append(row_perc)
        count_rows.append(row_counts)

    perc = np.array(perc_rows, dtype=float)
    counts = np.array(count_rows, dtype=int)
    return true_labels, pred_labels, perc, counts


def plot_confusion_heatmap(true_labels, pred_labels, perc, counts, out_path: Path):
    """
    绘制带数值标注的混淆矩阵热力图。
    """
    n_true, n_pred = perc.shape

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

    # 使用百分比作为颜色强度（0-100）
    im = ax.imshow(perc, cmap="Blues", vmin=0, vmax=100)

    # 坐标轴标签
    ax.set_xticks(np.arange(n_pred))
    ax.set_yticks(np.arange(n_true))
    ax.set_xticklabels(pred_labels)
    ax.set_yticklabels(true_labels)

    ax.set_xlabel("Llm predicted phase")
    ax.set_ylabel("Human true phase")
    ax.set_title("Human-LLM(Voting Ensemble) Confusion Matrix")

    # 不需要灰色背景网格，只保留色块本身
    ax.grid(False)

    # 横轴类别水平放置
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")

    # 在每个单元格中标注 "xx.xx%\n(n)"
    for i in range(n_true):
        for j in range(n_pred):
            text = f"{perc[i, j]:.1f}%\n({counts[i, j]})"
            # 根据背景颜色深浅选择字体颜色
            color = "white" if perc[i, j] > 50 else "black"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Row-wise accuracy (%)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Confusion matrix heatmap saved to: {out_path}")


def main():
    # 本脚本位于 MemEval/plot/，项目根目录为其上一级
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent

    conf_path = project_root / "data" / "output" / "evalresult" / "human_vs_voting_final_phase_confusion.txt"
    if not conf_path.exists():
        raise FileNotFoundError(f"未找到混淆矩阵文件: {conf_path}")

    out_dir = project_root / "data" / "output" / "plot_result"
    out_dir.mkdir(parents=True, exist_ok=True)

    true_labels, pred_labels, perc, counts = load_confusion_matrix(conf_path)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"human_vs_voting_final_phase_confusion_{timestamp}.png"
    plot_confusion_heatmap(true_labels, pred_labels, perc, counts, out_path)


if __name__ == "__main__":
    main()



