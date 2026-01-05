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


def get_stage_colors(label_names):
    """
    根据label名称的阶段前缀，分配4种主色调系统的颜色。
    Stage 1 (Extraction): 蓝色系 (深 -> 浅)
    Stage 2 (Update): 橙色/黄色系 (深 -> 浅)
    Stage 3 (Retrieval): 灰色/紫色系 (深 -> 浅)
    Stage 4 (Utilization): 绿色/青色系 (深 -> 浅)
    """
    # 定义每个阶段的颜色渐变（从深到浅）
    stage_color_maps = {
        "1": [  # Extraction - 蓝色系
            (0.12, 0.47, 0.71),  # 深蓝
            (0.26, 0.63, 0.85),  # 中蓝
            (0.50, 0.78, 0.95),  # 浅蓝
        ],
        "2": [  # Update - 橙色/黄色系
            (0.90, 0.50, 0.13),  # 深橙
            (0.98, 0.65, 0.30),  # 中橙
            (1.00, 0.80, 0.40),  # 浅橙/黄
        ],
        "3": [  # Retrieval - 灰色/紫色系
            (0.50, 0.40, 0.60),  # 深紫灰
            (0.65, 0.55, 0.75),  # 中紫灰
            (0.80, 0.75, 0.88),  # 浅紫灰
        ],
        "4": [  # Utilization - 绿色/青色系
            (0.13, 0.59, 0.53),  # 深青绿
            (0.30, 0.75, 0.68),  # 中青绿
            (0.55, 0.88, 0.82),  # 浅青绿
        ],
    }
    
    colors = []
    for label in label_names:
        # 提取阶段编号（label格式如 "1.1", "2.3" 等）
        stage = label.split(".")[0]
        sub_idx = int(label.split(".")[1]) - 1  # 子项索引（从0开始）
        
        if stage in stage_color_maps:
            color_list = stage_color_maps[stage]
            # 根据子项索引选择颜色（循环使用）
            color = color_list[sub_idx % len(color_list)]
            colors.append(color)
        else:
            # 默认灰色
            colors.append((0.7, 0.7, 0.7))
    
    return colors


def plot_pie_charts(
    categories, label_names, counts, out_dir: Path, filename: str
):
    """在一张图中绘制 4 个类别的 label 分布饼图（2x2）。"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=300)
    axes = axes.flatten()

    # 使用基于阶段的配色方案
    colors = get_stage_colors(label_names)

    for ax, cat in zip(axes, categories):
        data = np.array(counts[cat], dtype=float)
        total = data.sum()
        if total == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
            ax.set_axis_off()
            continue
        
        # 过滤掉数值为0的标签
        filtered_labels = [label_names[i] if data[i] > 0 else '' for i in range(len(label_names))]
        
        # 绘制饼图，使用引线连接标签
        wedges, texts, autotexts = ax.pie(
            data,
            labels=filtered_labels,  # 只显示非0的label
            autopct=lambda pct: f'{pct:.1f}%' if pct > 0 else '',
            startangle=90,
            colors=colors,
            textprops={"fontsize": 9, "fontweight": "bold"},
            pctdistance=0.75,
            labeldistance=1.15,  # label距离圆心的距离
            wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
            rotatelabels=False,
        )
        ax.set_title(CAT_NAME.get(cat, f"Category {cat}"), fontsize=18, fontweight="bold")
        
        # 只处理非空标签
        valid_indices = [i for i, label in enumerate(filtered_labels) if label != '']
        valid_texts = [texts[i] for i in valid_indices]
        valid_wedges = [wedges[i] for i in valid_indices]
        
        if not valid_texts:
            continue
        
        # 调整标签位置以避免重叠 - 改进版
        positions = []
        angles = []
        wedge_sizes = []
        
        for i, (text, wedge) in enumerate(zip(valid_texts, valid_wedges)):
            x, y = text.get_position()
            angle = (wedge.theta2 + wedge.theta1) / 2.0
            size = wedge.theta2 - wedge.theta1
            positions.append([x, y])
            angles.append(angle)
            wedge_sizes.append(size)
        
        # 按角度排序，相邻的标签交替设置不同的距离
        angle_order = np.argsort(angles)
        base_distances = [1.15] * len(angles)
        
        for idx, i in enumerate(angle_order):
            # 检查与前一个和后一个的角度差
            if idx > 0:
                prev_i = angle_order[idx - 1]
                angle_diff = abs(angles[i] - angles[prev_i])
                if angle_diff < 45:
                    # 交替设置距离
                    if idx % 2 == 0:
                        base_distances[i] = 1.25
                    else:
                        base_distances[prev_i] = 1.25
        
        # 应用基础距离
        for i, (angle, dist) in enumerate(zip(angles, base_distances)):
            angle_rad = np.deg2rad(angle)
            positions[i][0] = np.cos(angle_rad) * dist
            positions[i][1] = np.sin(angle_rad) * dist
        
        # 检测并调整重叠的标签 - 优先沿切向调整以避免引线交叉
        adjusted = True
        max_iterations = 60
        iteration = 0
        min_distance = 0.18
        max_radius = 1.3  # 限制标签最远距离
        
        while adjusted and iteration < max_iterations:
            adjusted = False
            iteration += 1
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    dx = positions[i][0] - positions[j][0]
                    dy = positions[i][1] - positions[j][1]
                    distance = np.sqrt(dx**2 + dy**2)
                    
                    # 如果两个标签距离太近
                    if distance < min_distance:
                        adjusted = True
                        
                        # 获取两个标签的角度
                        angle_i = np.arctan2(positions[i][1], positions[i][0])
                        angle_j = np.arctan2(positions[j][1], positions[j][0])
                        
                        # 优先沿切向（垂直于径向）推开，减少引线交叉
                        # 切向方向：垂直于从原点到标签的方向
                        tangent_i = np.array([-np.sin(angle_i), np.cos(angle_i)])
                        tangent_j = np.array([-np.sin(angle_j), np.cos(angle_j)])
                        
                        # 判断应该向哪个切向方向推
                        if angle_i < angle_j:
                            push_i = -tangent_i * 0.06  # 逆时针推
                            push_j = tangent_j * 0.06   # 顺时针推
                        else:
                            push_i = tangent_i * 0.06
                            push_j = -tangent_j * 0.06
                        
                        # 推开两个标签
                        new_pos_i = [positions[i][0] + push_i[0], positions[i][1] + push_i[1]]
                        new_pos_j = [positions[j][0] + push_j[0], positions[j][1] + push_j[1]]
                        
                        # 检查是否超出最大半径
                        radius_i = np.sqrt(new_pos_i[0]**2 + new_pos_i[1]**2)
                        radius_j = np.sqrt(new_pos_j[0]**2 + new_pos_j[1]**2)
                        
                        if radius_i <= max_radius:
                            positions[i] = new_pos_i
                        if radius_j <= max_radius:
                            positions[j] = new_pos_j
        
        # 应用调整后的位置
        for text, pos in zip(valid_texts, positions):
            text.set_position(pos)
        
        # 手动绘制引线：从扇形边缘到标签
        for wedge, text in zip(valid_wedges, valid_texts):
            # 获取扇形的中心角度
            angle = (wedge.theta2 + wedge.theta1) / 2.0
            # 计算扇形边缘的点（半径为1）
            x1 = np.cos(np.deg2rad(angle))
            y1 = np.sin(np.deg2rad(angle))
            # 获取标签位置
            x2, y2 = text.get_position()
            # 绘制引线
            ax.plot([x1, x2], [y1, y2], color='gray', linewidth=0.5, linestyle='-', alpha=0.6)

        # 只保留每个类别中前三大的百分比标注
        top3_idx = np.argsort(data)[-3:]
        for idx, autotext in enumerate(autotexts):
            if idx not in top3_idx:
                autotext.set_text("")
            else:
                autotext.set_fontsize(10)
                autotext.set_fontweight("bold")

    # 如果类别不足 4，剩余子图隐藏
    for j in range(len(categories), 4):
        axes[j].set_axis_off()

    # 不再添加统一图例
    fig.tight_layout()

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







