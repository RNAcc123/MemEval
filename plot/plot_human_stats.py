import argparse
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
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.titleweight": "bold",
        "axes.labelsize": 14,
        "axes.labelweight": "bold",
        "legend.fontsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
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
    Parse the ASCII table in human_annotation_stats.txt.

    Returns:
        categories: ['1','2','3','4']
        label_names: ['1.1', '1.2', ..., '4.3']
        counts: dict[str, list[int]]  # Label-count list for each category
    """
    lines = path.read_text(encoding="utf-8").splitlines()

    # Find the header row (contains Cat and total-sample column)
    header_line = None
    for line in lines:
        if line.startswith("|") and "Cat" in line and ("Total samples" in line or "\u603b\u6837\u672c\u6570" in line):
            header_line = line
            break
    if header_line is None:
        raise ValueError("Header row not found (must contain 'Cat' and total-sample column).")

    header_parts = [p.strip() for p in header_line.split("|")[1:-1]]
    # header_parts example:
    # ['Cat', '1.1', ..., '4.3', 'Total samples', 'Samples with label',
    #  'Total label count', 'Samples without label', 'Accuracy (%)']
    if header_parts[0] != "Cat":
        raise ValueError("Failed to parse header: first column is not 'Cat'.")

    # In this file format, label columns are 1.1 ~ 4.3 (11 columns)
    label_names = header_parts[1:12]

    counts: dict[str, list[int]] = {}
    categories: list[str] = []

    for line in lines:
        if not line.startswith("|"):
            continue
        if "| Overall |" in line or f"| {'\u603b\u8ba1'} |" in line:
            # Skip overall row
            continue

        parts = [p.strip() for p in line.split("|")[1:-1]]
        if not parts:
            continue

        cat = parts[0]
        if cat not in {"1", "2", "3", "4"}:
            continue

        # Read the label counts for this row
        label_values_str = parts[1 : 1 + len(label_names)]
        label_values = [int(v) for v in label_values_str]

        counts[cat] = label_values
        categories.append(cat)

    # Sort categories as 1,2,3,4
    categories = sorted(categories, key=lambda x: int(x))
    return categories, label_names, counts


def plot_line_chart(
    categories, label_names, counts, out_dir: Path, filename: str = "human_label_line.png"
):
    """Plot label-count distribution lines across categories."""
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


def plot_bar_chart(
    categories, label_names, counts, out_dir: Path, filename: str = "human_label_bar.png"
):
    """Plot stacked bars by label, grouped by question category."""
    x = np.arange(len(label_names))
    width = 0.72

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    cat_colors = {
        "1": "#4C72B0",
        "2": "#DD8452",
        "3": "#55A868",
        "4": "#C44E52",
    }
    hatches = ["", "//", "..", "xx"]

    bottom = np.zeros(len(label_names))
    for i, cat in enumerate(categories):
        y = np.array(counts[cat], dtype=float)
        ax.bar(
            x,
            y,
            width,
            bottom=bottom,
            label=CAT_NAME.get(cat, f"Category {cat}"),
            color=cat_colors.get(cat, "#999999"),
            edgecolor="white",
            linewidth=0.6,
            hatch=hatches[i % len(hatches)],
            alpha=0.88,
            zorder=3,
        )
        bottom += y

    for boundary in [2.5, 5.5, 8.5]:
        ax.axvline(x=boundary, color="#666666", linestyle="--", linewidth=1.2, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(label_names, rotation=0)
    ax.set_xlabel("Label Type")
    ax.set_ylabel("Number of Samples")
    ax.legend(
        frameon=True, ncol=4, loc="upper center",
        prop={"weight": "bold", "size": 10},
        fancybox=True, shadow=False,
        edgecolor="#cccccc", framealpha=0.9,
        bbox_to_anchor=(0.5, 1.12),
    )
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#aaaaaa")
    ax.spines["bottom"].set_color("#aaaaaa")

    y_max = float(np.max(bottom)) if len(bottom) else 0.0
    ax.set_ylim(0, y_max * 1.25)

    stage_regions = [(0, 2, "Stage 1"), (3, 5, "Stage 2"), (6, 8, "Stage 3"), (9, 10, "Stage 4")]
    for start, end, stage_name in stage_regions:
        center_x = (start + end) / 2
        ax.text(
            center_x, y_max * 1.08, stage_name,
            ha="center", va="bottom",
            fontsize=14, fontweight="bold", color="#444444",
            zorder=5,
        )

    fig.tight_layout()
    save_path = out_dir / filename
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Bar chart saved to: {save_path}")


def get_stage_colors(label_names):
    """
    Assign colors by stage prefix in label names.
    Stage 1 (Extraction): blue shades (dark -> light)
    Stage 2 (Update): orange/yellow shades (dark -> light)
    Stage 3 (Retrieval): gray/purple shades (dark -> light)
    Stage 4 (Utilization): green/cyan shades (dark -> light)
    """
    # Define stage color gradients (dark to light)
    stage_color_maps = {
        "1": [  # Extraction - blues
            (0.12, 0.47, 0.71),  # dark blue
            (0.26, 0.63, 0.85),  # medium blue
            (0.50, 0.78, 0.95),  # light blue
        ],
        "2": [  # Update - orange/yellow
            (0.90, 0.50, 0.13),  # dark orange
            (0.98, 0.65, 0.30),  # medium orange
            (1.00, 0.80, 0.40),  # light orange/yellow
        ],
        "3": [  # Retrieval - gray/purple
            (0.50, 0.40, 0.60),  # dark purple-gray
            (0.65, 0.55, 0.75),  # medium purple-gray
            (0.80, 0.75, 0.88),  # light purple-gray
        ],
        "4": [  # Utilization - green/cyan
            (0.13, 0.59, 0.53),  # dark teal-green
            (0.30, 0.75, 0.68),  # medium teal-green
            (0.55, 0.88, 0.82),  # light teal-green
        ],
    }
    
    colors = []
    for label in label_names:
        # Extract stage index (label format: "1.1", "2.3", etc.)
        stage = label.split(".")[0]
        sub_idx = int(label.split(".")[1]) - 1  # Sub-index (0-based)
        
        if stage in stage_color_maps:
            color_list = stage_color_maps[stage]
            # Pick color by sub-index (cycled)
            color = color_list[sub_idx % len(color_list)]
            colors.append(color)
        else:
            # Default gray
            colors.append((0.7, 0.7, 0.7))
    
    return colors


def plot_pie_charts(
    categories, label_names, counts, out_dir: Path, filename: str = "human_label_pies.png"
):
    """Plot a 2x2 grid of label-distribution pies for 4 categories."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), dpi=300)
    fig.subplots_adjust(wspace=0.05, hspace=0.35)
    axes = axes.flatten()

    # Use stage-based color palette
    colors = get_stage_colors(label_names)

    for ax, cat in zip(axes, categories):
        data = np.array(counts[cat], dtype=float)
        total = data.sum()
        if total == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
            ax.set_axis_off()
            continue
        
        # Filter out labels with zero values
        filtered_labels = [label_names[i] if data[i] > 0 else '' for i in range(len(label_names))]
        
        # Draw pie chart with connector lines
        wedges, texts, autotexts = ax.pie(
            data,
            labels=filtered_labels,
            autopct=lambda pct: f'{pct:.1f}%' if pct > 0 else '',
            startangle=90,
            colors=colors,
            textprops={"fontsize": 15, "fontweight": "bold"},
            pctdistance=0.72,
            labeldistance=1.18,
            wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
            rotatelabels=False,
        )
        ax.set_title(CAT_NAME.get(cat, f"Category {cat}"), fontsize=18, fontweight="bold", pad=18)
        
        # Process non-empty labels only
        valid_indices = [i for i, label in enumerate(filtered_labels) if label != '']
        valid_texts = [texts[i] for i in valid_indices]
        valid_wedges = [wedges[i] for i in valid_indices]
        
        if not valid_texts:
            continue
        
        # Adjust label positions to reduce overlap
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
        
        # Sort by angle and stagger neighboring label distances
        angle_order = np.argsort(angles)
        base_distances = [1.15] * len(angles)
        
        for idx, i in enumerate(angle_order):
            # Check angle difference with neighbors
            if idx > 0:
                prev_i = angle_order[idx - 1]
                angle_diff = abs(angles[i] - angles[prev_i])
                if angle_diff < 45:
                    # Alternate label distances
                    if idx % 2 == 0:
                        base_distances[i] = 1.25
                    else:
                        base_distances[prev_i] = 1.25
        
        # Apply base distances
        for i, (angle, dist) in enumerate(zip(angles, base_distances)):
            angle_rad = np.deg2rad(angle)
            positions[i][0] = np.cos(angle_rad) * dist
            positions[i][1] = np.sin(angle_rad) * dist
        
        # Detect and adjust overlaps; prefer tangential movement to reduce line crossings
        adjusted = True
        max_iterations = 60
        iteration = 0
        min_distance = 0.18
        max_radius = 1.3  # Maximum label radius
        
        while adjusted and iteration < max_iterations:
            adjusted = False
            iteration += 1
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    dx = positions[i][0] - positions[j][0]
                    dy = positions[i][1] - positions[j][1]
                    distance = np.sqrt(dx**2 + dy**2)
                    
                    # If two labels are too close
                    if distance < min_distance:
                        adjusted = True
                        
                        # Get label angles
                        angle_i = np.arctan2(positions[i][1], positions[i][0])
                        angle_j = np.arctan2(positions[j][1], positions[j][0])
                        
                        # Push along tangent first (perpendicular to radial direction)
                        # to reduce connector-line crossings
                        tangent_i = np.array([-np.sin(angle_i), np.cos(angle_i)])
                        tangent_j = np.array([-np.sin(angle_j), np.cos(angle_j)])
                        
                        # Decide push direction along tangent
                        if angle_i < angle_j:
                            push_i = -tangent_i * 0.06  # counter-clockwise
                            push_j = tangent_j * 0.06   # clockwise
                        else:
                            push_i = tangent_i * 0.06
                            push_j = -tangent_j * 0.06
                        
                        # Push the two labels apart
                        new_pos_i = [positions[i][0] + push_i[0], positions[i][1] + push_i[1]]
                        new_pos_j = [positions[j][0] + push_j[0], positions[j][1] + push_j[1]]
                        
                        # Check max radius constraint
                        radius_i = np.sqrt(new_pos_i[0]**2 + new_pos_i[1]**2)
                        radius_j = np.sqrt(new_pos_j[0]**2 + new_pos_j[1]**2)
                        
                        if radius_i <= max_radius:
                            positions[i] = new_pos_i
                        if radius_j <= max_radius:
                            positions[j] = new_pos_j
        
        # Apply adjusted positions
        for text, pos in zip(valid_texts, positions):
            text.set_position(pos)
        
        # Draw connector lines manually from wedge edge to label
        for wedge, text in zip(valid_wedges, valid_texts):
            # Get wedge center angle
            angle = (wedge.theta2 + wedge.theta1) / 2.0
            # Compute wedge-edge point (radius=1)
            x1 = np.cos(np.deg2rad(angle))
            y1 = np.sin(np.deg2rad(angle))
            # Get label position
            x2, y2 = text.get_position()
            # Draw connector line
            ax.plot([x1, x2], [y1, y2], color='gray', linewidth=0.5, linestyle='-', alpha=0.6)

        # Keep percentage labels only for the top-3 slices in each category
        top3_idx = np.argsort(data)[-3:]
        for idx, autotext in enumerate(autotexts):
            if idx not in top3_idx:
                autotext.set_text("")
            else:
                autotext.set_fontsize(15)
                autotext.set_fontweight("bold")

    # Hide unused subplots when categories < 4
    for j in range(len(categories), 4):
        axes[j].set_axis_off()

    save_path = out_dir / filename
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Pie charts saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot line/bar/pie charts for human-annotation label distribution "
            "from human_annotation_stats*.txt."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        help=(
            "Input stats file path. If omitted, auto-select the newest file matching "
            "data/output/evalresult/human_annotation_stats*.txt."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output directory for figures (default: data/output/plot_result/).",
    )
    args = parser.parse_args()

    # This script is in MemEval/plot/; project root is one level up
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent

    if args.input:
        stats_path = Path(args.input)
        if not stats_path.exists():
            raise FileNotFoundError(f"Specified stats file does not exist: {stats_path}")
    else:
        eval_dir = project_root / "data" / "output" / "evalresult"
        candidates = sorted(
            eval_dir.glob("human_annotation_stats*.txt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(
                f"No matching files found under {eval_dir}: human_annotation_stats*.txt"
            )
        stats_path = candidates[0]
        print(f"Using latest stats file: {stats_path}")

    out_dir = Path(args.output) if args.output else (project_root / "data" / "output" / "plot_result")
    out_dir.mkdir(parents=True, exist_ok=True)

    categories, label_names, counts = load_human_annotation_stats(stats_path)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    plot_line_chart(categories, label_names, counts, out_dir=out_dir, filename=f"human_label_line_{timestamp}.png")
    plot_bar_chart(categories, label_names, counts, out_dir=out_dir, filename=f"human_label_bar_{timestamp}.png")
    plot_pie_charts(categories, label_names, counts, out_dir=out_dir, filename=f"human_label_pies_{timestamp}.png")

    # Uncomment for interactive display
    # plt.show()


if __name__ == "__main__":
    main()


