import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Keep global style consistent with other plots
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
    Parse category-level exact-match rates from model_label_exact.txt
    for a given model.

    Returns:
        categories: ['1','2','3','4']
        rates: [float, ...]  # Percentage values
    """
    text = path.read_text(encoding="utf-8").splitlines()

    # Locate target model section (English-first, with backward-compatible Chinese fallback)
    start_idx = None
    header_candidates = [
        f"[Exact label match] Model: {model_name}",
        f"[Label 精确匹配] 模型: {model_name}",
    ]
    for i, line in enumerate(text):
        if line.strip() in header_candidates:
            start_idx = i
            break
    if start_idx is None:
        raise ValueError(f"Section not found in {path}: any of {header_candidates}")

    categories = []
    rates = []

    # Regex supports both:
    # - Category X exact label match rate (including EMPTY): Y%
    # - legacy Chinese variant of the same line
    pattern = re.compile(
        r"Category\s+(\d)\s*(?:exact label match rate \(including EMPTY\)|\uff08\u542b EMPTY\uff09label \u7cbe\u786e\u5339\u914d\u7387)\s*:\s*([\d.]+)%"
    )

    for line in text[start_idx + 1 :]:
        line = line.strip()
        if not line:
            # Section end
            break
        m = pattern.search(line)
        if m:
            cat = m.group(1)
            rate = float(m.group(2))
            categories.append(cat)
            rates.append(rate)

    if not categories:
        raise ValueError(
            f"No category-level exact-match lines parsed in section in {path}."
        )

    # Sort by category id
    paired = sorted(zip(categories, rates), key=lambda x: int(x[0]))
    categories_sorted = [p[0] for p in paired]
    rates_sorted = [p[1] for p in paired]
    return categories_sorted, rates_sorted


def load_phase_level_category_rates(path: Path, model_name: str = "voting_final"):
    """
    Parse category-level phase match rates (Overall column) from model_phase.txt
    for a given model.

    Returns:
        categories: ['1','2','3','4']
        rates: [float, ...]  # Percentage values
    """
    lines = path.read_text(encoding="utf-8").splitlines()

    # Locate target model section (English-first, with Chinese fallback)
    start_idx = None
    header_candidates = [f"Model: {model_name}", f"模型: {model_name}"]
    for i, line in enumerate(lines):
        if line.strip() in header_candidates:
            start_idx = i
            break
    if start_idx is None:
        raise ValueError(f"Section not found in {path}: any of {header_candidates}")

    categories = []
    rates = []

    # Inside model section, parse data rows after table header
    # |  1   |   97.4   | ... |  72.34   |
    in_table = False
    for line in lines[start_idx + 1 :]:
        stripped = line.strip()
        if not stripped:
            # Section end
            break
        if stripped.startswith("Total samples:") or stripped.startswith("\u603b\u6837\u672c\u6570:"):
            # Stop before summary lines
            break
        if stripped.startswith("| Cat"):
            in_table = True
            continue
        if not in_table:
            continue
        if not stripped.startswith("|"):
            continue
        if stripped.startswith("|---") or stripped.startswith("+"):
            # Separator line
            continue

        # Trim surrounding '|' and split columns
        parts = [p.strip() for p in stripped.strip("|").split("|")]
        if not parts or parts[0] == "Cat":
            continue

        cat = parts[0]
        # Overall is the last column
        try:
            overall_str = parts[-1]
            rate = float(overall_str)
        except ValueError as e:
            raise ValueError(f"Failed to parse Overall column: line={stripped}") from e

        categories.append(cat)
        rates.append(rate)

    if not categories:
        raise ValueError(
            f"No category Overall rows parsed in model section in {path}."
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
    """Plot a bar chart for one group of values."""
    x = np.arange(len(xlabels))
    width = 0.6

    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
    bars = ax.bar(x, rates, width=width, color="#4477AA", edgecolor="black")

    ax.set_xticks(x)
    # Keep x-axis labels horizontal, in given order
    ax.set_xticklabels(xlabels, rotation=0, ha="center")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, max(rates) * 1.15)

    # Annotate values above bars
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
    parser = argparse.ArgumentParser(
        description=(
            "Plot category-level consistency bar charts from model_label_exact and "
            "model_phase reports."
        )
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="voting_final",
        help="Model name to parse from report files (default: voting_final).",
    )
    parser.add_argument(
        "--label-file",
        type=str,
        default=None,
        help=(
            "Path to the label-level report. If omitted, use "
            "data/output/evalresult/model_label_exact.txt."
        ),
    )
    parser.add_argument(
        "--phase-file",
        type=str,
        default=None,
        help=(
            "Path to the phase-level report. If omitted, use "
            "data/output/evalresult/model_phase.txt."
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

    label_file = Path(args.label_file) if args.label_file else (
        project_root / "data" / "output" / "evalresult" / "model_label_exact.txt"
    )
    phase_file = Path(args.phase_file) if args.phase_file else (
        project_root / "data" / "output" / "evalresult" / "model_phase.txt"
    )

    if not label_file.exists():
        raise FileNotFoundError(f"File not found: {label_file}")
    if not phase_file.exists():
        raise FileNotFoundError(f"File not found: {phase_file}")

    out_dir = Path(args.output) if args.output else (project_root / "data" / "output" / "plot_result")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use specified model results (by category)
    categories_l, rates_l = load_label_level_category_rates(
        label_file, model_name=args.model
    )
    categories_p, rates_p = load_phase_level_category_rates(
        phase_file, model_name=args.model
    )

    # Map Cat IDs to more descriptive category names
    xlabels_l = [CAT_NAME.get(c, f"Cat {c}") for c in categories_l]
    xlabels_p = [CAT_NAME.get(c, f"Cat {c}") for c in categories_p]

    # Plot category-level bars for label-level metrics
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = args.model.replace(" ", "_")
    out_label = out_dir / f"model_label_exact_{safe_model}_bar_{timestamp}.png"
    plot_bar_chart(
        xlabels_l,
        rates_l,
        title="Label-level exact match by category",
        ylabel="Exact match accuracy (%)",
        out_path=out_label,
    )

    # Plot category-level bars for phase-level metrics
    out_phase = out_dir / f"model_phase_{safe_model}_bar_{timestamp}.png"
    plot_bar_chart(
        xlabels_p,
        rates_p,
        title="Phase-level match by category",
        ylabel="Match accuracy (%)",
        out_path=out_phase,
    )


if __name__ == "__main__":
    main()


