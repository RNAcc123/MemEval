import argparse
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


STAGES = ["Stage 1", "Stage 2", "Stage 3", "Stage 4", "EMPTY"]


def load_confusion_matrix(path: Path):
    """
    Parse confusion-matrix text from
    human_vs_voting_final_phase_confusion*.txt or stage_confusion*.txt.

    Returns:
        true_labels: list[str]  # Row-label order
        pred_labels: list[str]  # Column-label order
        perc: np.ndarray (n_true, n_pred)  # Percentages
        counts: np.ndarray (n_true, n_pred)  # Counts
    """
    lines = path.read_text(encoding="utf-8").splitlines()

    header_line = None
    for line in lines:
        if line.startswith("| True phase") or line.startswith("| True stage"):
            header_line = line
            break
    if header_line is None:
        raise ValueError("Header row not found: '| True phase/stage | ... |'")

    header_cells = [c.strip() for c in header_line.strip().strip("|").split("|")]
    # header_cells example:
    # ['True phase/stage', 'Phase/Stage 1', ..., 'Phase/Stage 4', 'EMPTY']
    # Normalize Phase -> Stage
    pred_labels = [label.replace("Phase", "Stage") for label in header_cells[1:]]

    true_labels = []
    perc_rows = []
    count_rows = []

    row_pattern = re.compile(r"\|\s*Phase|\|\s*Stage|\|\s*EMPTY")

    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        if not row_pattern.match(stripped):
            continue
        # Trim leading/trailing separators
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if not cells:
            continue

        true_label = cells[0]
        if true_label in ("True phase", "True stage"):
            # Header has already been parsed
            continue
        # Normalize Phase -> Stage
        true_label = true_label.replace("Phase", "Stage")

        # Each cell looks like "96.04% (364)"
        row_perc = []
        row_counts = []
        for cell in cells[1:]:
            # Extract percentage and count
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
    Draw a confusion-matrix heatmap with value annotations.
    """
    n_true, n_pred = perc.shape

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

    # Use percentage as color intensity (0-100)
    im = ax.imshow(perc, cmap="Blues", vmin=0, vmax=100)

    # Axis labels
    ax.set_xticks(np.arange(n_pred))
    ax.set_yticks(np.arange(n_true))
    ax.set_xticklabels(pred_labels)
    ax.set_yticklabels(true_labels)

    ax.set_xlabel("Llm predicted stage")
    ax.set_ylabel("Human true stage")
    ax.set_title("Human-LLM(Ensemble Voting) Confusion Matrix")

    # Keep only colored cells (no gray background grid)
    ax.grid(False)

    # Keep x labels horizontal
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")

    # Annotate each cell as "xx.x%\n(n)"
    for i in range(n_true):
        for j in range(n_pred):
            text = f"{perc[i, j]:.1f}%\n({counts[i, j]})"
            # Choose text color by background intensity
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
    parser = argparse.ArgumentParser(
        description=(
            "Plot a confusion-matrix heatmap from a confusion stats text file "
            "(phase/stage confusion)."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        help=(
            "Input confusion file path. If omitted, auto-select the newest file "
            "matching human_vs_voting_final_*_confusion_*.txt in evalresult."
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
        conf_path = Path(args.input)
    else:
        eval_dir = project_root / "data" / "output" / "evalresult"
        candidates = sorted(
            eval_dir.glob("human_vs_voting_final_*_confusion_*.txt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        conf_path = candidates[0] if candidates else Path("")
    if not conf_path.exists():
        raise FileNotFoundError(f"Confusion-matrix file not found: {conf_path}")
    print(f"Using confusion file: {conf_path}")

    out_dir = Path(args.output) if args.output else (project_root / "data" / "output" / "plot_result")
    out_dir.mkdir(parents=True, exist_ok=True)

    true_labels, pred_labels, perc, counts = load_confusion_matrix(conf_path)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"human_vs_voting_final_stage_confusion_{timestamp}.png"
    plot_confusion_heatmap(true_labels, pred_labels, perc, counts, out_path)


if __name__ == "__main__":
    main()



