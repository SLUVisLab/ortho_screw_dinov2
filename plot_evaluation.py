"""Plot an evaluation report: per-class accuracy, precision, and F1, plus a confusion matrix.

Usage:
    python plot_evaluation.py reports/checkpoint_evaluation.json
    python plot_evaluation.py runs/new_training/reevaluation.json --output runs/new_training/eval.svg

Writes a light and a dark variant (<output> and <output stem>_dark.<ext>). The README
embeds both with a <picture> element so the figure matches the reader's GitHub theme.
Precision and F1 are recomputed from the confusion matrix, so older reports also work.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Validated palette: categorical slots 1-3 (all-pairs CVD-safe), sequential blue ramp.
THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
        "ramp": ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70"],
        # dark mode: near-zero recedes into the dark surface, high values are bright
        "ramp": ["#1a1a19", "#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"],
    },
}
METRICS = [("recall", "Per-class accuracy (recall)"), ("precision", "Precision"), ("f1", "F1")]
SHORT_NAMES = {
    "Depuy expedium (Synthes)": "DePuy Expedium",
    "medtronic solera": "Medtronic Solera",
    "nuvasive reline": "NuVasive Reline",
    "seaspine mariner": "SeaSpine Mariner",
}


def per_class_metrics(confusion):
    support, predicted, correct = confusion.sum(1), confusion.sum(0), np.diag(confusion)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(support > 0, correct / support, np.nan)
        precision = np.where(predicted > 0, correct / predicted, np.nan)
        f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0.0)
    f1 = np.where(np.isnan(precision) | np.isnan(recall), np.nan, f1)
    return {"recall": recall, "precision": precision, "f1": f1}


def plot(report, output, theme_name):
    t = THEMES[theme_name]
    confusion = np.array(report["confusion_matrix_true_rows_predicted_columns"])
    classes = [SHORT_NAMES.get(c, c) for c in report["class_order"]]
    metrics = per_class_metrics(confusion)
    support = confusion.sum(1)
    accuracy = 100 * np.trace(confusion) / confusion.sum()
    balanced = 100 * np.nanmean(metrics["recall"])
    macro_f1 = 100 * np.nanmean(metrics["f1"])
    n_classes = len(classes)

    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "svg.fonttype": "none",  # keep text as text in the SVG
        "text.color": t["ink"], "axes.labelcolor": t["ink2"],
        "xtick.color": t["muted"], "ytick.color": t["ink2"],
    })
    fig = plt.figure(figsize=(12, 5.8), facecolor=t["surface"])
    grid = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.32,
                            left=0.06, right=0.97, top=0.70, bottom=0.15)

    # --- Title + headline numbers ---
    patients = report.get("evaluation_patients")
    pairs = report.get("valid_pairs", int(confusion.sum()))
    fig.text(0.06, 0.95, "Dual-view DINOv2: evaluation-split performance",
             fontsize=15, fontweight="bold", color=t["ink"], va="top")
    sub = f"{pairs} crop pairs" + (f" from {patients} patients" if patients else "")
    sub += " · checkpoint-selection split, not an independent test set"
    fig.text(0.06, 0.895, sub, fontsize=10, color=t["ink2"], va="top")
    headline = f"Accuracy {accuracy:.1f}%     Balanced accuracy {balanced:.1f}%     Macro-F1 {macro_f1:.1f}%"
    fig.text(0.06, 0.84, headline, fontsize=12, color=t["ink"], va="top")

    # --- Left: grouped bars per class ---
    ax = fig.add_subplot(grid[0])
    ax.set_facecolor(t["surface"])
    group_w, gap = 0.78, 0.02  # 3 bars per class with a small surface-colored gap
    bar_w = group_w / len(METRICS)
    x = np.arange(n_classes)
    for i, (key, label) in enumerate(METRICS):
        vals = 100 * np.nan_to_num(metrics[key], nan=0.0)
        offs = x - group_w / 2 + bar_w * (i + 0.5)
        ax.bar(offs, vals, width=bar_w - gap, color=t["series"][i], label=label, zorder=3)
        for xo, v, raw in zip(offs, vals, metrics[key]):
            if np.isnan(raw):  # undefined (class never predicted): mark instead of drawing a zero bar
                ax.text(xo, 2, "n/a", ha="center", va="bottom", fontsize=7.5, color=t["muted"], rotation=90)
    ax.axhline(100 / n_classes, color=t["muted"], linewidth=1, linestyle=(0, (4, 3)), zorder=2)
    ax.text(n_classes - 0.5, 100 / n_classes + 1.5, f"chance ({100 / n_classes:.0f}%)",
            ha="right", va="bottom", fontsize=8.5, color=t["muted"])
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([f"{v}%" for v in [0, 25, 50, 75, 100]], fontsize=9, color=t["muted"])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n{n} pairs" for c, n in zip(classes, support)], fontsize=9.5)
    ax.tick_params(axis="both", length=0, pad=6)
    ax.grid(axis="y", color=t["grid"], linewidth=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.set_title("Per-class metrics", loc="left", fontsize=11.5, color=t["ink"], pad=10)
    leg = ax.legend(loc="upper left", bbox_to_anchor=(0, -0.13), ncol=3, frameon=False,
                    fontsize=9, handlelength=1.0, handleheight=0.8, columnspacing=1.4)
    for text in leg.get_texts():
        text.set_color(t["ink2"])

    # --- Right: confusion matrix, row-normalized (share of each true class) ---
    ax2 = fig.add_subplot(grid[1])
    ax2.set_facecolor(t["surface"])
    shares = confusion / np.maximum(confusion.sum(1, keepdims=True), 1)
    cmap = LinearSegmentedColormap.from_list("seq", t["ramp"])
    ax2.imshow(shares, cmap=cmap, vmin=0, vmax=1, aspect="equal")
    for r in range(n_classes):
        for c in range(n_classes):
            share = shares[r, c]
            strong = share >= 0.45
            if theme_name == "light":
                color = "#ffffff" if strong else t["ink"]
            else:
                color = "#0b0b0b" if share >= 0.75 else t["ink"]
            ax2.text(c, r - 0.1, f"{100 * share:.0f}%", ha="center", va="center", fontsize=10.5,
                     color=color, fontweight="bold" if r == c else "normal")
            ax2.text(c, r + 0.22, f"{confusion[r, c]}", ha="center", va="center", fontsize=8,
                     color=color, alpha=0.85)
    # 2px-style surface gaps between cells
    for k in range(n_classes + 1):
        ax2.axhline(k - 0.5, color=t["surface"], linewidth=2)
        ax2.axvline(k - 0.5, color=t["surface"], linewidth=2)
    ax2.set_xticks(range(n_classes))
    ax2.set_yticks(range(n_classes))
    ax2.set_xticklabels([c.replace(" ", "\n", 1) for c in classes], fontsize=9, color=t["ink2"])
    ax2.set_yticklabels(classes, fontsize=9, color=t["ink2"])
    ax2.tick_params(length=0, pad=5)
    ax2.set_xlabel("Predicted manufacturer", fontsize=9.5, color=t["ink2"], labelpad=8)
    ax2.set_ylabel("True manufacturer", fontsize=9.5, color=t["ink2"], labelpad=8)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.set_title("Confusion matrix (% of each true class; pair counts)", loc="left",
                  fontsize=11.5, color=t["ink"], pad=10)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor=t["surface"], dpi=150)
    plt.close(fig)
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("report", type=Path, help="Evaluation JSON written by evaluate.py")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output image (.svg or .png). Default: next to the report, same stem.")
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    output = args.output or args.report.with_suffix(".svg")
    plot(report, output, "light")
    plot(report, output.with_name(f"{output.stem}_dark{output.suffix}"), "dark")


if __name__ == "__main__":
    main()
