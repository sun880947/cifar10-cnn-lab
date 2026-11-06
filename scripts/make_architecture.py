"""Draw the baseline and DS+SE residual block diagrams used in the report."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

OUT = os.path.join("report", "figures", "architecture.png")

COLORS = {
    "conv": "#dbe9f6",
    "norm": "#e8f3e4",
    "act": "#fdebd0",
    "att": "#f6dce8",
    "add": "#eeeeee",
}


def box(ax, x, y, w, h, text, color, fontsize=9.2):
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.0, edgecolor="#444444", facecolor=color, zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, zorder=3)


def arrow(ax, x1, y1, x2, y2, style="-|>", lw=1.1):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=9,
        linewidth=lw, color="#333333", zorder=1,
    ))


def skip(ax, x_start, y, x_end, y_top):
    ax.plot([x_start, x_start], [y + 0.46, y_top], color="#333333", lw=1.1, zorder=1)
    ax.plot([x_start, x_end], [y_top, y_top], color="#333333", lw=1.1, zorder=1)
    arrow(ax, x_end, y_top, x_end, y + 0.46)
    ax.text((x_start + x_end) / 2, y_top + 0.10, "identity shortcut", ha="center", fontsize=8.4, color="#555555")


def panel(ax, title, ds_se=False):
    ax.set_xlim(-0.35, 10.1)
    ax.set_ylim(0.2, 4.5)
    ax.axis("off")
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=6)
    y = 2.0
    h = 1.0

    if not ds_se:
        labels = [
            (1.15, "Conv 3×3\n+ BN + ReLU", COLORS["conv"]),
            (3.85, "Conv 3×3\n+ BN", COLORS["conv"]),
        ]
        widths = [1.85, 1.75]
    else:
        labels = [
            (1.55, "DSConv\n(DW 3×3 + PW 1×1)\n+ BN + ReLU", COLORS["conv"]),
            (4.45, "DSConv\n(DW 3×3 + PW 1×1)\n+ BN + SE", COLORS["conv"]),
        ]
        widths = [2.30, 2.30]

    # input
    ax.text(0.05, y, "x", fontsize=10, ha="center", va="center", style="italic")
    arrow(ax, 0.12, y, labels[0][0] - widths[0] / 2 - 0.02, y)
    for (x, text, color), w in zip(labels, widths):
        box(ax, x, y, w, h, text, color)

    # arrows between stage boxes
    arrow(ax, labels[0][0] + widths[0] / 2, y, labels[1][0] - widths[1] / 2, y)

    add_x = 6.55 if ds_se else 5.95
    arrow(ax, labels[1][0] + widths[1] / 2, y, add_x - 0.20, y)
    ax.add_patch(Circle((add_x, y), 0.21, facecolor=COLORS["add"], edgecolor="#444444", lw=1.0, zorder=3))
    ax.text(add_x, y, "+", fontsize=12, ha="center", va="center", zorder=4)
    skip(ax, 0.45, y, add_x, 3.55)

    arrow(ax, add_x + 0.21, y, 7.55, y)
    box(ax, 8.15, y, 1.05, 0.82, "ReLU", COLORS["act"], fontsize=9.5)
    arrow(ax, 8.68, y, 9.55, y)
    ax.text(9.80, y, "out", fontsize=9.5, ha="center", va="center")

    ax.text(
        5.0, 0.55,
        "SE: global average pool → FC(C/8) → ReLU → FC(C) → sigmoid → channel-wise scale"
        if ds_se else "Standard residual block: two 3×3 convolutions with BatchNorm",
        ha="center", fontsize=8.6, color="#444444",
    )


def main() -> None:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.2))
    panel(axes[0], "(a) Baseline BasicBlock (R20)", ds_se=False)
    panel(axes[1], "(b) Modified BasicBlock (R20-DS-SE)", ds_se=True)
    fig.tight_layout(h_pad=1.4)
    fig.savefig(OUT, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()