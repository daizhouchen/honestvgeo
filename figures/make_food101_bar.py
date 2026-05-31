"""Compact Food-101 cross-dataset bar chart (the natural form for a 4-method,
single-dataset comparison). The ESCI paired-test ordering
CoGEO > PGD-bare > Co-Attack > AdvCLIP is preserved on a second image
distribution; the CoGEO-PGD gap contracts from 1.61x (ESCI) to 1.17x (Food-101).

Values (Food-101, 505 paired triples, OpenAI ViT-L/14, eps=16/255):
  CoGEO 113.2  PGD-bare 96.5  Co-Attack 91.1  AdvCLIP 30.4
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#444",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": "#444",
        "ytick.color": "#444",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

MIST = ["#F3EEE8", "#D8D1C7", "#8A9199"]
SAGE = ["#E7E1D6", "#B7A99A", "#7F8F84"]
ROSE = ["#F2E9E6", "#D8C3BC", "#B88C8C"]
HERE = os.path.dirname(os.path.abspath(__file__))

METHODS = ["CoGEO", "PGD-bare", "Co-Attack", "AdvCLIP"]
VALS = [113.2, 96.5, 91.1, 30.4]
COLORS = [ROSE[2], SAGE[2], SAGE[1], MIST[2]]

fig, ax = plt.subplots(figsize=(3.3, 2.5))
x = np.arange(len(METHODS))
bars = ax.bar(x, VALS, width=0.62, color=COLORS, edgecolor="#333", linewidth=0.4, zorder=3)
for b, v in zip(bars, VALS):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.8, f"{v:.0f}", ha="center",
            fontsize=7.6, color="#2B2B2B")
ax.set_xticks(x)
ax.set_xticklabels(METHODS, fontsize=7.6)
ax.set_ylabel("Mean rank-lift")
ax.set_ylim(0, max(VALS) * 1.34)  # extra headroom so the gap note floats above every bar
ax.grid(axis="y", linestyle=":", alpha=0.4)
ax.set_title("Food-101: ordering preserved", fontsize=8.6)
ax.text(0.985, 0.99, "CoGEO$-$PGD gap\n$1.61\\times\\!\\to\\!1.17\\times$", transform=ax.transAxes,
        ha="right", va="top", fontsize=6.6, color="#5C3A3A", style="italic")

fig.subplots_adjust(top=0.89, bottom=0.13, left=0.17, right=0.97)
fig.savefig(os.path.join(HERE, "fig_food101_bar.pdf"))
fig.savefig(os.path.join(HERE, "fig_food101_bar.png"), dpi=200)
plt.close(fig)
print("done: fig_food101_bar (compact 4-method bar)")
