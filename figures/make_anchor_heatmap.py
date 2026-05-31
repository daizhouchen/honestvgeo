"""Anchor x backbone heatmap: CoGEO mean rank-lift for 5 VLM caption anchors
(rows) x 6 CLIP retrieval backbones (columns) on the 491-ASIN ESCI manifest
(eps=16/255, 224x224). This dense 5x6 numeric matrix is the natural heatmap:
the per-backbone column means span 4.1-16.9 while the per-anchor row means span
only 7.7-9.6, so the retriever --- not the anchor captioner --- explains the bulk
of the spread (the operational signature of an anchor-decoupled protocol).

Same Morandi colormap as the per-cohort figure family.
Values match the anchor-source-robustness table exactly.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 8.6,
        "axes.labelsize": 8,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#444",
        "xtick.color": "#444",
        "ytick.color": "#444",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

HERE = os.path.dirname(os.path.abspath(__file__))

ANCHORS = ["BLIP", "BLIP-2", "LLaVA-1.5", "Qwen2-VL", "Qwen2.5-VL"]
BACKBONES = ["OAI\nL/14", "EVA\nL/14", "OC-L\nL/14", "SigLIP\nB/16", "OC-B\nB/32", "OC-H\nH/14"]
VALS = np.array([
    [14.11, 6.85, 5.72, 5.97, 16.49, 4.24],   # BLIP
    [12.91, 10.69, 4.60, 6.10, 17.79, 5.24],  # BLIP-2
    [10.16, 7.92, 3.28, 4.65, 17.26, 3.08],   # LLaVA-1.5
    [14.61, 8.30, 5.23, 5.01, 17.72, 3.87],   # Qwen2-VL
    [12.58, 7.20, 1.87, 6.25, 15.12, 4.07],   # Qwen2.5-VL
])
col_mean = VALS.mean(axis=0)   # per-backbone (12.87,8.19,4.14,5.60,16.87,4.10)
row_mean = VALS.mean(axis=1)   # per-anchor   (8.90,9.55,7.73,9.12,7.85)

# same Morandi sequential ramp (sage -> mist -> rose) used by the cohort family
cmap = LinearSegmentedColormap.from_list(
    "morandi_seq", ["#7F8F84", "#B7A99A", "#EFEAE3", "#E6CFC6", "#C99B98", "#9B5F5C"])
norm = Normalize(vmin=1.5, vmax=18.5)

fig, ax = plt.subplots(figsize=(3.45, 2.42))
im = ax.imshow(VALS, cmap=cmap, norm=norm, aspect="auto")
for i in range(VALS.shape[0]):
    for j in range(VALS.shape[1]):
        v = VALS[i, j]
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.0,
                color="white" if v >= 13.5 else "#2B2B2B")

ax.set_xticks(np.arange(len(BACKBONES)))
ax.set_xticklabels(BACKBONES, fontsize=6.6)
ax.set_yticks(np.arange(len(ANCHORS)))
ax.set_yticklabels(ANCHORS, fontsize=7.2)
ax.tick_params(length=0)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.set_xticks(np.arange(-0.5, len(BACKBONES), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(ANCHORS), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=2.0)
ax.tick_params(which="minor", length=0)
ax.set_xlabel("CLIP retrieval backbone", fontsize=7.4)
ax.set_ylabel("VLM caption anchor", fontsize=7.4)
ax.set_title("CoGEO rank-lift: anchor $\\times$ backbone", fontsize=8.4, pad=6)

# the column-spread >> row-spread message lives in the LaTeX caption; keeping it
# out of the figure avoids duplicated text and the prior right-edge clipping.

cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03, aspect=18)
cbar.set_label("mean rank-lift", fontsize=6.8)
cbar.set_ticks([3, 6, 9, 12, 15, 18])
cbar.ax.tick_params(labelsize=6.2, length=2)
cbar.outline.set_visible(False)

fig.subplots_adjust(top=0.90, bottom=0.17, left=0.20, right=0.88)
fig.savefig(os.path.join(HERE, "fig_anchor_heatmap.pdf"))
fig.savefig(os.path.join(HERE, "fig_anchor_heatmap.png"), dpi=200)
plt.close(fig)
print("done: fig_anchor_heatmap (5 anchors x 6 backbones)")
print("col means:", [round(x, 2) for x in col_mean], "row means:", [round(x, 2) for x in row_mean])
