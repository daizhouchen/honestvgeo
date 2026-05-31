"""Standalone per-cohort heterogeneity figure (C2) for the main body.

Information-dense ANNOTATED HEATMAP (replaces the former grouped-bar version) so
the paper's most memorable finding --- per-cohort heterogeneity, with CoGEO
peaking on the harder Complement cohort (+36.8, 1.95x PGD-bare) and only matched
on Exact --- is legible at a glance and reads as a single warm "Complement
column" rather than a wall of bars.

Every number matches Table II / the cohort paragraph exactly (n=560 ESCI):
  CoGEO     [3.90, 5.24, 36.77, 25.43]
  PGD-bare  [7.12, 1.84, 18.86, 18.53]
  AdvCLIP   [1.00, 0.94, 14.46,  5.53]
  Co-Attack [6.18, 5.47, -1.86, 16.94]

Morandi palette (sage-clay for low/negative -> mist neutral -> dust-rose for the
high-lift cells), white background, low saturation, one slim colorbar.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#444",
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

METHODS = ["CoGEO", "PGD-bare", "AdvCLIP", "Co-Attack"]
COHORTS = ["Exact", "Substitute", "Complement", "Irrelevant"]
COHORT_N = [125, 125, 185, 125]
VALS = np.array(
    [
        [3.90, 5.24, 36.77, 25.43],   # CoGEO
        [7.12, 1.84, 18.86, 18.53],   # PGD-bare
        [1.00, 0.94, 14.46, 5.53],    # AdvCLIP
        [6.18, 5.47, -1.86, 16.94],   # Co-Attack
    ]
)

# Morandi diverging colormap: sage (negative) -> warm mist (mid) -> deep rose (high lift)
cmap = LinearSegmentedColormap.from_list(
    "morandi_div",
    ["#7F8F84", "#B7A99A", "#EFEAE3", "#E6CFC6", "#C99B98", "#9B5F5C"],
)
norm = TwoSlopeNorm(vmin=-3.0, vcenter=8.0, vmax=37.0)

fig, ax = plt.subplots(figsize=(3.45, 2.55))
im = ax.imshow(VALS, cmap=cmap, norm=norm, aspect="auto")

# annotate every cell; text colour flips to white on the deep-rose high cells
for i in range(VALS.shape[0]):
    for j in range(VALS.shape[1]):
        v = VALS[i, j]
        txt_col = "white" if v >= 22 else "#2B2B2B"
        ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                fontsize=8.2, color=txt_col,
                fontweight="bold" if (i == 0 and j == 2) else "normal")

# highlight the headline CoGEO x Complement peak cell
ax.add_patch(Rectangle((2 - 0.5, 0 - 0.5), 1, 1, fill=False,
                        edgecolor="#5C3A3A", linewidth=1.8, zorder=5))

ax.set_xticks(np.arange(len(COHORTS)))
ax.set_xticklabels([f"{c}\n$n{{=}}{n}$" for c, n in zip(COHORTS, COHORT_N)], fontsize=7.3)
ax.set_yticks(np.arange(len(METHODS)))
ax.set_yticklabels(METHODS, fontsize=7.8)
ax.tick_params(length=0)
for spine in ax.spines.values():
    spine.set_visible(False)

# minor grid lines between cells for a clean tiled look
ax.set_xticks(np.arange(-0.5, len(COHORTS), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(METHODS), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=2.2)
ax.tick_params(which="minor", length=0)

ax.set_title("Per-cohort rank-lift heatmap (560-pair ESCI)", fontsize=8.6, pad=24)

# single clean headline callout above the peak cell (cells + caption carry the rest)
ax.annotate(
    "CoGEO peak $+36.8$ ($1.95\\times$ PGD-bare)",
    xy=(2, -0.48), xytext=(2, -1.18),
    fontsize=6.8, color="#5C3A3A", ha="center", va="bottom",
    arrowprops=dict(arrowstyle="->", color="#8A6A6A", linewidth=1.0, shrinkB=3),
    annotation_clip=False,
)

# slim colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03, aspect=18)
cbar.set_label("mean rank-lift", fontsize=6.8)
cbar.ax.tick_params(labelsize=6.2, length=2)
cbar.outline.set_visible(False)

fig.subplots_adjust(top=0.78, bottom=0.20, left=0.20, right=0.99)
fig.savefig(os.path.join(HERE, "fig_cohort_c2.pdf"))
fig.savefig(os.path.join(HERE, "fig_cohort_c2.png"), dpi=200)
plt.close(fig)
print("done: fig_cohort_c2 heatmap (per-cohort heterogeneity, single-column)")
print("CoGEO row:", list(VALS[0]), "Complement peak:", VALS[0, 2])
