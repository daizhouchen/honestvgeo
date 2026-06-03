"""Main-body figure that surfaces the previously table-only robustness/defense
evidence as one information-dense two-panel display (ICDM has no supplement
channel, so these results must live in the 10-page main text).

Panel (a) Transfer below the white-box ceiling: every out-of-source transfer
probe (AE-CoGEO 4-encoder consensus, BLIP fused reranker, BLIP-2 Q-Former,
late-interaction MaxSim) sits far below the 18.47 white-box rank-lift ceiling.

Panel (b) Layered defense bounds the residual: input purification, cohort-aware
oracle purification, and a matched adaptive attacker as before->after dumbbells,
plus the cohort-adaptive-budget (CAB) negative (an adaptive budget is WORSE than
uniform), all far below the same ceiling; a training-free detector flags every
family at AUROC 0.99-1.00.

All numbers are hardcoded to match the manuscript exactly:
  transfer    AE 3.29/4.80/5.85/8.60 (pooled 5.64); BLIP 7.52; MaxSim<=1.6; BLIP-2 ~0
  purify(e=4) 8.50->1.98 (-77%); cohort-oracle ->1.18; adaptive 5.15->4.77
  CAB(e=16)   uniform 15.6 vs adaptive 13.0; ceiling 18.47

Morandi palette, white background, low saturation.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
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
CEIL = 18.47

fig, (axa, axb, axc) = plt.subplots(3, 1, figsize=(3.5, 3.9))

# ============================ (a) transfer below ceiling ============================
# rows top->bottom
labels_a = [
    "AE-CoGEO",
    "BLIP rerank",
    "MaxSim",
    "BLIP-2 QF",
]
ya = np.arange(len(labels_a))[::-1]
# AE consensus: range 3.29-8.60, pooled marker 5.64
ae_lo, ae_hi, ae_pool = 3.29, 8.60, 5.64
axa.plot([ae_lo, ae_hi], [ya[0], ya[0]], color=SAGE[1], linewidth=4.0,
         solid_capstyle="round", alpha=0.55, zorder=1)
axa.scatter([ae_pool], [ya[0]], s=55, color=ROSE[2], edgecolor="#333", linewidth=0.5, zorder=3)
axa.text(ae_hi + 0.5, ya[0], "3.3$-$8.6", va="center", fontsize=6.6, color="#444")
# single-value probes
probes = [(ya[1], 7.52, "7.52"), (ya[2], 1.59, "$\\leq$1.6"), (ya[3], 0.0, "$\\approx$0")]
for y, v, lab in probes:
    axa.scatter([v], [y], s=52, color=ROSE[2], edgecolor="#333", linewidth=0.5, zorder=3)
    axa.text(v + 0.6, y, lab, va="center", fontsize=6.6, color="#444")
# ceiling
axa.axvline(CEIL, color="#8A6A6A", linewidth=1.2, linestyle="--", zorder=2)
axa.text(CEIL - 0.4, ya[-1] - 0.62, f"white-box ceiling {CEIL:.2f}",
         ha="right", va="center", fontsize=6.4, color="#5C3A3A", style="italic")
# light band for held-out CLIP range
axa.axvspan(0, ae_hi, color=SAGE[0], alpha=0.35, zorder=0)
axa.set_yticks(ya)
axa.set_yticklabels(labels_a, fontsize=6.8)
axa.set_xlim(-0.5, 20.0)
axa.set_ylim(-1.0, len(labels_a) - 0.4)
axa.set_xlabel("Mean transfer rank-lift")
axa.grid(axis="x", linestyle=":", alpha=0.4)
axa.set_title("(a) Transfer below ceiling ($n{=}498$)", fontsize=9.0)

# ============================ (b) layered defense ============================
# before -> after dumbbells (rank-lift)
rows_b = [
    ("Purify $\\epsilon$4", 8.50, 1.98, "$-77\\%$", "#9B5F5C"),
    ("+cohort $\\epsilon$4", 1.98, 1.18, "$-40\\%$", "#9B5F5C"),
    ("Adaptive $\\epsilon$4", 5.15, 4.77, "$-7\\%$", "#9B5F5C"),
    ("CAB $\\epsilon$16", 15.6, 13.0, "backfires", "#7F8F84"),
]
yb = np.arange(len(rows_b))[::-1]
for k, (lab, before, after, tag, col) in enumerate(rows_b):
    y = yb[k]
    axb.plot([before, after], [y, y], color=SAGE[1], linewidth=2.0, zorder=1)
    axb.scatter([before], [y], s=46, color=MIST[1], edgecolor="#333", linewidth=0.5, zorder=3)
    axb.scatter([after], [y], s=52, color=col, edgecolor="#333", linewidth=0.5, zorder=3)
    if lab.startswith("CAB"):
        # the CAB row reaches toward the ceiling line, so keep its tag above the
        # dumbbell midpoint instead of to the right (where it collided with the
        # 18.47 dashed ceiling).
        axb.text((before + after) / 2, y + 0.34, tag, ha="center", va="bottom",
                 fontsize=6.4, color=col)
    else:
        axb.text(max(before, after) + 0.7, y, tag, ha="left", va="center",
                 fontsize=6.4, color=col)
axb.axvline(CEIL, color="#8A6A6A", linewidth=1.2, linestyle="--", zorder=2)
axb.set_yticks(yb)
axb.set_yticklabels([r[0] for r in rows_b], fontsize=6.8)
axb.set_xlim(-0.5, 21.0)
# extra room below the CAB row so the detector caption sits on its own line,
# well clear of the CAB dumbbell markers above it
axb.set_ylim(-1.7, len(rows_b) - 0.4)
axb.set_xlabel("Mean rank-lift")
axb.grid(axis="x", linestyle=":", alpha=0.4)
axb.legend(handles=[
    Line2D([0], [0], marker="o", color="none", markerfacecolor=MIST[1],
           markeredgecolor="#333", markersize=6, label="before"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#9B5F5C",
           markeredgecolor="#333", markersize=6, label="after defense"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#7F8F84",
           markeredgecolor="#333", markersize=6, label="CAB attacker"),
], loc="upper center", bbox_to_anchor=(0.5, -0.46), ncol=3, fontsize=6.2,
   handletextpad=0.3, columnspacing=2.0, frameon=False)
# detector caption on its own dedicated line well below the CAB dumbbell
# (data y=-1.0), left-aligned from x=0 -- clear of all markers and the x-axis
axb.text(0.0, -1.0, "Detector AUROC $\\geq$0.99 @ 5% FPR",
         fontsize=6.2, color="#5C3A3A", style="italic", ha="left", va="center")
axb.set_title("(b) Layered defense ($n{=}500$)", fontsize=9.0)

# ============================ (c) 1430-pair scale-up replication ============================
COH = ["E", "S", "C", "I"]
COH_N = [375, 375, 305, 375]
cg = [22.06, 30.77, 39.36, 72.49]
pg = [17.05, 24.66, 42.91, 50.30]
xc = np.arange(len(COH))
axc.plot(xc, cg, "^-", color=ROSE[2], linewidth=1.9, markersize=7,
         markeredgecolor="#333", label="CoGEO")
axc.plot(xc, pg, "s-", color=SAGE[2], linewidth=1.9, markersize=6,
         markeredgecolor="#333", label="PGD-bare")
axc.set_xticks(xc)
axc.set_xticklabels([f"{c}\n$n{{=}}{n}$" for c, n in zip(COH, COH_N)], fontsize=6.8)
axc.set_ylabel("Mean rank-lift")
axc.set_ylim(0, 80)
axc.grid(axis="both", linestyle=":", alpha=0.4)
axc.legend(loc="upper left", fontsize=6.6, handlelength=1.3, borderaxespad=0.3)
axc.text(0.97, 0.06, "overall $+7.89$ (95% CI $[2.0,13.6]$)",
         transform=axc.transAxes, ha="right", va="bottom", fontsize=6.2,
         color="#5C3A3A", style="italic")
axc.set_title("(c) Scale-up (1,430 pairs)", fontsize=9.0)

fig.subplots_adjust(top=0.955, bottom=0.075, left=0.265, right=0.945, hspace=1.25)
fig.savefig(os.path.join(HERE, "fig_bounded_defense.pdf"))
fig.savefig(os.path.join(HERE, "fig_bounded_defense.png"), dpi=185)
plt.close(fig)
print("done: fig_bounded_defense (transfer below ceiling + layered defense)")
