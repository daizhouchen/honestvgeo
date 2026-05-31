"""Cross-dataset ordering-preservation bump chart (replaces the Food-101 bar).

Shows that the paired-test method ordering on ESCI (CoGEO > PGD-bare > Co-Attack
> AdvCLIP; Sec. pairwise-Wilcoxon) is preserved on a second image distribution
(Food-101), as connected rank positions rather than a 5th grouped bar chart.
Endpoint labels carry the true mean rank-lift so magnitude stays visible despite
the different candidate-pool scales (ESCI C=491, Food-101 C=505).

Values (OpenAI ViT-L/14, eps=16/255):
  ESCI      CoGEO 19.86  PGD 12.37  Co-Attack 5.77  AdvCLIP 6.45
  Food-101  CoGEO 113.2  PGD 96.5   Co-Attack 91.1  AdvCLIP 30.4
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#444",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

MIST = ["#F3EEE8", "#D8D1C7", "#8A9199"]
SAGE = ["#E7E1D6", "#B7A99A", "#7F8F84"]
ROSE = ["#F2E9E6", "#D8C3BC", "#B88C8C"]

HERE = os.path.dirname(os.path.abspath(__file__))

# (method, colour, marker, ESCI rank-lift, Food-101 rank-lift) in paired-test order
ROWS = [
    ("CoGEO",     ROSE[2], "^", 19.86, 113.2),
    ("PGD-bare",  SAGE[2], "s", 12.37, 96.5),
    ("Co-Attack", SAGE[1], "D", 5.77, 91.1),
    ("AdvCLIP",   MIST[2], "o", 6.45, 30.4),
]
# rank positions (1 = strongest) in each dataset, paired-test reading
ESCI_RANK = {"CoGEO": 1, "PGD-bare": 2, "Co-Attack": 3, "AdvCLIP": 4}
FOOD_RANK = {"CoGEO": 1, "PGD-bare": 2, "Co-Attack": 3, "AdvCLIP": 4}

fig, ax = plt.subplots(figsize=(3.4, 2.45))
for name, col, mk, e_val, f_val in ROWS:
    yL, yR = ESCI_RANK[name], FOOD_RANK[name]
    ax.plot([0, 1], [yL, yR], "-", color=col, linewidth=2.0, alpha=0.9, zorder=2)
    ax.scatter([0, 1], [yL, yR], s=46, color=col, marker=mk,
               edgecolor="#333", linewidth=0.5, zorder=3)
    ax.text(-0.06, yL, f"{name}  {e_val:.1f}", ha="right", va="center", fontsize=7.2, color="#2B2B2B")
    ax.text(1.06, yR, f"{f_val:.0f}  {name}", ha="left", va="center", fontsize=7.2, color="#2B2B2B")

ax.set_xlim(-0.95, 1.95)
ax.set_ylim(4.6, 0.4)  # rank 1 on top
ax.set_xticks([0, 1])
ax.set_xticklabels(["ESCI\n($C{=}491$)", "Food-101\n($C{=}505$)"], fontsize=7.6)
ax.set_yticks([1, 2, 3, 4])
ax.set_yticklabels(["1st", "2nd", "3rd", "4th"], fontsize=7)
ax.set_ylabel("Rank by mean rank-lift")
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.set_title("Ordering preserved across datasets", fontsize=8.6, pad=6)
ax.text(0.5, 0.20, "CoGEO$-$PGD gap $1.61\\times\\!\\to\\!1.17\\times$",
        transform=ax.transAxes, ha="center", fontsize=6.6, color="#5C3A3A", style="italic")

fig.subplots_adjust(top=0.88, bottom=0.18, left=0.20, right=0.84)
fig.savefig(os.path.join(HERE, "fig_food101_slope.pdf"))
fig.savefig(os.path.join(HERE, "fig_food101_slope.png"), dpi=200)
plt.close(fig)
print("done: fig_food101_slope bump chart (ordering preservation)")
