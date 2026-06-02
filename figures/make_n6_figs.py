"""Generate three new figures from N6 measured runs.

- fig_per_backbone_bars: 3 backbones x 4 methods (rank-lift)
- fig_eps_sweep_curve: 3 eps x 2 methods (CoGEO vs PGD-bare)
- fig_food101_bars: 4 methods on Food-101

Morandi palette: mist-stone + sage-clay + dust-rose accents.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 16,
        "axes.labelsize": 10,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#444",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": "#444",
        "ytick.color": "#444",
        "legend.frameon": False,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

MIST = ["#F3EEE8", "#D8D1C7", "#8A9199"]
SAGE = ["#E7E1D6", "#B7A99A", "#7F8F84"]
ROSE = ["#F2E9E6", "#D8C3BC", "#B88C8C"]

ROOT = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(ROOT, "..", "results")
RUNS = os.path.join(PAPER, "backbones")


def load_cell(name):
    p = os.path.join(RUNS, name, "cell_summary.json")
    with open(p) as f:
        return json.load(f)


def get_rl(cell, method):
    return cell["methods"][method]["rank_summary"]["rank_lift_mean"]


# ============================================================
# Figure 1: per-backbone bars (3 backbones x 4 methods)
# OpenAI numbers from main result (Table 1): 19.86 / 12.37 / 6.45 / 5.77
# ============================================================
e1 = load_cell("E1_laion2b_eps16")
e5 = load_cell("E5_eva02_eps16")

backbones = ["OpenAI ViT-L/14", "OpenCLIP laion2b\nViT-L/14", "EVA-02\nViT-L/14"]
methods = ["CoGEO", "PGD-bare", "AdvCLIP", "Co-Attack"]
mkeys = ["cogeo", "pgd_bare", "advclip", "coattack"]
# OpenAI 16/255 main numbers from prior recorded main experiment
main_open = {"cogeo": 19.86, "pgd_bare": 12.37, "advclip": 6.45, "coattack": 5.77}
laion = {k: get_rl(e1, k) for k in mkeys}
eva = {k: get_rl(e5, k) for k in mkeys}

vals = np.array(
    [
        [main_open[k] for k in mkeys],
        [laion[k] for k in mkeys],
        [eva[k] for k in mkeys],
    ]
)

fig, ax = plt.subplots(figsize=(7.0, 4.2))
x = np.arange(len(backbones))
w = 0.20
colors = [ROSE[2], SAGE[2], MIST[2], SAGE[1]]
for i, (m, c) in enumerate(zip(methods, colors)):
    ax.bar(x + (i - 1.5) * w, vals[:, i], w, color=c, edgecolor="#333", linewidth=0.4, label=m)

ax.axhline(0, color="#999", linewidth=0.6)
ax.set_xticks(x)
ax.set_xticklabels(backbones)
ax.set_ylabel("Mean rank-lift")
ax.set_title(
    "Cross-backbone four-way comparison (ESCI 491, $\\epsilon=16/255$, 224$\\times$224)",
    pad=14,
)
# Legend placed BELOW the x-axis labels so it cannot collide with the title.
ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
ax.grid(axis="y", linestyle=":", alpha=0.4)

# Annotate AdvCLIP-laion2b negative bar ABOVE the bar (toward y=0), not below,
# so the annotation cannot overlap the x-axis tick labels.
neg_y = laion["advclip"]
ax.annotate(
    f"{neg_y:.2f} (complete failure)",
    xy=(1 + 0.5 * w, neg_y),
    xytext=(1 + 0.5 * w, neg_y + 1.6),
    ha="center",
    va="bottom",
    fontsize=8,
    color="#8E5C5C",
    arrowprops=dict(arrowstyle="->", color="#B88C8C", linewidth=0.6),
)
# Give extra bottom margin so the legend (placed below the x-axis labels) fits.
fig.subplots_adjust(top=0.86, bottom=0.28, left=0.10, right=0.96)
fig.savefig(os.path.join(ROOT, "fig_per_backbone_bars.pdf"))
fig.savefig(os.path.join(ROOT, "fig_per_backbone_bars.png"), dpi=160)
plt.close(fig)

# ============================================================
# Figure 2: eps-sweep curve (3 eps x 2 methods)
# ============================================================
e2_4 = load_cell("E2_openai_eps4")
e2_8 = load_cell("E2_openai_eps8")
eps = [4, 8, 16]
cogeo_y = [get_rl(e2_4, "cogeo"), get_rl(e2_8, "cogeo"), 19.86]
pgd_y = [get_rl(e2_4, "pgd_bare"), get_rl(e2_8, "pgd_bare"), 12.37]

fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.plot(eps, cogeo_y, "o-", color=ROSE[2], linewidth=1.8, markersize=7, label="CoGEO")
ax.plot(eps, pgd_y, "s-", color=SAGE[2], linewidth=1.8, markersize=7, label="PGD-bare")

# Annotate winner per eps. For the topmost point, drop the label below the
# marker so it does not run into the title.
y_top = max(max(cogeo_y), max(pgd_y))
for i, e in enumerate(eps):
    if cogeo_y[i] > pgd_y[i]:
        label, base_y, color = f"+{cogeo_y[i] - pgd_y[i]:.1f}", cogeo_y[i], "#7F4F4F"
    else:
        label, base_y, color = f"{cogeo_y[i] - pgd_y[i]:.1f}", pgd_y[i], "#5A6A5F"
    below = base_y >= y_top - 1e-6
    ax.annotate(
        label,
        xy=(e, base_y),
        xytext=(e, base_y - 1.5 if below else base_y + 1.5),
        ha="center",
        va="top" if below else "bottom",
        fontsize=8,
        color=color,
    )
ax.set_xticks(eps)
ax.set_xticklabels([f"$\\epsilon=${e}/255" for e in eps])
ax.set_ylabel("Mean rank-lift")
ax.set_title("Perturbation budget sweep (ESCI 491, ViT-L/14)")
ax.legend(loc="upper left")
ax.grid(linestyle=":", alpha=0.4)
plt.tight_layout()
fig.savefig(os.path.join(ROOT, "fig_eps_sweep_curve.pdf"))
fig.savefig(os.path.join(ROOT, "fig_eps_sweep_curve.png"), dpi=160)
plt.close(fig)

# ============================================================
# Figure 3: Food-101 bars (4 methods)
# ============================================================
e4 = load_cell("E4_food101_eps16")
food_vals = [get_rl(e4, k) for k in mkeys]

fig, ax = plt.subplots(figsize=(5.6, 3.4))
bars = ax.bar(
    methods, food_vals, color=[ROSE[2], SAGE[2], MIST[2], SAGE[1]], edgecolor="#333", linewidth=0.4
)
ax.set_ylabel("Mean rank-lift")
ax.set_title("Cross-dataset four-way comparison\n(Food-101 505, ViT-L/14, $\\epsilon=16/255$)")
ax.grid(axis="y", linestyle=":", alpha=0.4)
for b, v in zip(bars, food_vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 2.5, f"{v:.1f}", ha="center", fontsize=9)
ax.set_ylim(0, max(food_vals) * 1.18)
plt.tight_layout()
fig.savefig(os.path.join(ROOT, "fig_food101_bars.pdf"))
fig.savefig(os.path.join(ROOT, "fig_food101_bars.png"), dpi=160)
plt.close(fig)

print("done: fig_per_backbone_bars / fig_eps_sweep_curve / fig_food101_bars")
print("openai_main:", main_open)
print("laion:", laion)
print("eva:", eva)
print("eps4 (cogeo,pgd):", get_rl(e2_4, "cogeo"), get_rl(e2_4, "pgd_bare"))
print("eps8 (cogeo,pgd):", get_rl(e2_8, "cogeo"), get_rl(e2_8, "pgd_bare"))
print("food101:", food_vals)
