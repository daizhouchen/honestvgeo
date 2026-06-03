"""Regenerate three figures from measured ESCI data.

1. fig_percohort_bars_v2: clean per-cohort 4-method bars (Figure 5 redo, fixes
   legend/title squeeze and bottom-label overlap from earlier review).
2. fig_d_heatmap: 5-VLM × 6-backbone CoGEO rank-lift heatmap from D matrix
   (new evidence: anchor-source × encoder heterogeneity).
3. fig_f_panel_v2: 3-backbone × 4-method bars from measured F panel (replaces
   fig_per_backbone_bars which mixed measured + hardcoded numbers).

Morandi palette is matched to the rest of the paper's figures.
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
        "axes.titlesize": 18,
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

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(HERE, "..", "results")
AGG_PATH = os.path.join(PAPER, "esci_full_aggregate.json")

with open(AGG_PATH) as f:
    AGG = json.load(f)

# ============================================================
# Figure 5 redo — fig_percohort_bars_v2
# Uses the existing per-cohort numbers (560-pair main, OpenAI L/14).
# Visual fixes: title at top with breathing room; legend below the figure
# (outside axes); short x-tick labels with cohort size on second line; no
# inline annotation arrow.
# ============================================================
COHORTS = ["E (Exact)", "S (Substitute)", "C (Complement)", "I (Irrelevant)"]
COHORT_N = [125, 125, 185, 125]
METHODS_5 = ["CoGEO", "PGD-bare", "AdvCLIP", "Co-Attack"]
COLORS_5 = [ROSE[2], SAGE[2], MIST[2], SAGE[1]]
VALS_5 = np.array(
    [
        [3.90, 5.24, 28.26, 25.43],  # CoGEO  E,S,C,I
        [7.12, 1.84, 27.43, 18.53],  # PGD-bare
        [1.00, 0.94, 14.46, 5.53],   # AdvCLIP
        [6.18, 5.47, 22.69, 16.94],  # Co-Attack
    ]
)

fig, ax = plt.subplots(figsize=(7.4, 4.6))
x = np.arange(len(COHORTS))
w = 0.20
for i, (m, c) in enumerate(zip(METHODS_5, COLORS_5)):
    ax.bar(
        x + (i - 1.5) * w, VALS_5[i, :], w, color=c, edgecolor="#333", linewidth=0.4, label=m
    )
    for j, v in enumerate(VALS_5[i, :]):
        if v >= 0:
            ax.text(x[j] + (i - 1.5) * w, v + 0.8, f"{v:+.1f}", ha="center", fontsize=7.5, color="#333")
        else:
            ax.text(x[j] + (i - 1.5) * w, v - 2.6, f"{v:+.1f}", ha="center", fontsize=7.5, color="#7F4F4F")

ax.axhline(0, color="#999", linewidth=0.6)
ax.set_xticks(x)
ax.set_xticklabels([f"{c}\n$n={n}$" for c, n in zip(COHORTS, COHORT_N)])
ax.set_ylabel("Mean rank-lift")
ax.set_ylim(-5.5, 45)
ax.grid(axis="y", linestyle=":", alpha=0.4)
# Legend goes BELOW the figure to avoid cramming the title.
ax.legend(
    ncol=4,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
    frameon=False,
    columnspacing=2.2,
    handletextpad=0.6,
)
# Title in standard top position with explicit pad so it does not touch
# the top of the data area.
ax.set_title(
    "Per-cohort rank-lift on the 560-pair ESCI manifest (OpenAI ViT-L/14, $\\epsilon=16/255$)",
    pad=12,
)
# Give the bottom enough room for the legend, and the sides + top enough room
# that nothing overlaps anything else.
fig.subplots_adjust(top=0.88, bottom=0.22, left=0.10, right=0.97)
fig.savefig(os.path.join(HERE, "fig_percohort_bars_v2.pdf"))
fig.savefig(os.path.join(HERE, "fig_percohort_bars_v2.png"), dpi=160)
plt.close(fig)

# ============================================================
# Figure 6 new — fig_d_heatmap
# 5 VLM rows × 6 backbone columns of CoGEO rank-lift mean.
# Each cell ≡ a different (anchor source, retrieval backbone) pair on the
# same 491-image ESCI manifest. Shows that the backbone dimension dominates
# the anchor dimension.
# ============================================================
VLM_ORDER = ["BLIP", "BLIP-2", "LLaVA-1.5", "Qwen2-VL", "Qwen2.5-VL"]
VLM_KEYS = ["blip", "blip2", "llava", "qwen2", "qwen25"]
BB_ORDER = ["OpenAI L/14", "EVA-02 L/14", "OpenCLIP L/14 (laion2b)", "SigLIP B/16", "OpenCLIP B/32", "OpenCLIP H/14"]
BB_KEYS = ["openai", "eva02", "laion2b", "siglip", "vitb32", "vith14"]

mat = np.zeros((len(VLM_KEYS), len(BB_KEYS)))
for i, v in enumerate(VLM_KEYS):
    for j, b in enumerate(BB_KEYS):
        cell = AGG["d_matrix"][f"{v}_{b}"]
        mat[i, j] = cell["rank_lift_mean"] if cell else np.nan

fig, ax = plt.subplots(figsize=(8.2, 4.2))
# Morandi-tuned diverging-ish colormap from light mist to sage to rose.
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list(
    "morandi_warm",
    ["#F3EEE8", "#E7E1D6", "#D8C3BC", "#B88C8C"],
    N=256,
)
im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=max(20.0, np.nanmax(mat) + 1))
ax.set_xticks(range(len(BB_ORDER)))
ax.set_xticklabels(BB_ORDER, rotation=22, ha="right")
ax.set_yticks(range(len(VLM_ORDER)))
ax.set_yticklabels(VLM_ORDER)
ax.set_xlabel("CLIP retrieval backbone")
ax.set_ylabel("VLM caption anchor")
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat[i, j]
        if np.isnan(v):
            continue
        # Use dark ink on light cells, light ink on dark cells.
        color = "#222" if v < 12 else "#FFF"
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=9, color=color)
cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.025)
cbar.set_label("CoGEO mean rank-lift")
cbar.outline.set_visible(False)
ax.set_title(
    "CoGEO rank-lift across 5 VLM anchors × 6 CLIP backbones (ESCI 491 ASINs, $\\epsilon=16/255$)",
    pad=12,
)
fig.subplots_adjust(top=0.86, bottom=0.24, left=0.16, right=0.93)
fig.savefig(os.path.join(HERE, "fig_d_heatmap.pdf"))
fig.savefig(os.path.join(HERE, "fig_d_heatmap.png"), dpi=160)
plt.close(fig)

# ============================================================
# Figure 4 redo combined — fig_per_backbone_bars_v2 (6 backbones × 4 methods)
# Combines the existing 3 L/14 cross-pretraining cells (E1 laion2b, E5 EVA-02,
# OpenAI main) with the new 3 architecture-diverse F-panel backbones (SigLIP
# B/16, OpenCLIP B/32, OpenCLIP H/14). All use product-title anchor at
# eps=16/255 on the same 491-ASIN ESCI manifest.
# ============================================================
def load_remote_cell(name):
    p = os.path.join(PAPER, "backbones", name, "cell_summary.json")
    with open(p) as f:
        return json.load(f)

e1 = load_remote_cell("E1_laion2b_eps16")  # OpenCLIP laion2b L/14
e5 = load_remote_cell("E5_eva02_eps16")     # EVA-02 L/14
# OpenAI ViT-L/14 main: numbers from the main 4-way table (Section 4).
main_open_rl = {"cogeo": 15.71, "pgd_bare": 13.73, "advclip": 5.06, "coattack": 12.82}

bb6_keys = ["openai_main", "laion2b_e1", "eva02_e5", "siglip", "vitb32", "vith14"]
bb6_pretty = [
    "OpenAI\nViT-L/14",
    "OpenCLIP\nViT-L/14",
    "EVA-02\nViT-L/14",
    "SigLIP\nViT-B/16",
    "OpenCLIP\nViT-B/32",
    "OpenCLIP\nViT-H/14",
]
m6_keys = ["cogeo", "pgd_bare", "advclip", "coattack"]
m6_pretty = ["CoGEO", "PGD-bare", "AdvCLIP", "Co-Attack"]
m6_colors = [ROSE[2], SAGE[2], MIST[2], SAGE[1]]

bb6_vals = np.zeros((len(bb6_keys), len(m6_keys)))
for j, m in enumerate(m6_keys):
    bb6_vals[0, j] = main_open_rl[m]
    bb6_vals[1, j] = e1["methods"][m]["rank_summary"]["rank_lift_mean"]
    bb6_vals[2, j] = e5["methods"][m]["rank_summary"]["rank_lift_mean"]
    bb6_vals[3, j] = AGG["f_panel"]["siglip"][m]["rank_lift_mean"]
    bb6_vals[4, j] = AGG["f_panel"]["vitb32"][m]["rank_lift_mean"]
    bb6_vals[5, j] = AGG["f_panel"]["vith14"][m]["rank_lift_mean"]

fig, ax = plt.subplots(figsize=(8.2, 4.2))
xx6 = np.arange(len(bb6_keys))
ww6 = 0.20
for j, (m, col) in enumerate(zip(m6_pretty, m6_colors)):
    bars = ax.bar(
        xx6 + (j - 1.5) * ww6,
        bb6_vals[:, j],
        ww6,
        color=col,
        edgecolor="#333",
        linewidth=0.4,
        label=m,
    )
    for k, v in enumerate(bb6_vals[:, j]):
        ypad = 0.45 if v >= 0 else -1.4
        clr = "#333" if v >= 0 else "#7F4F4F"
        ax.text(xx6[k] + (j - 1.5) * ww6, v + ypad, f"{v:+.1f}", ha="center", fontsize=7.5, color=clr)

ax.axhline(0, color="#999", linewidth=0.6)
ax.set_xticks(xx6)
ax.set_xticklabels(bb6_pretty)
ax.set_ylabel("Mean rank-lift")
ax.set_ylim(min(-8.0, np.nanmin(bb6_vals) - 1.5), max(22.0, np.nanmax(bb6_vals) + 2.5))
ax.grid(axis="y", linestyle=":", alpha=0.4)
ax.legend(
    ncol=4,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.17),
    frameon=False,
    columnspacing=2.2,
)
ax.set_title(
    "Six-backbone four-way comparison (ESCI 491, product-title anchor, $\\epsilon=16/255$)",
    pad=12,
)
fig.subplots_adjust(top=0.88, bottom=0.22, left=0.08, right=0.98)
fig.savefig(os.path.join(HERE, "fig_per_backbone_bars_v2.pdf"))
fig.savefig(os.path.join(HERE, "fig_per_backbone_bars_v2.png"), dpi=160)
plt.close(fig)
print(
    f"6-backbone combined panel: CoGEO range "
    f"[{bb6_vals[:,0].min():.2f}, {bb6_vals[:,0].max():.2f}], "
    f"AdvCLIP min {bb6_vals[:,2].min():.2f}"
)

# ============================================================
# Figure 4 redo — fig_f_panel_v2
# 3 backbones × 4 methods, all measured (replaces fig_per_backbone_bars which
# mixed measured E1/E5 data with hardcoded OpenAI numbers).
# ============================================================
F_BB_KEYS = ["siglip", "vitb32", "vith14"]
F_BB_PRETTY = ["SigLIP\nViT-B/16", "OpenCLIP\nViT-B/32", "OpenCLIP\nViT-H/14"]
M_KEYS = ["cogeo", "pgd_bare", "advclip", "coattack"]
M_PRETTY = ["CoGEO", "PGD-bare", "AdvCLIP", "Co-Attack"]
M_COLORS = [ROSE[2], SAGE[2], MIST[2], SAGE[1]]

fmat = np.zeros((len(F_BB_KEYS), len(M_KEYS)))
for i, b in enumerate(F_BB_KEYS):
    for j, m in enumerate(M_KEYS):
        c = AGG["f_panel"][b][m]
        fmat[i, j] = c["rank_lift_mean"] if c else np.nan

fig, ax = plt.subplots(figsize=(7.2, 4.0))
xx = np.arange(len(F_BB_KEYS))
ww = 0.20
for j, (m, col) in enumerate(zip(M_PRETTY, M_COLORS)):
    bars = ax.bar(
        xx + (j - 1.5) * ww,
        fmat[:, j],
        ww,
        color=col,
        edgecolor="#333",
        linewidth=0.4,
        label=m,
    )
    for k, v in enumerate(fmat[:, j]):
        ypad = 0.45 if v >= 0 else -1.3
        clr = "#333" if v >= 0 else "#7F4F4F"
        ax.text(xx[k] + (j - 1.5) * ww, v + ypad, f"{v:+.1f}", ha="center", fontsize=8, color=clr)

ax.axhline(0, color="#999", linewidth=0.6)
ax.set_xticks(xx)
ax.set_xticklabels(F_BB_PRETTY)
ax.set_ylabel("Mean rank-lift")
ax.set_ylim(min(-2.0, np.nanmin(fmat) - 1.5), max(22.0, np.nanmax(fmat) + 2.5))
ax.grid(axis="y", linestyle=":", alpha=0.4)
ax.legend(
    ncol=4,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.16),
    frameon=False,
    columnspacing=2.2,
)
ax.set_title(
    "Architecture-diverse four-way comparison (ESCI 491, product-title anchor, $\\epsilon=16/255$)",
    pad=12,
)
fig.subplots_adjust(top=0.88, bottom=0.22, left=0.10, right=0.97)
fig.savefig(os.path.join(HERE, "fig_f_panel_v2.pdf"))
fig.savefig(os.path.join(HERE, "fig_f_panel_v2.png"), dpi=160)
plt.close(fig)

print("done: fig_percohort_bars_v2 / fig_d_heatmap / fig_f_panel_v2")
print(f"D matrix shape {mat.shape}; F panel shape {fmat.shape}")
print(
    f"D matrix range: min {np.nanmin(mat):.2f}, median {np.nanmedian(mat):.2f}, "
    f"max {np.nanmax(mat):.2f}"
)
print(
    f"F panel CoGEO per-bb: "
    f"SigLIP {fmat[0,0]:.2f}, ViT-B/32 {fmat[1,0]:.2f}, ViT-H/14 {fmat[2,0]:.2f}"
)
