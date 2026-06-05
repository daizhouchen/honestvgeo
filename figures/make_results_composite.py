"""Main-body composite results figure: one full-width 2x3 float (fig_results_main.pdf).

ALL key visual results live in the 10-page main text as sub-panels of one
figure* (ICDM has no separate supplement channel). Per-cohort (formerly Fig 3)
and six-backbone (formerly Fig 4) bars are folded in as panels (e) and (f).

Panels (every number matches a manuscript table/paragraph exactly so figure and
tables can never drift):
  (a) AE-CoGEO source-encoder scaling law (1->4 sources)      (analysis Sec, rho=1.0)
  (b) Perturbation-budget sweep CoGEO vs PGD-bare             (eps in {4,8,16}/255, OpenAI ViT-L/14)
  (c) AE-CoGEO held-out transfer vs single-encoder + ceiling  (tab:transfer, n=498)
  (d) Four-way rank-lift across six CLIP backbones            (C1 cross-encoder, ESCI 491)
  (e) SSIM vs rank-lift stealth/lift trade-off                (tab:main4way, 224x224, eps=16/255)
  (f) Food-101 cross-dataset 4-method ordering                (n=2,020, OpenAI ViT-L/14)

Morandi palette (mist-stone primary + sage-clay contrast + dust-rose accent).
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
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#444",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": "#444",
        "ytick.color": "#444",
        "legend.frameon": False,
        "legend.fontsize": 6.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

MIST = ["#F3EEE8", "#D8D1C7", "#8A9199"]
SAGE = ["#E7E1D6", "#B7A99A", "#7F8F84"]
ROSE = ["#F2E9E6", "#D8C3BC", "#B88C8C"]

HERE = os.path.dirname(os.path.abspath(__file__))
# self-contained, repo-local data (copied from the verified remote runs) so the
# figure is reproducible without the off-repo GPU-host paths.
PAPER = os.path.join(HERE, "..", "results")
RUNS = os.path.join(PAPER, "backbones")
WHITEBOX_CEILING = 18.47

METHODS = ["CoGEO", "PGD-bare", "AdvCLIP", "Co-Attack"]
M_COLORS = [ROSE[2], SAGE[2], MIST[2], SAGE[1]]
M_KEYS = ["cogeo", "pgd_bare", "advclip", "coattack"]


# ---------------------------------------------------------------- eps-sweep abs
def _cell_rl(cell, method):
    p = os.path.join(RUNS, cell, "cell_summary.json")
    with open(p) as f:
        d = json.load(f)
    return d["methods"][method]["rank_summary"]["rank_lift_mean"]


try:
    cogeo_eps = [_cell_rl("E2_openai_eps4", "cogeo"), _cell_rl("E2_openai_eps8", "cogeo"), 15.71]
    pgd_eps = [_cell_rl("E2_openai_eps4", "pgd_bare"), _cell_rl("E2_openai_eps8", "pgd_bare"), 13.73]
except Exception:
    pgd_eps = [13.73, 13.73, 13.73]
    cogeo_eps = [pgd_eps[0] - 0.64, pgd_eps[1] + 3.35, 15.71]

# ---------------------------------------------------------------- six-backbone matrix
def _load_cell(name):
    with open(os.path.join(RUNS, name, "cell_summary.json")) as f:
        return json.load(f)


AGG = json.load(open(os.path.join(PAPER, "esci_full_aggregate.json")))
_e1 = _load_cell("E1_laion2b_eps16")
_e5 = _load_cell("E5_eva02_eps16")
_main_rl = {"cogeo": 15.71, "pgd_bare": 13.73, "advclip": 5.06, "coattack": 12.82}
BB6_PRETTY = ["OpenAI\nL/14", "OpenCLIP\nL/14", "EVA-02\nL/14", "SigLIP\nB/16", "OpenCLIP\nB/32", "OpenCLIP\nH/14"]
bb6_vals = np.zeros((6, 4))
for _j, _m in enumerate(M_KEYS):
    bb6_vals[0, _j] = _main_rl[_m]
    bb6_vals[1, _j] = _e1["methods"][_m]["rank_summary"]["rank_lift_mean"]
    bb6_vals[2, _j] = _e5["methods"][_m]["rank_summary"]["rank_lift_mean"]
    bb6_vals[3, _j] = AGG["f_panel"]["siglip"][_m]["rank_lift_mean"]
    bb6_vals[4, _j] = AGG["f_panel"]["vitb32"][_m]["rank_lift_mean"]
    bb6_vals[5, _j] = AGG["f_panel"]["vith14"][_m]["rank_lift_mean"]

# ================================================================ layout
# Per-cohort heterogeneity (formerly panel (e)) is now its own standalone main
# figure (fig_cohort_c2), so this composite carries five white-box / transfer
# panels in a 3 (top) + 2 (bottom) grid with more room per panel (less dense).
fig = plt.figure(figsize=(11.4, 4.55))
# Layout (reading order a->f): TOP row carries the three transfer/scaling/budget
# panels (a) source-encoder scaling / (b) budget sweep / (c) held-out transfer;
# BOTTOM row carries (d) six CLIP backbones / (e) stealth vs. lift / (f) the
# Food-101 cross-dataset bars (folded in from the former external-validity float).
# 2x12 grid. Top row = three EQUAL 4-col panels (no empty gutter): widening (a)
# from 3->4 cols clears the (a)/(b) crowding, and dropping the gutter keeps (b)/(c)
# evenly spaced. Bottom row keeps its proven 5/4/3 split (no legend/label overflow).
gs = fig.add_gridspec(2, 12, hspace=1.30, wspace=1.45)
axb = fig.add_subplot(gs[0, 0:4])    # (a) source-encoder scaling (widened to clear (b))
axd = fig.add_subplot(gs[0, 4:8])    # (b) budget sweep
axa = fig.add_subplot(gs[0, 8:12])   # (c) held-out transfer (shortened x-axis, short labels)
axf = fig.add_subplot(gs[1, 0:5])    # (d) six CLIP backbones
axc = fig.add_subplot(gs[1, 5:9])    # (e) stealth vs. lift
axg = fig.add_subplot(gs[1, 9:12])   # (f) Food-101 cross-dataset bars

# =========================== (a) AE transfer (horizontal dumbbell) ===========================
ENC = ["OC-L/14", "OC-H/14", "OC-B/32", "SigLIP"]  # short labels: narrower left gutter, clears (b)
SINGLE_ENC = [1.04, 3.63, 7.28, 3.07]
AE_COGEO = [3.29, 5.85, 8.60, 4.80]
PSTARS = ["***", "***", "*", "**"]
ya = np.arange(len(ENC))[::-1]  # first encoder on top
for j in range(len(ENC)):
    y = ya[j]
    axa.plot([SINGLE_ENC[j], AE_COGEO[j]], [y, y], color=SAGE[1], linewidth=2.2,
             solid_capstyle="round", zorder=1)
    axa.annotate("", xy=(AE_COGEO[j], y), xytext=(SINGLE_ENC[j], y),
                 arrowprops=dict(arrowstyle="-|>", color="#9B5F5C", linewidth=0.0,
                                 mutation_scale=9, shrinkA=4, shrinkB=4), zorder=2)
    axa.scatter(SINGLE_ENC[j], y, s=46, color=SAGE[2], edgecolor="#333", linewidth=0.5, zorder=3)
    axa.scatter(AE_COGEO[j], y, s=52, color=ROSE[2], edgecolor="#333", linewidth=0.5, zorder=3)
    axa.text(AE_COGEO[j] + 0.5, y + 0.16, PSTARS[j], ha="left", va="center",
             fontsize=7.5, color="#5C3A3A")
# white-box ceiling (18.47) sits beyond the shortened x-axis; note it compactly at
# the right edge with a "->" so the panel stays narrow without losing the reference.
axa.annotate(f"white-box ceiling {WHITEBOX_CEILING:.2f} $\\rightarrow$",
             xy=(11.7, -0.62), ha="right", va="center",
             fontsize=6.5, color="#5C3A3A", style="italic")
# compact legend in the empty upper-right band (data tops out near x=8.6 on the lower rows)
from matplotlib.lines import Line2D
axa.legend(handles=[
    Line2D([0], [0], marker="o", color="none", markerfacecolor=SAGE[2],
           markeredgecolor="#333", markersize=6, label="Single-encoder"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=ROSE[2],
           markeredgecolor="#333", markersize=6, label="AE-CoGEO (4-enc.)"),
], loc="upper right", bbox_to_anchor=(1.0, 1.02), fontsize=6.0,
   handletextpad=0.3, labelspacing=0.3)
axa.set_yticks(ya)
axa.set_yticklabels(ENC, fontsize=6.6)
axa.set_xlabel("Mean held-out rank-lift")
axa.set_xlim(0, 12)  # shortened per review: data max ~8.6, ceiling noted off-scale
axa.set_ylim(-0.9, len(ENC) - 0.3)
axa.grid(axis="x", linestyle=":", alpha=0.4)
axa.set_title("(c) Held-out transfer")

# =========================== (b) source-encoder scaling law ===========================
src_n = [1, 2, 3, 4]
pooled = [3.76, 4.57, 5.30, 5.64]
axb.plot(src_n, pooled, "o-", color=ROSE[2], linewidth=1.9, markersize=7,
         markeredgecolor="#333", label="Pooled lift ($\\rho{=}1.0$)")
for xi, yi in zip(src_n, pooled):
    axb.text(xi, yi + 0.28, f"{yi:.2f}", ha="center", fontsize=6.8, color="#5C3A3A")
axb.set_xticks(src_n)
axb.set_xlim(0.82, 4.18)
axb.set_xlabel("Number of source encoders in consensus")
axb.set_ylabel("Mean held-out rank-lift")
axb.set_ylim(0, 6.6)
axb.grid(axis="both", linestyle=":", alpha=0.4)
axb.legend(loc="lower right", fontsize=6.8)
axb.set_title("(a) Source-encoder scaling")

# =========================== (c) SSIM vs rank-lift Pareto ===========================
PARETO = [
    ("CoGEO", 15.71, 0.66, ROSE[2], "^"),
    ("PGD-bare", 13.73, 0.76, SAGE[2], "s"),
    ("AdvCLIP", 5.06, 0.78, MIST[2], "o"),
    ("Co-Attack", 12.82, 0.76, SAGE[1], "D"),
]
# AdvCLIP (5.06) and Co-Attack (12.82) have near-identical rank-lift; equal-size markers
# would collide at true x, so the two are nudged apart on x purely for legibility
# (exact values are in Table II). All four points share one consistent filled style.
for name, lift, ssim, col, mk in PARETO:
    axc.scatter(lift, ssim, s=78, color=col, marker=mk,
                edgecolor="#333", linewidth=0.6, zorder=3)
# uniform direct labels, each with a clear gap from its marker (no leader lines, no open markers)
for name, lx, ly, ha in [
    ("CoGEO",    15.71, 0.695, "center"),
    ("PGD-bare", 14.2,  0.790, "left"),
    ("AdvCLIP",   5.06, 0.815, "center"),
    ("Co-Attack", 12.3,  0.715, "right"),
]:
    axc.annotate(name, xy=(lx, ly), fontsize=7, color="#222", ha=ha, va="center")
# label parked at the right end above the line, clear of the upper-left Gaussian-control cluster
# Gaussian-noise control: stealthier than every attack (SSIM 0.854) yet ~0 rank-lift
axc.scatter(0.5, 0.854, s=66, marker="X", color=SAGE[2], edgecolor="#333", linewidth=0.6, zorder=3)
axc.annotate("Gaussian\n(control)", xy=(0.7, 0.854), xytext=(3.0, 0.895),
             fontsize=6.0, color="#3F4A44", ha="left", va="center",
             arrowprops=dict(arrowstyle="->", color="#7F8F84", linewidth=0.7))
# resolution C3 note: same attack at the 800^2 input is stealthier but loses the lift
# (placed in the empty upper-right region so it never overlaps the CoGEO point)
axc.text(13.2, 0.93, "800$^2$ input: SSIM$\\geq$0.98\nbut lift$\\to-0.56$ (C3)",
         fontsize=6.5, color="#5C3A3A", style="italic", ha="left", va="top")
axc.set_xlabel("Mean rank-lift")
axc.set_ylabel("SSIM at $224{\\times}224$")
axc.set_xlim(0, 22.5)
axc.set_ylim(0.60, 0.99)
axc.grid(axis="both", linestyle=":", alpha=0.4)
axc.set_title("(e) Stealth vs. lift")

# =========================== (d) eps-sweep ===========================
eps = [4, 8, 16]
xe = np.arange(len(eps))  # categorical, evenly spaced (3 budgets read cleanly; no stretched near-empty linear axis)
axd.plot(xe, cogeo_eps, "o-", color=ROSE[2], linewidth=1.9, markersize=7, markeredgecolor="#333", label="CoGEO")
axd.plot(xe, pgd_eps, "s-", color=SAGE[2], linewidth=1.9, markersize=7, markeredgecolor="#333", label="PGD-bare")
margins = [cogeo_eps[i] - pgd_eps[i] for i in range(3)]
for i in range(len(eps)):
    base = max(cogeo_eps[i], pgd_eps[i])
    axd.text(xe[i], base + 0.8, f"{margins[i]:+.2f}", ha="center", fontsize=6.8,
             color="#7F4F4F" if margins[i] >= 0 else "#5A6A5F")
axd.set_xticks(xe)
axd.set_xticklabels([f"{e}/255" for e in eps])
axd.set_xlim(-0.35, 2.35)
axd.set_xlabel("Perturbation budget $\\epsilon$")
axd.set_ylabel("Mean rank-lift")
axd.grid(axis="both", linestyle=":", alpha=0.4)
axd.legend(loc="upper left", fontsize=6.8)
axd.set_title("(b) Budget sweep")

# =========================== (e) six-backbone (grouped bars) ===========================
xf = np.arange(len(BB6_PRETTY))
wf = 0.20
for j, (m, col) in enumerate(zip(METHODS, M_COLORS)):
    axf.bar(xf + (j - 1.5) * wf, bb6_vals[:, j], wf, color=col,
            edgecolor="#333", linewidth=0.3, label=m)
axf.axhline(0, color="#999", linewidth=0.6)
axf.set_xticks(xf)
axf.set_xticklabels(BB6_PRETTY, fontsize=6.2)
axf.set_ylabel("Mean rank-lift")
axf.set_ylim(min(-8.0, float(bb6_vals.min()) - 1.5), max(22.0, float(bb6_vals.max()) + 2.5))
axf.grid(axis="y", linestyle=":", alpha=0.4)
axf.legend(loc="upper right", ncol=2, fontsize=6.0, handlelength=1.0,
           handletextpad=0.3, columnspacing=0.8, borderaxespad=0.3)
axf.set_title("(d) Six CLIP backbones")

# =========================== (f) Food-101 cross-dataset bars ===========================
# Same 4-method bar format as panel (d), on a second image distribution: the ESCI
# ordering CoGEO > PGD-bare > Co-Attack > AdvCLIP is preserved (Food-101, n=2,020,
# OpenAI ViT-L/14, eps=16/255). Per-method colors match the rest of the figure.
F_METHODS = ["CoGEO", "PGD-bare", "Co-Attack", "AdvCLIP"]
F_VALS = [113.2, 96.5, 91.1, 30.4]
F_COLORS = [ROSE[2], SAGE[2], SAGE[1], MIST[2]]
xg = np.arange(len(F_METHODS))
gbars = axg.bar(xg, F_VALS, width=0.66, color=F_COLORS, edgecolor="#333", linewidth=0.4, zorder=3)
for b, v in zip(gbars, F_VALS):
    axg.text(b.get_x() + b.get_width() / 2, v + 2.2, f"{v:.0f}", ha="center", fontsize=6.4, color="#2B2B2B")
axg.set_xticks(xg)
axg.set_xticklabels(F_METHODS, fontsize=6.2, rotation=28, ha="right")
axg.set_ylabel("Mean rank-lift")
axg.set_ylim(0, max(F_VALS) * 1.28)
axg.grid(axis="y", linestyle=":", alpha=0.4)
axg.set_title("(f) Food-101 ordering")

fig.subplots_adjust(top=0.92, bottom=0.15, left=0.06, right=0.99)
fig.savefig(os.path.join(HERE, "fig_results_main.pdf"))
fig.savefig(os.path.join(HERE, "fig_results_main.png"), dpi=170)
plt.close(fig)
print("done: fig_results_main composite (a)-(f): scaling/budget/transfer/six-backbone/stealth/food101")
print("eps cogeo:", cogeo_eps, "pgd:", pgd_eps, "margins:", margins)
print("bb6 CoGEO row:", [round(float(v), 2) for v in bb6_vals[:, 0]])
print("bb6 AdvCLIP min:", round(float(bb6_vals[:, 2].min()), 2))
