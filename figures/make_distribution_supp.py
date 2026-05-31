"""Distribution-of-rank-lift figure (supplement / extended version).

Per-pair raw rank-lift arrays live on the remote run host, so the genuine
unit-level distribution cannot be redrawn here; instead this figure shows the
distribution of the aggregate measured rank-lift across the experimental
configurations that ARE available locally:

(a) CoGEO rank-lift across the 30 anchor x backbone configurations (5 VLM
    caption anchors x 6 CLIP retrieval backbones; source: n3_esci_full_d_long.csv,
    matches Fig. anchor x backbone heatmap cell-for-cell). Shows the heterogeneity
    that the per-backbone column means already summarise.
(b) Four-way rank-lift distribution across the 6 CLIP backbones, per method
    (source: n3_esci_full_f_long.csv, matches Fig. results-main six-backbone
    panel). Shows CoGEO's distribution sits highest and AdvCLIP can go negative.

Same Morandi palette as the rest of the figure family.
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.6,
    "axes.labelsize": 8, "axes.linewidth": 0.8, "axes.edgecolor": "#444",
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": "#444", "ytick.color": "#444",
    "figure.facecolor": "white", "axes.facecolor": "white",
})
MIST = ["#F3EEE8", "#D8D1C7", "#8A9199"]
SAGE = ["#E7E1D6", "#B7A99A", "#7F8F84"]
ROSE = ["#F2E9E6", "#D8C3BC", "#B88C8C"]

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.normpath(os.path.join(HERE, "..", ".."))


def load_long(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


# ---------------- (a) CoGEO rank-lift across 30 anchor x backbone configs -------
d_rows = load_long(os.path.join(PAPER, "n3_esci_full_d_long.csv"))
cog_cells = [float(r["value"]) for r in d_rows if r["metric"] == "rank_lift_mean"]

# ---------------- (b) four methods across all 6 CLIP backbones ------------------
# Reconstruct the full 6-backbone x 4-method rank-lift matrix exactly as the
# main-body six-backbone panel does (OpenAI headline + E1 LAION + E5 EVA-02 +
# aggregate SigLIP/ViT-B32/ViT-H14), so the spread matches the paper cell-for-cell.
import json
METHODS = ["cogeo", "pgd_bare", "advclip", "coattack"]
PRETTY = {"cogeo": "CoGEO", "pgd_bare": "PGD-bare", "advclip": "AdvCLIP", "coattack": "Co-Attack"}
MCOL = {"cogeo": ROSE[2], "pgd_bare": SAGE[2], "advclip": MIST[2], "coattack": SAGE[1]}
RUNS = os.path.join(PAPER, "remote_runs")


def _cell(name):
    with open(os.path.join(RUNS, name, "cell_summary.json")) as f:
        return json.load(f)


_AGG = json.load(open(os.path.join(PAPER, "n3_esci_full_aggregate.json")))
_e1 = _cell("E1_laion2b_eps16")
_e5 = _cell("E5_eva02_eps16")
_main_rl = {"cogeo": 19.86, "pgd_bare": 12.37, "advclip": 6.45, "coattack": 5.77}
by_method = {m: [] for m in METHODS}
for m in METHODS:
    by_method[m].append(_main_rl[m])
    by_method[m].append(_e1["methods"][m]["rank_summary"]["rank_lift_mean"])
    by_method[m].append(_e5["methods"][m]["rank_summary"]["rank_lift_mean"])
    by_method[m].append(_AGG["f_panel"]["siglip"][m]["rank_lift_mean"])
    by_method[m].append(_AGG["f_panel"]["vitb32"][m]["rank_lift_mean"])
    by_method[m].append(_AGG["f_panel"]["vith14"][m]["rank_lift_mean"])

fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 2.7))

# ---- panel (a): histogram of the 30 CoGEO config means ----
arr = np.array(cog_cells)
axa.hist(arr, bins=np.arange(0, 20.1, 2.0), color=MIST[1], edgecolor="#5C3A3A",
         linewidth=0.7, zorder=2)
axa.axvline(arr.mean(), color=ROSE[2], linewidth=1.8, zorder=3,
            label=f"mean {arr.mean():.1f}")
axa.axvline(np.median(arr), color=SAGE[2], linewidth=1.6, linestyle="--", zorder=3,
            label=f"median {np.median(arr):.1f}")
# rug of individual configs
for v in arr:
    axa.plot([v, v], [-0.45, -0.05], color="#5C3A3A", linewidth=0.6, alpha=0.7,
             zorder=1, clip_on=False)
axa.set_xlim(0, 20)
axa.set_ylim(0, max(np.histogram(arr, bins=np.arange(0, 20.1, 2.0))[0]) + 1.2)
axa.set_xlabel("CoGEO mean rank-lift")
axa.set_ylabel("Number of anchor$\\times$backbone configs")
axa.grid(axis="y", linestyle=":", alpha=0.4)
axa.legend(loc="upper right", fontsize=6.6, handlelength=1.3)
axa.set_title(f"(a) CoGEO rank-lift spread ($n{{=}}{len(arr)}$ configs)")

# ---- panel (b): per-method strip + box across the 6 backbones ----
xs = np.arange(len(METHODS))
for i, m in enumerate(METHODS):
    vals = np.array(by_method[m])
    axb.boxplot([vals], positions=[i], widths=0.5, showfliers=False,
                medianprops=dict(color="#333", linewidth=1.2),
                boxprops=dict(color="#777", linewidth=0.9),
                whiskerprops=dict(color="#777", linewidth=0.9),
                capprops=dict(color="#777", linewidth=0.9))
    jit = (np.random.default_rng(i + 1).random(len(vals)) - 0.5) * 0.22
    axb.scatter(np.full(len(vals), i) + jit, vals, s=26, color=MCOL[m],
                edgecolor="#333", linewidth=0.4, zorder=3)
axb.axhline(0, color="#999", linewidth=0.6)
axb.set_xticks(xs)
axb.set_xticklabels([PRETTY[m] for m in METHODS], fontsize=6.8)
axb.set_ylabel("Mean rank-lift (per backbone)")
axb.grid(axis="y", linestyle=":", alpha=0.4)
axb.set_title("(b) Four-way spread across backbones")

fig.subplots_adjust(top=0.90, bottom=0.17, left=0.085, right=0.985, wspace=0.30)
fig.savefig(os.path.join(HERE, "fig_distribution.pdf"))
fig.savefig(os.path.join(HERE, "fig_distribution.png"), dpi=190)
plt.close(fig)
print("done: fig_distribution")
print("(a) cogeo 30-config: n=%d mean=%.2f median=%.2f min=%.2f max=%.2f"
      % (len(arr), arr.mean(), np.median(arr), arr.min(), arr.max()))
for m in METHODS:
    v = np.array(by_method[m])
    print("(b) %-9s n=%d vals=%s" % (m, len(v), [round(x, 2) for x in v]))
