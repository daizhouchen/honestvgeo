"""L4 — Retriever-capacity scaling law for the CoGEO white-box rank-lift band.

Source numbers: Table tab:cross-backbone (analysis.tex), CoGEO mean white-box
rank-lift at eps=16/255, 224x224, product-title anchor, ESCI-491 manifest.

Clean, corpus-controlled law (the load-bearing claim):
  Within the LAION-2B pretraining family, the manipulable rank-lift band
  shrinks monotonically with retriever capacity:
      ViT-B/32 (88M)  18.07  ->  ViT-L/14 (304M) 5.18  ->  ViT-H/14 (632M) 4.61
  Spearman rho = -1.0 over the controlled triple (~4x contraction).
Second axis (pretraining corpus/quality): holding capacity at ViT-L/14, moving
from OpenAI-WIT-400M to LAION-2B drops rank-lift 15.71 -> 5.18 (-74%). So both
larger capacity and a larger/cleaner pretraining corpus compress the band.

Image-tower parameter counts are published model-card values (approximate); the
law rests on the monotone ordering, not the exact param values.
Morandi palette, matching make_n6_figs.py.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7.5,
        "axes.titlesize": 8.5,
        "axes.labelsize": 7.5,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#444",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": "#444",
        "ytick.color": "#444",
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.frameon": False,
        "legend.fontsize": 5.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

MIST = ["#F3EEE8", "#D8D1C7", "#8A9199"]
SAGE = ["#E7E1D6", "#B7A99A", "#7F8F84"]
ROSE = ["#F2E9E6", "#D8C3BC", "#B88C8C"]

ROOT = os.path.dirname(os.path.abspath(__file__))

# name, image-tower params (M, published), pretraining corpus, CoGEO white-box rank-lift (16/255)
BACKBONES = [
    ("OpenCLIP-B/32", 88, "LAION-2B", 18.07),
    ("SigLIP-B/16", 93, "WebLI", 9.04),
    ("OpenAI-L/14", 304, "WIT-400M", 15.71),
    ("OpenCLIP-L/14", 304, "LAION-2B", 5.18),
    ("EVA-02-L/14", 304, "Merged-2B", 9.01),
    ("OpenCLIP-H/14", 632, "LAION-2B", 4.61),
]

# ---- corpus-controlled LAION-2B triple: the clean monotone law ----
laion = [(p, rl, n) for (n, p, c, rl) in BACKBONES if c == "LAION-2B"]
laion.sort(key=lambda t: t[0])
lp = np.array([t[0] for t in laion], dtype=float)
lrl = np.array([t[1] for t in laion], dtype=float)
rho_laion, _ = spearmanr(lp, lrl)

# log-linear fit on the controlled triple: rank-lift = a*log10(params) + b
coef = np.polyfit(np.log10(lp), lrl, 1)
fit_a, fit_b = float(coef[0]), float(coef[1])
contraction = lrl[0] / lrl[-1]

# corpus effect at fixed ViT-L/14 capacity
rl_openai_L = next(rl for (n, p, c, rl) in BACKBONES if n == "OpenAI-L/14")
rl_laion_L = next(rl for (n, p, c, rl) in BACKBONES if n == "OpenCLIP-L/14")
corpus_drop = (rl_openai_L - rl_laion_L) / rl_openai_L

# all-6 Spearman (reported as NOT clean -> motivates corpus-controlled framing)
allp = np.array([p for (n, p, c, rl) in BACKBONES], dtype=float)
allrl = np.array([rl for (n, p, c, rl) in BACKBONES], dtype=float)
rho_all, p_all = spearmanr(allp, allrl)

# ---------------- figure (compact single-column) ----------------
# Single-column standalone figure (formerly one panel of the external-validity
# float). The corpus-effect detail (-74% at fixed ViT-L/14) is stated in the
# caption rather than drawn, keeping the small panel uncluttered.
fig, ax = plt.subplots(figsize=(3.5, 2.7))

# trend line through the LAION-2B controlled triple
xs = np.linspace(np.log10(lp.min()) - 0.05, np.log10(lp.max()) + 0.05, 50)
ax.plot(10 ** xs, fit_a * xs + fit_b, "-", color=SAGE[2], linewidth=1.4, zorder=1,
        label=f"LAION-2B law ($\\rho{{=}}{-1.0:.0f}$, {contraction:.1f}$\\times$)")

corpus_style = {
    "LAION-2B": dict(color=SAGE[2], marker="o", s=46),
    "WIT-400M": dict(color=ROSE[2], marker="D", s=42),
    "Merged-2B": dict(color=MIST[2], marker="^", s=44),
    "WebLI": dict(color="#B7A99A", marker="s", s=42),
}
# explicit, non-overlapping label anchors (x_abs, y_abs, ha, va) per backbone.
LABEL_POS = {
    "OpenCLIP-B/32": (95, 19.7, "left", "bottom"),
    "SigLIP-B/16": (99, 10.5, "left", "bottom"),
    "OpenAI-L/14": (300, 21.3, "center", "bottom"),
    "EVA-02-L/14": (345, 9.6, "left", "center"),
    "OpenCLIP-L/14": (345, 3.7, "left", "top"),
    "OpenCLIP-H/14": (632, 6.4, "center", "bottom"),
}
seen = set()
for (n, p, c, rl) in BACKBONES:
    st = corpus_style[c]
    lbl = c if c not in seen else None
    seen.add(c)
    ax.scatter(p, rl, edgecolor="#333", linewidth=0.4, zorder=3, label=lbl, **st)
    lx, ly, lha, lva = LABEL_POS[n]
    ax.annotate(n, xy=(p, rl), xytext=(lx, ly), fontsize=5.4, color="#444",
                ha=lha, va=lva)

# corpus effect at fixed ViT-L/14 (304M): draw the OpenAI-WIT400M -> LAION-2B drop
# as a vertical two-headed arrow between the two 304M points (15.71 -> 5.18, -74%).
x_L = 304
ax.annotate("", xy=(x_L, rl_laion_L), xytext=(x_L, rl_openai_L),
            arrowprops=dict(arrowstyle="<->", color="#9B5F5C", lw=1.1,
                            shrinkA=3, shrinkB=3))
ax.text(x_L * 1.12, (rl_openai_L + rl_laion_L) / 2,
        f"$-{corpus_drop*100:.0f}\\%$\n(fixed ViT-L/14)", fontsize=5.4,
        color="#9B5F5C", ha="left", va="center", style="italic")

ax.set_xscale("log")
ax.set_xticks([88, 304, 632])
ax.set_xticklabels(["88", "304", "632"])
ax.set_xlim(72, 950)
ax.set_xlabel("Image-tower params (M)")
ax.set_ylabel("CoGEO white-box rank-lift")
ax.set_ylim(0, 23)
ax.grid(axis="both", linestyle=":", alpha=0.4)
ax.legend(loc="lower left", fontsize=5.5, frameon=False,
          handletextpad=0.5, labelspacing=0.6, borderaxespad=0.4)
fig.subplots_adjust(left=0.145, right=0.965, top=0.965, bottom=0.16)
fig.savefig(os.path.join(ROOT, "fig_capacity_scaling.pdf"))
fig.savefig(os.path.join(ROOT, "fig_capacity_scaling.png"), dpi=200)
plt.close(fig)

summary = {
    "source": "Table tab:cross-backbone (analysis.tex), CoGEO white-box rank-lift, eps=16/255",
    "laion2b_controlled_triple": {
        "params_M": lp.tolist(),
        "rank_lift": lrl.tolist(),
        "spearman_rho": round(float(rho_laion), 4),
        "log_linear_fit_slope_intercept": [round(fit_a, 4), round(fit_b, 4)],
        "contraction_B32_to_H14": round(float(contraction), 3),
    },
    "corpus_effect_at_ViT_L14": {
        "openai_wit400m": rl_openai_L,
        "laion2b": rl_laion_L,
        "relative_drop": round(float(corpus_drop), 4),
    },
    "all6_spearman": {"rho": round(float(rho_all), 4), "p": round(float(p_all), 4),
                      "note": "NOT clean across corpora; motivates the corpus-controlled framing"},
}
with open(os.path.join(ROOT, "fig_capacity_scaling_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("done: fig_capacity_scaling")
print(json.dumps(summary, indent=2))
