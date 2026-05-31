"""Main-body results figure: two panels on one full-width float.

Panel (a) Per-cohort rank-lift, four white-box methods x four ESCI cohorts
          (data == Table tab:percohort, 560-pair OpenAI ViT-L/14).
Panel (b) AE-CoGEO transfer: single-encoder vs four-encoder consensus on four
          held-out encoders, with the white-box ceiling drawn as a reference
          line (data == Table tab:transfer; ceiling 18.47).

All numbers are hardcoded to match the manuscript tables exactly so the figure
and tables can never drift. Morandi palette matches make_panels_v2.py.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
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

# ----------------------------------------------------------------------
# Panel (a) data -- per-cohort rank-lift (Table tab:percohort)
# ----------------------------------------------------------------------
COHORTS = ["E (Exact)", "S (Substitute)", "C (Complement)", "I (Irrelevant)"]
COHORT_N = [125, 125, 185, 125]
METHODS_4 = ["CoGEO", "PGD-bare", "AdvCLIP", "Co-Attack"]
COLORS_4 = [ROSE[2], SAGE[2], MIST[2], SAGE[1]]
VALS = np.array(
    [
        [3.90, 5.24, 36.77, 25.43],  # CoGEO
        [7.12, 1.84, 18.86, 18.53],  # PGD-bare
        [1.00, 0.94, 14.46, 5.53],   # AdvCLIP
        [6.18, 5.47, -1.86, 16.94],  # Co-Attack
    ]
)

# ----------------------------------------------------------------------
# Panel (b) data -- AE-CoGEO held-out transfer (Table tab:transfer)
# ----------------------------------------------------------------------
ENCODERS = [
    "OpenCLIP-L/14\n(LAION-2B)",
    "OpenCLIP-H/14\n(LAION-2B)",
    "OpenCLIP-B/32\n(LAION-2B)",
    "SigLIP\nViT-B/16",
]
SINGLE_ENC = [1.04, 3.63, 7.28, 3.07]
AE_COGEO = [3.29, 5.85, 8.60, 4.80]
PSTARS = ["***", "***", "*", "**"]  # 7e-5, <1e-3, 0.016, 1.3e-3
WHITEBOX_CEILING = 18.47

fig, (axa, axb) = plt.subplots(1, 2, figsize=(13.6, 4.3))

# ===================== Panel (a) =====================
x = np.arange(len(COHORTS))
w = 0.20
for i, (m, c) in enumerate(zip(METHODS_4, COLORS_4)):
    axa.bar(x + (i - 1.5) * w, VALS[i, :], w, color=c, edgecolor="#333", linewidth=0.4, label=m)
    for j, v in enumerate(VALS[i, :]):
        if v >= 0:
            axa.text(x[j] + (i - 1.5) * w, v + 0.7, f"{v:+.1f}", ha="center", fontsize=7, color="#333")
        else:
            axa.text(x[j] + (i - 1.5) * w, v - 2.4, f"{v:+.1f}", ha="center", fontsize=7, color="#7F4F4F")
axa.axhline(0, color="#999", linewidth=0.6)
axa.set_xticks(x)
axa.set_xticklabels([f"{c}\n$n={n}$" for c, n in zip(COHORTS, COHORT_N)], fontsize=8.5)
axa.set_ylabel("Mean rank-lift")
axa.set_ylim(-5.5, 45)
axa.grid(axis="y", linestyle=":", alpha=0.4)
axa.legend(ncol=2, loc="upper left", frameon=False, columnspacing=1.4, handletextpad=0.5, fontsize=8.5)
axa.set_title("(a) Per-cohort rank-lift (560-pair ESCI, OpenAI ViT-L/14)", pad=8)

# ===================== Panel (b) =====================
xb = np.arange(len(ENCODERS))
wb = 0.34
b1 = axb.bar(xb - wb / 2, SINGLE_ENC, wb, color=SAGE[2], edgecolor="#333", linewidth=0.4,
             label="Single-encoder transfer")
b2 = axb.bar(xb + wb / 2, AE_COGEO, wb, color=ROSE[2], edgecolor="#333", linewidth=0.4,
             label="AE-CoGEO (4-encoder consensus)")
for j in range(len(ENCODERS)):
    axb.text(xb[j] - wb / 2, SINGLE_ENC[j] + 0.25, f"{SINGLE_ENC[j]:.2f}", ha="center", fontsize=7.5, color="#333")
    axb.text(xb[j] + wb / 2, AE_COGEO[j] + 0.25, f"{AE_COGEO[j]:.2f}", ha="center", fontsize=7.5, color="#5C3A3A")
    # significance bracket over the AE-CoGEO bar
    ytop = max(SINGLE_ENC[j], AE_COGEO[j]) + 1.2
    axb.text(xb[j], ytop, PSTARS[j], ha="center", fontsize=10, color="#5C3A3A")
# white-box ceiling reference
axb.axhline(WHITEBOX_CEILING, color="#8A6A6A", linewidth=1.1, linestyle="--")
axb.text(len(ENCODERS) - 1 + 0.05, WHITEBOX_CEILING - 1.4,
         f"white-box ceiling = {WHITEBOX_CEILING:.2f}", ha="right", fontsize=8.5,
         color="#5C3A3A", style="italic")
axb.set_xticks(xb)
axb.set_xticklabels(ENCODERS, fontsize=8.5)
axb.set_ylabel("Mean held-out rank-lift")
axb.set_ylim(0, 20.5)
axb.grid(axis="y", linestyle=":", alpha=0.4)
# place legend in the empty band below the white-box ceiling line (18.47) so the
# dashed reference line never cuts through the legend text
axb.legend(loc="upper left", bbox_to_anchor=(0.0, 0.80), frameon=False, fontsize=8.5)
axb.set_title("(b) AE-CoGEO transfer to held-out encoders ($n=498$, $\\epsilon=16/255$)", pad=8)

fig.subplots_adjust(top=0.90, bottom=0.16, left=0.06, right=0.99, wspace=0.18)
fig.savefig(os.path.join(HERE, "fig_results_main_OLD2panel_DEPRECATED.pdf"))
fig.savefig(os.path.join(HERE, "fig_results_main_OLD2panel_DEPRECATED.png"), dpi=160)
plt.close(fig)
print("done: fig_results_main (panel a per-cohort, panel b AE-CoGEO transfer)")
print(f"  transfer single-enc {SINGLE_ENC} -> AE-CoGEO {AE_COGEO}; ceiling {WHITEBOX_CEILING}")
