"""Two previously prose-only results surfaced as one compact single-column figure
(information-dense, no new page budget beyond one small float).

(a) Five attack families on the OpenAI ViT-L/14 white-box surrogate (n=500,
    eps=16/255): a diffusion-prior generative family is added to the four headline
    families and lands mid-pack, below CoGEO.
    CoGEO 15.60 > PGD-bare 13.58 ~ Diffusion-prior 13.18 ~ Co-Attack 12.82 >> AdvCLIP 5.06

(b) Training-free Laplacian detector AUROC vs perturbation budget: near-perfect at
    eps=16/255 (uniform 0.99-1.00 across the four families and the AE-CoGEO transfer
    perturbation; 1.00 on a held-out Food-101 domain after clean-only recalibration),
    degrading as the budget shrinks (the regime where attack efficacy also collapses).
    eps=16 -> ~0.995, eps=8 -> 0.963, eps=4 -> 0.871

Same Morandi palette as the rest of the figure family.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.4,
    "axes.labelsize": 8, "axes.linewidth": 0.8, "axes.edgecolor": "#444",
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": "#444", "ytick.color": "#444",
    "figure.facecolor": "white", "axes.facecolor": "white",
})
MIST = ["#F3EEE8", "#D8D1C7", "#8A9199"]
SAGE = ["#E7E1D6", "#B7A99A", "#7F8F84"]
ROSE = ["#F2E9E6", "#D8C3BC", "#B88C8C"]
HERE = os.path.dirname(os.path.abspath(__file__))

fig, (axa, axb) = plt.subplots(2, 1, figsize=(3.45, 2.8))

# ---------- (a) five-family white-box bars ----------
fam = ["CoGEO", "PGD-bare", "Diffusion-\nprior", "Co-Attack", "AdvCLIP"]
val = [15.60, 13.58, 13.18, 12.82, 5.06]
col = [ROSE[2], SAGE[2], ROSE[1], SAGE[1], MIST[2]]
x = np.arange(len(fam))
bars = axa.bar(x, val, width=0.66, color=col, edgecolor="#333", linewidth=0.4, zorder=3)
# highlight the newly added diffusion family with a heavier edge
bars[2].set_edgecolor("#5C3A3A"); bars[2].set_linewidth(1.4)
for b, v in zip(bars, val):
    axa.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f}", ha="center", fontsize=6.8, color="#2B2B2B")
axa.set_xticks(x); axa.set_xticklabels(fam, fontsize=6.6)
axa.set_ylabel("Mean rank-lift")
axa.set_ylim(0, 18)
axa.grid(axis="y", linestyle=":", alpha=0.4)
axa.set_title("(a) Five attack families (white-box, $n{=}500$)")

# ---------- (b) detector AUROC vs budget ----------
eps = [4, 8, 16]
auroc = [0.871, 0.963, 0.995]
axb.plot(eps, auroc, "o-", color=ROSE[2], linewidth=1.9, markersize=7, markeredgecolor="#333", zorder=3)
for e, a in zip(eps, auroc):
    axb.text(e, a + 0.006, f"{a:.2f}", ha="center", fontsize=6.8, color="#5C3A3A")
axb.axhline(0.95, color=MIST[2], linestyle=":", linewidth=0.9)
axb.text(4.1, 0.955, "0.95", fontsize=6.0, color=MIST[2], va="bottom")
axb.set_xticks(eps); axb.set_xticklabels([f"{e}/255" for e in eps])
axb.set_xlabel("Perturbation budget $\\epsilon$")
axb.set_ylabel("Detector AUROC")
axb.set_ylim(0.82, 1.01)
axb.grid(axis="both", linestyle=":", alpha=0.4)
axb.text(0.5, 0.06, "@$\\epsilon{=}16$: 0.99$-$1.00 across 4 families + AE-transfer;\n1.00 on held-out Food-101 (clean-only recal.)",
         transform=axb.transAxes, ha="center", va="bottom", fontsize=5.7, color="#3F4A44", style="italic")
axb.set_title("(b) Training-free detector AUROC vs budget")

fig.subplots_adjust(top=0.93, bottom=0.115, left=0.16, right=0.97, hspace=0.55)
fig.savefig(os.path.join(HERE, "fig_stats_extra.pdf"))
fig.savefig(os.path.join(HERE, "fig_stats_extra.png"), dpi=190)
plt.close(fig)
print("done: fig_stats_extra (5-family + detector AUROC)")
