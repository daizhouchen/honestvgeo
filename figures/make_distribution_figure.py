#!/usr/bin/env python3
"""make_distribution_figure.py — beautiful standalone per-pair rank-lift distribution.

Two panels, real per-pair data (remote_pull/L7_5way/rank_<fam>_per_pair.csv, n=500 each,
columns query_id,cohort,rank_lift):
  (a) Ridgeline (joyplot) of the five attack families' rank-lift KDE — shape / heavy tail.
  (b) Empirical CDF of the same five families — magnitude, reads off tail mass.

Morandi palette, matched to the rest of the figure set. Dependency-free KDE (manual
Gaussian kernel) so it renders anywhere numpy+matplotlib exist.
"""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- repo Morandi style (matches make_bounded_defense.py etc.) --------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8.5,
    "axes.edgecolor": "#5b5b5b",
    "axes.linewidth": 0.7,
    "savefig.dpi": 300,
    "font.serif": ["DejaVu Serif"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "mathtext.fontset": "dejavuserif",
})
MIST, SAGE, CLAY, STONE = "#F3EEE8", "#7F8F84", "#B7A99A", "#8A9199"
ROSE, DROSE, DSTONE = "#B88C8C", "#9B5F5C", "#5b5b5b"
PALE = "#C9BBA8"

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "remote_pull" / "L7_5way"

# ordered by overall mean rank-lift (CoGEO strongest on top of the ridgeline)
# Five clearly distinct (but still desaturated) hues so the families separate by
# COLOUR alone — red / green / blue / amber / mauve.
C_COGEO, C_PGD, C_DIFF, C_COAT, C_ADV = "#B0413E", "#4F7A5B", "#3F6493", "#C8893A", "#7D6699"
FAMS = [
    ("cogeo",     "CoGEO",            C_COGEO),
    ("pgd_bare",  "PGD",              C_PGD),
    ("diffprior", "Diffusion-prior",  C_DIFF),
    ("coattack",  "Co-Attack",        C_COAT),
    ("advclip",   "AdvCLIP",          C_ADV),
]


def load(fam):
    rows = list(csv.DictReader(open(DATA / f"rank_{fam}_per_pair.csv")))
    return np.array([float(r["rank_lift"]) for r in rows])


def gauss_kde(x, grid, bw):
    """Manual Gaussian KDE on `grid`; dependency-free."""
    x = x[:, None]
    g = grid[None, :]
    k = np.exp(-0.5 * ((g - x) / bw) ** 2) / (bw * np.sqrt(2 * np.pi))
    return k.mean(axis=0)


def main():
    data = [(name, load(fam), col) for fam, name, col in FAMS]

    # data-derived annotation values (kept honest: computed, never hard-coded)
    gmax = max(v.max() for _, v, _ in data)
    min_pct_le0 = min(100.0 * np.mean(v <= 0) for _, v, _ in data)
    print("=== per-family per-pair rank-lift (verification) ===")
    for name, v, _ in data:
        print(f"{name:16s} n={len(v):3d} mean={v.mean():6.2f} median={np.median(v):5.1f}"
              f" min={v.min():5.0f} max={v.max():5.0f} %<=0={100*np.mean(v<=0):4.1f}"
              f" %>=20={100*np.mean(v>=20):4.1f}")
    print(f"global max={gmax:.0f}  min %<=0 across families={min_pct_le0:.1f}")

    fig = plt.figure(figsize=(7.4, 3.7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.32, 1.0], wspace=0.32,
                          left=0.135, right=0.985, top=0.88, bottom=0.135)

    # ============================ (a) RIDGELINE ============================
    axA = fig.add_subplot(gs[0, 0])
    XLO, XHI = -22, 90           # readable window; tail annotated separately
    grid = np.linspace(XLO, XHI, 600)
    n = len(data)
    spacing = 1.0
    height = 0.92 * spacing      # ridges just clear of each other (no overlap)
    lab_tf = axA.get_yaxis_transform()  # x in axes-fraction, y in data coords

    for i, (name, vals, col) in enumerate(data):
        base = (n - 1 - i) * spacing            # CoGEO (i=0) at the top
        # wider bandwidth -> soft waves instead of a spike at 0
        bw = max(5.5, 1.06 * np.std(vals) * len(vals) ** (-1 / 5))
        dens = gauss_kde(vals, grid, bw)
        dens = dens / dens.max() * height       # normalise shape per family
        axA.fill_between(grid, base, base + dens, color=col, alpha=0.93,
                         linewidth=0.0, zorder=i * 2)
        axA.plot(grid, base + dens, color=DSTONE, linewidth=0.7, zorder=i * 2 + 1)
        # median tick + mean dot. Rows no longer overlap, so both markers sit on
        # THIS row's own baseline and can't be mistaken for the ridge below.
        # median = short dark tick inside the ridge near 0; mean = filled dot at
        # its (far-right) value, dark-edged so it reads against the fill.
        med, mean = np.median(vals), np.mean(vals)
        axA.plot([med, med], [base + 0.01, base + 0.40], color="#333333", lw=1.4,
                 zorder=99, solid_capstyle="round")
        axA.scatter([mean], [base + 0.09], marker="o", s=34, color=col,
                    edgecolor="#333333", linewidth=1.0, zorder=100, clip_on=False)
        # family label in the left margin (classic ridgeline)
        axA.text(-0.012, base + 0.12, name, transform=lab_tf, fontsize=8.6,
                 color=DSTONE, fontweight="bold", va="center", ha="right",
                 zorder=101, clip_on=False)

    # no-effect reference drawn ON TOP of the ridges (high zorder) so it stays
    # visible across every fill instead of being buried under them
    axA.axvline(0, color="#333333", lw=0.9, ls=(0, (4, 3)), alpha=0.75, zorder=250)
    axA.text(0, n * spacing + 0.50, "no effect", fontsize=7.4, color="#333333",
             ha="center", va="bottom", style="italic", zorder=251)
    # heavy-tail annotation (data-derived max)
    axA.annotate(f"heavy right tail\n(max $\\approx$ {gmax:.0f})", xy=(XHI - 3, 0.42),
                 xytext=(XHI - 34, 2.55), fontsize=7.2, color=C_COGEO, ha="left",
                 arrowprops=dict(arrowstyle="->", color=C_COGEO, lw=0.9,
                                 connectionstyle="arc3,rad=-0.25"))
    axA.set_xlim(XLO, XHI)
    axA.set_ylim(-0.30, n * spacing + 0.95)
    axA.set_yticks([])
    axA.spines["left"].set_visible(False)
    axA.set_xlabel("Per-pair rank lift")
    axA.set_title("(a) Distribution of per-pair rank lift", fontsize=9.2,
                  loc="left", color=DSTONE, pad=6)
    # legend for the markers
    axA.scatter([], [], marker="o", s=34, color=STONE, edgecolor="#333333",
                linewidth=1.0, label="mean")
    axA.plot([], [], color="#333333", lw=1.4, label="median")
    axA.legend(loc="upper right", frameon=False, fontsize=7, handletextpad=0.4,
               borderaxespad=0.2)

    # ============================== (b) ECDF ==============================
    axB = fig.add_subplot(gs[0, 1])
    # The data is heavy-tailed (<0 .. ~390) but every bit of curve separation
    # lives in the 0-30 band; on a linear axis the five families collapse into
    # one bundle near 0. A symlog x-axis keeps a small linear window around 0
    # (so the "<=0 unaffected" mass is honest) and log-compresses the long tail,
    # letting the five families fan apart where they actually differ.
    # Clean plain step ECDF (the earlier uncluttered style); the five families
    # are separated purely by clearly distinct hues now, no markers/dash tricks.
    for name, vals, col in data:
        xs = np.sort(vals)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        lw = 2.2 if name == "CoGEO" else 1.5
        z = 100 if name == "CoGEO" else 10 + len(name)
        axB.step(xs, ys, where="post", color=col, lw=lw, label=name,
                 alpha=0.97, zorder=z, solid_capstyle="round",
                 solid_joinstyle="round")
    axB.axvline(0, color=DSTONE, lw=0.8, ls=(0, (4, 3)), alpha=0.6, zorder=1)
    axB.set_xscale("symlog", linthresh=2, linscale=0.85)
    axB.set_xlim(-25, 430)
    axB.set_ylim(0, 1.001)
    axB.set_xticks([-10, 0, 10, 100])
    axB.set_xticklabels(["$-$10", "0", "10", "100"])
    axB.tick_params(axis="x", which="minor", length=2, color="#9a9a9a")
    axB.set_xlabel("Per-pair rank lift  (symlog scale)")
    axB.set_ylabel("Cumulative fraction of pairs")
    axB.set_title("(b) Empirical CDF of rank lift", fontsize=9.2,
                  loc="left", color=DSTONE, pad=6)
    axB.legend(loc="lower right", frameon=True, framealpha=0.9,
               edgecolor="#cccccc", fontsize=7.2, handlelength=1.8,
               labelspacing=0.34, borderaxespad=0.3)
    # annotate the median pile-up near zero (data-derived floor); placed in the
    # empty upper-left so it never collides with the lower-right legend
    axB.annotate(f"$\\geq$ {min_pct_le0:.0f}% of pairs\nunaffected ($\\leq 0$)",
                 xy=(0, 0.47), xytext=(-21, 0.74), fontsize=7.2, color=DSTONE,
                 ha="left", va="center", arrowprops=dict(arrowstyle="->",
                 color=DSTONE, lw=0.8, connectionstyle="arc3,rad=-0.3"))

    out = HERE / "fig_distribution.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print("wrote", out, "and", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
