#!/usr/bin/env python3
"""make_distribution_figure.py — beautiful standalone per-pair rank-lift distribution.

Two panels, real per-pair data (results/L7_5way/rank_<fam>_per_pair.csv, n=500 each,
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
    "font.family": "DejaVu Sans",
    "font.size": 7.0,
    "axes.edgecolor": "#5b5b5b",
    "axes.linewidth": 0.7,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "mathtext.fontset": "dejavusans",
})
MIST, SAGE, CLAY, STONE = "#F3EEE8", "#7F8F84", "#B7A99A", "#8A9199"
ROSE, DROSE, DSTONE = "#B88C8C", "#9B5F5C", "#5b5b5b"
PALE = "#C9BBA8"

HERE = Path(__file__).resolve().parent
DATA = HERE / ".." / "results" / "L7_5way"

# ordered by overall mean rank-lift (CoGEO strongest on top of the ridgeline)
# Five clearly distinct (but still desaturated) hues so the families separate by
# COLOUR alone — red / green / blue / amber / mauve.
C_COGEO, C_PGD, C_DIFF, C_COAT, C_ADV = "#B0413E", "#4F7A5B", "#3F6493", "#C8893A", "#7D6699"
FAMS = [
    ("cogeo",     "CoGEO",      C_COGEO),
    ("pgd_bare",  "PGD-bare",        C_PGD),
    ("diffprior", "Diffusion-prior", C_DIFF),
    ("coattack",  "Co-Attack",  C_COAT),
    ("advclip",   "AdvCLIP",    C_ADV),
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

    # Single-column figure: two panels side by side WITHIN one column width.
    # Splitting the column between the two panels makes each narrow-and-tall, and
    # the whole figure occupies only one column so body text keeps flowing in the
    # other column -- this is what actually saves vertical space (vs. a full-width
    # figure* or a stacked two-row layout).
    fig = plt.figure(figsize=(3.5, 2.36))
    # extra bottom room reserved for a single shared figure-level legend strip
    # (two rows, 4+3) placed UNDER both panels, so neither panel carries an
    # in-plot legend.
    # (a) no longer needs a wide left margin for family names, so left shrinks and
    # the freed width goes to panel (b) (wider width-ratio + a touch more wspace
    # for its y-axis), letting (b)'s x-axis run a little longer.
    gs = fig.add_gridspec(1, 2, width_ratios=[1.04, 1.0], wspace=0.30,
                          left=0.055, right=0.985, top=0.915, bottom=0.35)

    # ============================ (a) RIDGELINE ============================
    axA = fig.add_subplot(gs[0, 0])
    XLO, XHI = -18, 48           # tighter window so the mean dots (5--15.6) fan
    grid = np.linspace(XLO, XHI, 600)   # apart; the long tail is annotated, not shown
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
        axA.fill_between(grid, base, base + dens, color=col, alpha=1.0,
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
        # (family names are no longer drawn at the left margin -- the shared
        #  colour key below both panels identifies every family, so per-row
        #  labels would be redundant; this also frees left margin for panel (b).)

    # no-effect reference drawn ON TOP of the ridges (high zorder) so it stays
    # visible across every fill instead of being buried under them
    axA.axvline(0, color="#333333", lw=0.9, ls=(0, (4, 3)), alpha=0.75, zorder=250)
    axA.text(0, n * spacing + 0.50, "no effect", fontsize=6.2, color="#333333",
             ha="center", va="bottom", style="italic", zorder=251)
    # heavy-tail annotation (data-derived max). For x>=45 the CoGEO row tops out
    # at ~4.05 while ylim reaches 5.95, so the upper-right corner (vacated by
    # moving the mean/median legend to lower-right) is clean whitespace above the
    # CoGEO tail -- place the two-line label there, arrow down to the long tail.
    # label sits high-centre; a clear horizontal-ish stem runs from just below the
    # label rightward to the tip of the CoGEO right tail, landing just ABOVE the
    # ridge outline (~4.15) so the head never sits inside the red fill.
    axA.text(XHI - 19, 5.55, "heavy right tail\n(max $\\approx$ {:.0f})".format(gmax),
             fontsize=6.0, color=C_COGEO, ha="center", va="top", zorder=260)
    axA.annotate("", xy=(XHI - 0.5, 4.18), xytext=(XHI - 17, 4.78),
                 arrowprops=dict(arrowstyle="->", color=C_COGEO, lw=0.9,
                                 shrinkA=1, shrinkB=1,
                                 connectionstyle="arc3,rad=0.0"), zorder=260)
    axA.set_xlim(XLO, XHI)
    axA.set_xticks([0, 20, 40])
    axA.set_ylim(-0.30, n * spacing + 0.95)
    axA.set_yticks([])
    axA.spines["left"].set_visible(False)
    axA.set_xlabel("Per-pair rank lift")
    axA.set_title("(a) Per-pair rank-lift density", fontsize=7.4,
                  loc="left", x=0.0, color="#000000", pad=5)
    # mean/median marker handles are NOT drawn as an in-panel legend; they go into
    # the shared figure-level legend strip below both panels (built at the end).
    from matplotlib.lines import Line2D
    # median proxy is a VERTICAL bar (marker '|'), matching the short vertical
    # median ticks drawn on each ridge in (a) -- not a horizontal line.
    marker_handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=4.6,
               markerfacecolor=STONE, markeredgecolor="#333333",
               markeredgewidth=1.0, label="mean"),
        Line2D([], [], marker="|", linestyle="none", markersize=7,
               markeredgecolor="#333333", markeredgewidth=1.4, label="median"),
    ]

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
        lw = 1.45 if name == "CoGEO" else 0.95
        z = 100 if name == "CoGEO" else 10 + len(name)
        axB.step(xs, ys, where="post", color=col, lw=lw, label=name,
                 alpha=1.0, zorder=z, solid_capstyle="round",
                 solid_joinstyle="round")
    axB.axvline(0, color=DSTONE, lw=0.8, ls=(0, (4, 3)), alpha=0.6, zorder=1)
    axB.set_xscale("symlog", linthresh=2, linscale=0.85)
    axB.set_xlim(-25, 430)
    axB.set_ylim(0, 1.001)
    axB.set_xticks([-10, 0, 10, 100])
    axB.set_xticklabels(["$-$10", "0", "10", "100"])
    axB.tick_params(axis="x", which="minor", length=2, color="#9a9a9a")
    axB.set_xlabel("Rank lift (symlog)")
    axB.set_ylabel("Cumulative fraction of pairs")
    axB.set_title("(b) Empirical CDF", fontsize=7.4,
                  loc="left", color="#000000", pad=5)
    # ===== shared figure-level legend strip UNDER both panels =====
    # Neither panel carries an in-plot legend: the five family colours (shared by
    # the (a) ridgelines and the (b) CDF curves) plus the mean/median markers are
    # collected into one horizontal key centred below the two panels, so it never
    # overlaps any ridge peak or CDF curve. Family swatches use proxy patches in
    # the FAMS order; mean/median marker proxies are appended.
    from matplotlib.patches import Patch
    fam_handles = [Patch(facecolor=col, edgecolor="#333333", linewidth=0.5,
                         label=name) for name, _, col in data]
    # Bottom legend, two stacked rows matched to what each key describes:
    #  - upper row, centred under panel (a): the mean/median markers (an (a)-only
    #    encoding -- the CDF in (b) has no such markers);
    #  - lower row, centred under BOTH panels: the five family colours, shared by
    #    the (a) ridgelines and the (b) CDF curves.
    leg_mm = fig.legend(handles=marker_handles, loc="lower center",
                        bbox_to_anchor=(0.33, 0.085), ncol=2, frameon=False,
                        fontsize=5.8, handlelength=1.0, handletextpad=0.35,
                        columnspacing=1.1, borderaxespad=0.0)
    fig.add_artist(leg_mm)
    fig.legend(handles=fam_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=5, frameon=False,
               fontsize=5.8, handlelength=0.95, handletextpad=0.35,
               columnspacing=0.9, borderaxespad=0.0)

    out = HERE / "fig_distribution.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print("wrote", out, "and", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
