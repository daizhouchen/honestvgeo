"""Build the four publication figures for the ICMR 2026 anchor-decoupled paper.

Figures (vector PDF + PNG preview):
  1. fig_framework.pdf          — anchor-perp-query system diagram (full text width)
  2. fig_percohort_bars.pdf     — per-cohort grouped rank-lift bars (full text width)
  3. fig_wilcoxon_heatmap.pdf   — 4x4 paired Wilcoxon p-value heatmap (single column)
  4. fig_ssim_pareto_v2.pdf     — stealth-vs-lift Pareto with frontier (single column)

Source data: experiments/main/n3_esci/runs/n3_h2_summary.json
Palette: Morandi (mist-stone primary + sage-clay contrast).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "experiments" / "main" / "n3_esci" / "runs" / "n3_h2_summary.json"
FIGDIR = Path(__file__).parent

PALETTE = {
    "mist-1": "#F3EEE8",
    "mist-2": "#D8D1C7",
    "mist-3": "#8A9199",
    "sage-1": "#E7E1D6",
    "sage-2": "#B7A99A",
    "sage-3": "#7F8F84",
    "rose-1": "#F2E9E6",
    "rose-2": "#D8C3BC",
    "rose-3": "#B88C8C",
}

METHOD_STYLE = {
    "advclip":  {"color": PALETTE["mist-3"], "marker": "o", "label": "AdvCLIP"},
    "pgd_bare": {"color": PALETTE["sage-3"], "marker": "s", "label": "PGD-bare"},
    "coattack": {"color": PALETTE["sage-2"], "marker": "D", "label": "Co-Attack"},
    "cogeo":    {"color": PALETTE["rose-3"], "marker": "^", "label": "CoGEO"},
}
METHOD_ORDER = ["cogeo", "pgd_bare", "coattack", "advclip"]
COHORTS = ["E", "S", "C", "I"]
COHORT_FULLNAME = {"E": "Exact", "S": "Substitute",
                   "C": "Complement", "I": "Irrelevant"}

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.edgecolor":     PALETTE["mist-3"],
    "axes.linewidth":     0.8,
    "axes.labelcolor":    "#333333",
    "xtick.color":        "#444444",
    "ytick.color":        "#444444",
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "grid.color":         PALETTE["mist-2"],
    "grid.linestyle":     "--",
    "grid.linewidth":     0.5,
    "grid.alpha":         0.7,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.06,
    "patch.linewidth":    0.8,
    "legend.frameon":     False,
})


def _load_data() -> dict:
    return json.loads(SRC.read_text())


def _save(fig: plt.Figure, name: str) -> None:
    pdf = FIGDIR / f"{name}.pdf"
    png = FIGDIR / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    print(f"wrote {pdf}")
    print(f"wrote {png}")


# ----------------------------- Figure 1: framework ---------------------------

def fig_framework() -> None:
    """Vertical 4-stage flow with separator banners (clean, non-overlapping)."""
    fig, ax = plt.subplots(figsize=(11.0, 7.6))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 13)
    ax.axis("off")
    ax.set_facecolor("#FFFFFF")

    def box(x, y, w, h, *, fc, ec="#444444", text="", fontsize=9,
            italic=False, lw=0.9, txt_color="#1d1d1d", fontweight="normal", zorder=2):
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.05,rounding_size=0.10",
            linewidth=lw, facecolor=fc, edgecolor=ec, zorder=zorder,
        )
        ax.add_patch(patch)
        if text:
            style = "italic" if italic else "normal"
            ax.text(x + w / 2, y + h / 2, text,
                    ha="center", va="center", fontsize=fontsize,
                    color=txt_color, style=style, fontweight=fontweight,
                    zorder=zorder + 1)

    def arrow(p1, p2, *, color="#555555", lw=1.1, style="-|>",
              connectionstyle="arc3,rad=0"):
        ar = FancyArrowPatch(
            p1, p2,
            arrowstyle=style, mutation_scale=12,
            color=color, linewidth=lw, zorder=3,
            connectionstyle=connectionstyle,
        )
        ax.add_patch(ar)

    # ---- Top banner: Pre-flight gate ----
    box(0.5, 11.5, 17.0, 1.1, fc=PALETTE["rose-1"], ec=PALETTE["rose-3"], lw=1.1,
        text=r"Pre-flight no-op gate:  require  $\overline{\cos}(E_I(x_p),\,E_T(q))_{\,\ell=E} - \overline{\cos}_{\,\ell=I} \geq 0.05$  "
             "    (else swap CLIP backbone)",
        fontsize=10, fontweight="bold", txt_color="#7a3f3f")

    # ---- Stage 1: Anchor path ----
    ax.text(0.5, 10.85, "(1)  Anchor path  —  seller-side, image-only",
            fontsize=10, color="#5a6f64", fontweight="bold", style="italic")
    box(0.7, 9.4, 2.6, 1.2, fc=PALETTE["mist-1"],
        text=r"Product image $x_p$", fontsize=10, fontweight="bold")
    box(4.0, 9.4, 4.4, 1.2, fc=PALETTE["sage-1"],
        text="Seller-side anchor\n(title / category / VLM caption)", fontsize=9)
    box(9.1, 9.4, 3.4, 1.2, fc=PALETTE["sage-1"],
        text=r"CLIP text encoder $E_T$", fontsize=9)
    box(13.2, 9.4, 4.3, 1.2, fc=PALETTE["sage-2"],
        text=r"$v_\mathrm{anchor}(x_p)$  (target embedding)",
        fontsize=10, fontweight="bold")
    arrow((3.3, 10.0), (4.0, 10.0))
    arrow((8.4, 10.0), (9.1, 10.0))
    arrow((12.5, 10.0), (13.2, 10.0))

    # ---- Stage 2: Attack pipeline ----
    ax.text(0.5, 8.65, "(2)  Four-way attack pipeline  —  matched $\\varepsilon = 16/255$, only $x_p$ enters optimizer",
            fontsize=10, color="#5a6f64", fontweight="bold", style="italic")
    method_specs = [
        ("CoGEO",     "Sobel + JPEG\n+ MI-FGSM",     PALETTE["rose-3"]),
        ("PGD-bare",  "Anchor only\n(no mask)",       PALETTE["sage-3"]),
        ("AdvCLIP",   "Universal patch\n30 epochs",   PALETTE["mist-3"]),
        ("Co-Attack", "Cross-modal\nalternating",     PALETTE["sage-2"]),
    ]
    attack_x0 = 0.7
    attack_w = 2.5
    attack_gap = 0.20
    for i, (name, sub, col) in enumerate(method_specs):
        x0 = attack_x0 + i * (attack_w + attack_gap)
        box(x0, 6.6, attack_w, 1.7, fc=col,
            text=f"{name}\n{sub}", fontsize=9, txt_color="#181818", fontweight="bold")
    # x_p down to attack lane (vertical from x_p box)
    arrow((1.9, 9.4), (1.9, 8.3), color="#444444")
    # v_anchor down to attack lane (vertical, drops to right edge of attack 4)
    arrow((11.5, 9.4), (11.5, 8.3), color="#7F8F84")
    ax.text(11.65, 8.7, r"$v_\mathrm{anchor}$", fontsize=8, color="#5a6f64",
            style="italic", ha="left")
    # consolidating right arrow from attacks → x_p*
    box(13.4, 6.8, 4.0, 1.3, fc=PALETTE["rose-1"], ec=PALETTE["rose-3"], lw=1.0,
        text=r"adversarial image" + "\n" + r"$x_p^* = x_p + \delta$",
        fontsize=10, fontweight="bold")
    arrow((11.5, 7.45), (13.4, 7.45))

    # ---- Stage 3: Evaluation pipeline ----
    ax.text(0.5, 5.85, "(3)  Held-out evaluation  —  shopper query never enters any attack gradient",
            fontsize=10, color="#5a6f64", fontweight="bold", style="italic")
    box(0.7, 4.2, 4.0, 1.4, fc=PALETTE["sage-1"],
        text=r"ESCI shopper query $q$" + "\n" + r"human-graded $\ell\!\in\!\{E,S,C,I\}$",
        fontsize=9)
    box(5.2, 4.2, 3.6, 1.4, fc=PALETTE["sage-1"],
        text=r"CLIP text encoder $E_T$" + "\n" + r"$\to E_T(q)$",
        fontsize=9)
    box(9.3, 4.2, 4.1, 1.4, fc=PALETTE["mist-1"],
        text=r"$\cos(E_I(x_p^*),\,E_T(q))$" + "\n" + "on candidate pool",
        fontsize=9)
    box(13.9, 4.2, 3.6, 1.4, fc=PALETTE["mist-1"],
        text="Per-cohort metrics:\nrank-lift / Top-K / $\\Delta\\cos$ / SSIM",
        fontsize=9)

    arrow((4.7, 4.9), (5.2, 4.9))
    arrow((8.8, 4.9), (9.3, 4.9))
    arrow((13.4, 4.9), (13.9, 4.9))
    # x_p* feeds DOWN into cos block (cos block top: y=5.6, x range 9.3..13.4 → center 11.35)
    arrow((15.4, 6.8), (11.35, 5.6), color="#7F8F84",
          connectionstyle="arc3,rad=0.15")
    ax.text(13.6, 6.10, r"$x_p^*$", fontsize=9, color="#5a6f64",
            style="italic", ha="left")

    # ---- Final claim banner ----
    box(0.7, 2.4, 16.8, 1.1, fc=PALETTE["mist-2"],
        text="Aggregate + per-cohort report  →  Claims C1 (per-image gradient dominates),  "
             "C2 (cohort heterogeneity),  C3 (stealth–lift trade-off)",
        fontsize=10, fontweight="bold", txt_color="#222222")
    arrow((15.7, 4.2), (15.7, 3.5), color="#444444")

    # ---- Bottom banner: invariant ----
    box(0.5, 0.4, 17.0, 1.4, fc=PALETTE["mist-1"], ec=PALETTE["mist-3"], lw=1.0,
        text=r"Anchor$\perp$query invariant:  $v_\mathrm{anchor}(x_p)$ depends only on $x_p$;  the eval-time query $q$ never enters any optimization gradient.",
        fontsize=11, fontweight="bold", txt_color="#3b3b3b", italic=True)

    fig.tight_layout()
    _save(fig, "fig_framework")


# ----------------------------- Figure 2: per-cohort bars ---------------------

def fig_percohort_bars() -> None:
    data = _load_data()
    per_label = data["per_label_rank_lift"]

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.set_facecolor(PALETTE["mist-1"])
    ax.grid(True, axis="y", zorder=0)

    n_methods = len(METHOD_ORDER)
    x = np.arange(len(COHORTS))
    width = 0.20

    for i, m in enumerate(METHOD_ORDER):
        vals = []
        for c in COHORTS:
            key = f"{m}_rank_lift_mean"
            v = per_label[c][key]
            vals.append(v)
        offsets = (i - (n_methods - 1) / 2) * width
        bars = ax.bar(
            x + offsets, vals, width,
            color=METHOD_STYLE[m]["color"],
            edgecolor="#444444", linewidth=0.6,
            label=METHOD_STYLE[m]["label"], zorder=3,
        )
        for b, v in zip(bars, vals):
            sign = "+" if v >= 0 else ""
            ax.text(b.get_x() + b.get_width() / 2,
                    v + (0.8 if v >= 0 else -0.8),
                    f"{sign}{v:.1f}",
                    ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=7, color="#222222")

    ax.axhline(0, color="#555555", linewidth=0.6, zorder=2)
    ax.set_xticks(x)
    cohort_tick_labels = [
        f"{c}  ({COHORT_FULLNAME[c]})\n$n={per_label[c]['n']}$"
        for c in COHORTS
    ]
    ax.set_xticklabels(cohort_tick_labels, fontsize=9)
    ax.set_ylabel("Mean rank-lift")
    ax.set_ylim(-7, 46)
    ax.set_title("Per-cohort rank-lift on the 560-pair manifest  (CLIP ViT-L/14, $\\varepsilon=16/255$)",
                 fontsize=10, color="#222222", pad=24)

    leg = ax.legend(loc="upper center", fontsize=9, ncol=4, handlelength=1.4,
                    columnspacing=1.6, bbox_to_anchor=(0.5, 1.10))
    leg.get_frame().set_alpha(0.0)

    # annotate the negative Co-Attack bar in a clear margin
    ax.annotate(
        "Co-Attack\nturns negative on\nComplement (text-anchor drift)",
        xy=(2 + 0.5 * width, -1.86), xytext=(0.10, 30),
        fontsize=7.5, color="#7a3f3f", ha="left", va="top",
        arrowprops=dict(arrowstyle="-|>", color="#B88C8C", lw=0.8,
                        connectionstyle="arc3,rad=-0.2"),
    )
    fig.tight_layout()
    _save(fig, "fig_percohort_bars")


# ----------------------------- Figure 3: Wilcoxon heatmap --------------------

def fig_wilcoxon_heatmap() -> None:
    data = _load_data()
    pw = data["pairwise"]

    method_keys = ["cogeo", "pgd_bare", "advclip", "coattack"]
    n = len(method_keys)
    p = np.full((n, n), np.nan)
    for i, a in enumerate(method_keys):
        for j, b in enumerate(method_keys):
            if i == j:
                continue
            key = f"{a}_vs_{b}"
            rev = f"{b}_vs_{a}"
            if key in pw:
                p[i, j] = pw[key]["wilcoxon_p_greater"]
            elif rev in pw:
                p[i, j] = max(1.0 - pw[rev]["wilcoxon_p_greater"], 1e-300)

    log_p = -np.log10(np.where(np.isnan(p), 1.0, p))
    log_p_disp = np.where(np.isnan(p), np.nan, log_p)

    cmap = LinearSegmentedColormap.from_list(
        "morandi-sig",
        [PALETTE["mist-1"], PALETTE["sage-2"], PALETTE["rose-3"]],
    )

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    im = ax.imshow(log_p_disp, cmap=cmap, vmin=0, vmax=30, aspect="auto", zorder=2)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    labels = [METHOD_STYLE[m]["label"] for m in method_keys]
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("column method (vs)")
    ax.set_ylabel("row method (greater)")
    ax.set_title(r"Paired Wilcoxon $p$-values, one-sided row $>$ column",
                 fontsize=10, color="#222222")

    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", fontsize=10, color="#666666")
                continue
            pv = p[i, j]
            if pv >= 0.5:
                txt = "n.s."
                col = "#444444"
            elif pv >= 0.05:
                txt = f"{pv:.2g}"
                col = "#444444"
            else:
                exp = int(math.floor(math.log10(pv)))
                mant = pv / (10 ** exp)
                txt = f"{mant:.1f}e{exp}"
                col = "#FFFFFF" if log_p[i, j] > 14 else "#222222"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=col)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label(r"$-\log_{10}(p)$", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_edgecolor(PALETTE["mist-3"])
    cbar.outline.set_linewidth(0.6)

    fig.tight_layout()
    _save(fig, "fig_wilcoxon_heatmap")


# ----------------------------- Figure 4: SSIM Pareto v2 ----------------------

def fig_ssim_pareto_v2() -> None:
    data = _load_data()
    rows = []
    for tag in METHOD_ORDER:
        rows.append({
            "tag":  tag,
            "lift": data["per_method"][tag]["rank_lift_mean"],
            "ssim": data["ssim"][tag]["mean"],
        })

    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    ax.set_facecolor(PALETTE["mist-1"])
    ax.grid(True, axis="both", zorder=0)

    pts = sorted([(r["lift"], r["ssim"], r["tag"]) for r in rows])
    front = []
    best_ssim = -math.inf
    for x, y, t in reversed(pts):
        if y > best_ssim:
            front.append((x, y, t))
            best_ssim = y
    front_x = [p[0] for p in reversed(front)]
    front_y = [p[1] for p in reversed(front)]
    ax.plot(front_x, front_y, color=PALETTE["mist-3"], linestyle="--",
            linewidth=1.0, alpha=0.7, zorder=2, label="Pareto frontier")

    # explicit non-overlapping label placement (axes coords)
    label_anchor = {
        "advclip":  (4.0, 0.92),
        "coattack": (1.5, 0.71),
        "pgd_bare": (15.0, 0.83),
        "cogeo":    (21.5, 0.71),
    }
    label_arrow_color = {
        "advclip":  PALETTE["mist-3"],
        "coattack": PALETTE["sage-2"],
        "pgd_bare": PALETTE["sage-3"],
        "cogeo":    PALETTE["rose-3"],
    }
    for r in rows:
        style = METHOD_STYLE[r["tag"]]
        ax.scatter(
            r["lift"], r["ssim"],
            color=style["color"],
            marker=style["marker"],
            s=130, edgecolor="#333333", linewidth=0.8, zorder=4,
        )
        ax_x, ax_y = label_anchor[r["tag"]]
        txt = ax.text(
            ax_x, ax_y,
            f"{style['label']}\n({r['lift']:.1f}, {r['ssim']:.2f})",
            fontsize=8, color="#1d1d1d", zorder=5,
            ha="center", va="center",
        )
        txt.set_path_effects([pe.withStroke(linewidth=2.4, foreground=PALETTE["mist-1"])])
        # leader arrow
        ax.annotate(
            "", xy=(r["lift"], r["ssim"]), xytext=(ax_x, ax_y),
            arrowprops=dict(arrowstyle="-", color=label_arrow_color[r["tag"]],
                            lw=0.8, alpha=0.6),
            zorder=3,
        )

    ax.set_xlabel("Mean rank-lift on 560 paired triples")
    ax.set_ylabel(r"SSIM at 224$\times$224 (retriever native)")
    ax.set_xlim(0, 25)
    ax.set_ylim(0.60, 1.00)
    ax.axhline(0.96, color=PALETTE["rose-3"], linestyle=":", linewidth=1.0,
               alpha=0.85, zorder=3)
    ax.text(0.4, 0.965, "0.96 perceptual-quality reference",
            fontsize=8, color=PALETTE["rose-3"], alpha=0.95)
    ax.text(0.4, 0.605, "lower SSIM ↔ stronger perturbation",
            fontsize=7, color="#666666", style="italic")

    leg = ax.legend(loc="upper right", fontsize=8)
    leg.get_frame().set_alpha(0.0)

    ax.set_title("Stealth–lift Pareto across four CLIP-targeted attacks",
                 fontsize=10, color="#222222")

    fig.tight_layout()
    _save(fig, "fig_ssim_pareto_v2")


def main() -> None:
    fig_framework()
    fig_percohort_bars()
    fig_wilcoxon_heatmap()
    fig_ssim_pareto_v2()


if __name__ == "__main__":
    main()
