"""Render Figure 1: SSIM vs mean rank-lift Pareto across the four attack families.

Morandi mist-stone primary + sage-clay contrast. Single column width, vector PDF.
Source data: experiments/main/n3_esci/runs/n3_h2_summary.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "experiments" / "main" / "n3_esci" / "runs" / "n3_h2_summary.json"
OUT_PDF = Path(__file__).parent / "ssim_pareto.pdf"
OUT_PNG = Path(__file__).parent / "ssim_pareto.png"

PALETTE = {
    "mist-stone-1": "#F3EEE8",
    "mist-stone-2": "#D8D1C7",
    "mist-stone-3": "#8A9199",
    "sage-clay-1":  "#E7E1D6",
    "sage-clay-2":  "#B7A99A",
    "sage-clay-3":  "#7F8F84",
    "dust-rose-3":  "#B88C8C",
}

METHOD_STYLE = {
    "advclip":   {"color": PALETTE["mist-stone-3"], "marker": "o", "label": "AdvCLIP"},
    "pgd_bare":  {"color": PALETTE["sage-clay-3"],  "marker": "s", "label": "PGD-bare"},
    "coattack":  {"color": PALETTE["sage-clay-2"],  "marker": "D", "label": "Co-Attack"},
    "cogeo":     {"color": PALETTE["dust-rose-3"],  "marker": "^", "label": "CoGEO"},
}

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.edgecolor":     PALETTE["mist-stone-3"],
    "axes.linewidth":     0.8,
    "axes.labelcolor":    "#333333",
    "xtick.color":        "#444444",
    "ytick.color":        "#444444",
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "grid.color":         PALETTE["mist-stone-2"],
    "grid.linestyle":     "--",
    "grid.linewidth":     0.5,
    "grid.alpha":         0.7,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
})


def main() -> None:
    data = json.loads(SRC.read_text())
    rows = []
    for tag in ["advclip", "pgd_bare", "coattack", "cogeo"]:
        rows.append({
            "tag":  tag,
            "lift": data["per_method"][tag]["rank_lift_mean"],
            "ssim": data["ssim"][tag]["mean"],
        })

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.set_facecolor(PALETTE["mist-stone-1"])
    ax.grid(True, axis="both")

    for r in rows:
        style = METHOD_STYLE[r["tag"]]
        ax.scatter(
            r["lift"], r["ssim"],
            color=style["color"],
            marker=style["marker"],
            s=110, edgecolor="#444444", linewidth=0.6, zorder=3,
        )
        offset = (6, 4) if r["tag"] != "pgd_bare" else (-46, 6)
        txt = ax.annotate(
            style["label"],
            xy=(r["lift"], r["ssim"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=9, color="#222222",
        )
        txt.set_path_effects([pe.withStroke(linewidth=2.0, foreground=PALETTE["mist-stone-1"])])

    ax.set_xlabel("Mean rank-lift")
    ax.set_ylabel("SSIM at 224×224")
    ax.set_xlim(0, 24)
    ax.set_ylim(0.62, 0.82)
    ax.axhline(0.96, color=PALETTE["mist-stone-3"], linestyle=":", linewidth=0.8, alpha=0.6)
    ax.text(
        0.4, 0.965,
        "0.96 perceptual-quality reference",
        fontsize=7, color=PALETTE["mist-stone-3"], alpha=0.95,
    )
    ax.text(
        0.4, 0.625,
        "lower SSIM ↔ stronger perturbation",
        fontsize=7, color="#666666", style="italic",
    )

    fig.tight_layout()
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG)
    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
