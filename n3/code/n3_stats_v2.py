#!/usr/bin/env python3
"""Extended N3.5 stats: 4-way method comparison.

Methods: cogeo, pgd_bare, advclip, coattack.
For each pair (A, B) of methods, compute paired Wilcoxon (alt=greater) of
rank_lift(A) vs rank_lift(B) on the same (example_id, product_id) keys.
Also compute per-label rank_lift means, SSIM means, I-label top-3 promotion
rates, and bootstrap CI95 of (A - B) rank_lift mean.

Inputs (relative to the project root passed as argv[1]):
    runs/n3_eval/rank_<method>_per_pair.csv  for each method in --methods
    runs/n3_<method>_eps16/img/             for SSIM (path varies by method)
    data/esci500_thumbs/img/                for original SSIM reference

Output:
    runs/n3_eval/n3_h2_summary.json
"""
from __future__ import annotations
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from skimage.metrics import structural_similarity as ssim_f


METHOD_ATTACK_DIR = {
    "cogeo": "runs/n3_cogeo_eps16/img",
    "pgd_bare": "runs/n3_pgd_bare_eps16/img",
    "advclip": "runs/n3_advclip_eps16/img",
    "coattack": "runs/n3_coattack_eps16/img",
}


def load_per_pair(root: Path, method: str) -> pd.DataFrame:
    p = root / "runs" / "n3_eval" / f"rank_{method}_per_pair.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="project root, e.g. ${PROJECT_ROOT}/n3")
    parser.add_argument(
        "--methods", default="cogeo,pgd_bare,advclip,coattack",
        help="comma-separated method tags (must match rank_<tag>_per_pair.csv files)",
    )
    parser.add_argument("--out", type=Path, default=None,
                        help="output json path; default <root>/runs/n3_eval/n3_h2_summary.json")
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ssim-image-size", type=int, default=224)
    args = parser.parse_args(argv)

    root = args.root
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    print(f"methods: {methods}")

    # Load per-pair tables; merge on (example_id, product_id) so all are aligned.
    tables = {}
    for m in methods:
        df = load_per_pair(root, m)
        tables[m] = df
        print(f"  rank_{m}_per_pair.csv n={len(df)}  cols={list(df.columns)[:6]}...")

    # Pivot to wide table on (example_id, product_id).
    base = tables[methods[0]][["example_id", "product_id", "esci_label", "rank_orig"]].copy()
    base.rename(columns={"esci_label": "esci_label_ref",
                         "rank_orig": "rank_orig_ref"}, inplace=True)
    wide = base
    for m in methods:
        sub = tables[m][["example_id", "product_id", "rank_lift", "rank_adv", "delta_held_out"]].copy()
        sub.rename(columns={
            "rank_lift": f"rl_{m}",
            "rank_adv": f"rank_adv_{m}",
            "delta_held_out": f"dho_{m}",
        }, inplace=True)
        wide = wide.merge(sub, on=["example_id", "product_id"], how="inner")
    print(f"merged wide n={len(wide)} (intersection over methods)")

    # Per-method aggregate.
    per_method = {}
    for m in methods:
        rl = wide[f"rl_{m}"].to_numpy()
        per_method[m] = {
            "rank_lift_mean": float(rl.mean()),
            "rank_lift_median": float(np.median(rl)),
            "rank_lift_std": float(rl.std()),
            "delta_held_out_mean": float(wide[f"dho_{m}"].mean()),
        }

    # Per-label rank_lift per method.
    per_label = {}
    for lab in ("E", "S", "C", "I"):
        sub = wide[wide["esci_label_ref"] == lab]
        per_label[lab] = {"n": int(len(sub))}
        for m in methods:
            per_label[lab][f"{m}_rank_lift_mean"] = float(sub[f"rl_{m}"].mean()) if len(sub) else None

    # Pairwise Wilcoxon (alt=greater) and bootstrap CI95 of (A - B) rank_lift mean.
    pairwise = {}
    rng = np.random.RandomState(args.seed)
    for a, b in itertools.permutations(methods, 2):
        diff = wide[f"rl_{a}"].to_numpy() - wide[f"rl_{b}"].to_numpy()
        try:
            w = stats.wilcoxon(wide[f"rl_{a}"], wide[f"rl_{b}"], alternative="greater")
            w_stat = float(w.statistic)
            w_p = float(w.pvalue)
        except Exception as e:
            w_stat, w_p = None, None
            print(f"wilcoxon {a}>{b} failed: {e}")
        boot_means = np.array([
            np.random.choice(diff, size=len(diff), replace=True).mean()
            for _ in range(args.bootstrap_iters)
        ])
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
        pairwise[f"{a}_vs_{b}"] = {
            "diff_mean": float(diff.mean()),
            "diff_median": float(np.median(diff)),
            "ci95": [float(ci_lo), float(ci_hi)],
            "wilcoxon_W": w_stat,
            "wilcoxon_p_greater": w_p,
        }

    # I-label top-3 promotion rate per method (uses esci_label_ref ⇒ I label only).
    i_top3 = {}
    i_orig_top3 = float(((wide["esci_label_ref"] == "I") & (wide["rank_orig_ref"] <= 3)).mean())
    i_top3["original"] = i_orig_top3
    for m in methods:
        rate = float(((wide["esci_label_ref"] == "I") & (wide[f"rank_adv_{m}"] <= 3)).mean())
        i_top3[m] = rate
        i_top3[f"{m}_gain_pp_vs_original"] = rate - i_orig_top3

    # SSIM per method on unique ASINs.
    img_dir = root / "data/esci500_thumbs/img"
    asins = wide["product_id"].drop_duplicates().tolist()
    ssim_per_method = {}
    print(f"computing SSIM on {len(asins)} unique ASINs across {len(methods)} methods ...")
    sz = (args.ssim_image_size, args.ssim_image_size)
    for m in methods:
        adv_dir = root / METHOD_ATTACK_DIR[m]
        ssims, n_skip = [], 0
        for asin in asins:
            try:
                orig = np.asarray(Image.open(img_dir / f"{asin}.jpg").convert("RGB").resize(sz)) / 255.0
                adv = np.asarray(Image.open(adv_dir / f"{asin}.jpg").convert("RGB").resize(sz)) / 255.0
                ssims.append(float(ssim_f(orig, adv, channel_axis=2, data_range=1.0)))
            except Exception as e:
                n_skip += 1
        ssim_per_method[m] = {
            "mean": float(np.mean(ssims)) if ssims else None,
            "n": len(ssims),
            "n_skip": n_skip,
        }

    summary = {
        "n_pairs": int(len(wide)),
        "n_unique_asins": len(asins),
        "methods": methods,
        "per_method": per_method,
        "per_label_rank_lift": per_label,
        "pairwise": pairwise,
        "i_label_top3": i_top3,
        "ssim": ssim_per_method,
    }

    out = args.out or root / "runs" / "n3_eval" / "n3_h2_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    print()
    print("=== N3.5 H2 SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    sys.exit(main() or 0)
