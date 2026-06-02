#!/usr/bin/env python3
"""Compute paired Wilcoxon + bootstrap CI95 + SSIM for the N3 H1 hypothesis."""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from PIL import Image
from skimage.metrics import structural_similarity as ssim_f

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
out = ROOT / "runs" / "n3_eval" / "n3_h1_summary.json"

c = pd.read_csv(ROOT / "runs/n3_eval/rank_cogeo_per_pair.csv")
p = pd.read_csv(ROOT / "runs/n3_eval/rank_pgd_bare_per_pair.csv")
print(f"cogeo n={len(c)}, pgd_bare n={len(p)}")

m = c.merge(p, on=["example_id", "product_id"], suffixes=("_cg", "_pb"))
print(f"merged n={len(m)}")

diff_rl = m["rank_lift_cg"] - m["rank_lift_pb"]

# Bootstrap CI95
rng = np.random.RandomState(42)
boot_means = np.array([
    diff_rl.sample(len(diff_rl), replace=True, random_state=rng.randint(99999)).mean()
    for _ in range(1000)
])
ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

# Wilcoxon paired (one-sided: cogeo > pgd_bare)
w = stats.wilcoxon(m["rank_lift_cg"], m["rank_lift_pb"], alternative="greater")

# delta_held_out diff
diff_d = m["delta_held_out_cg"] - m["delta_held_out_pb"]

# Per-label rank_lift
per_label = {}
for lab in ("E", "S", "C", "I"):
    sub = m[m["esci_label_cg"] == lab]
    if len(sub) > 0:
        cg_m = float(sub["rank_lift_cg"].mean())
        pb_m = float(sub["rank_lift_pb"].mean())
        per_label[lab] = {"n": int(len(sub)), "cg_mean": cg_m, "pb_mean": pb_m, "diff": cg_m - pb_m}

# I-label top-K promotion
i_orig_top3 = float(((m["esci_label_cg"] == "I") & (m["rank_orig_cg"] <= 3)).mean())
i_cg_top3 = float(((m["esci_label_cg"] == "I") & (m["rank_adv_cg"] <= 3)).mean())
i_pb_top3 = float(((m["esci_label_cg"] == "I") & (m["rank_adv_pb"] <= 3)).mean())

# SSIM on full set (491 unique ASINs)
print("computing SSIM on all unique ASINs ...")
img_dir = ROOT / "data/esci500_thumbs/img"
cg_dir = ROOT / "runs/n3_cogeo_eps16/img"
pb_dir = ROOT / "runs/n3_pgd_bare_eps16/img"
asins = m["product_id"].drop_duplicates().tolist()
ssims_cg, ssims_pb = [], []
for asin in asins:
    try:
        orig = np.asarray(Image.open(img_dir / f"{asin}.jpg").convert("RGB").resize((224, 224))) / 255.0
        cg = np.asarray(Image.open(cg_dir / f"{asin}.jpg").convert("RGB").resize((224, 224))) / 255.0
        pb = np.asarray(Image.open(pb_dir / f"{asin}.jpg").convert("RGB").resize((224, 224))) / 255.0
        ssims_cg.append(float(ssim_f(orig, cg, channel_axis=2, data_range=1.0)))
        ssims_pb.append(float(ssim_f(orig, pb, channel_axis=2, data_range=1.0)))
    except Exception as e:
        print(f"  ssim skip {asin}: {e}")
ssim_cg = float(np.mean(ssims_cg))
ssim_pb = float(np.mean(ssims_pb))

summary = {
    "n_pairs": int(len(m)),
    "n_unique_asins": len(asins),
    "rank_lift_cogeo_mean": float(m["rank_lift_cg"].mean()),
    "rank_lift_pgd_bare_mean": float(m["rank_lift_pb"].mean()),
    "rank_lift_diff_mean": float(diff_rl.mean()),
    "rank_lift_diff_median": float(diff_rl.median()),
    "rank_lift_diff_ci95": [float(ci_lo), float(ci_hi)],
    "wilcoxon_paired_greater": {"W": float(w.statistic), "p_value": float(w.pvalue)},
    "delta_held_out_diff_mean": float(diff_d.mean()),
    "per_label_rank_lift": per_label,
    "i_label_top3_orig": i_orig_top3,
    "i_label_top3_cogeo": i_cg_top3,
    "i_label_top3_pgd_bare": i_pb_top3,
    "i_label_top3_gain_cg_vs_pb": (i_cg_top3 - i_pb_top3),
    "ssim_cogeo_mean": ssim_cg,
    "ssim_pgd_bare_mean": ssim_pb,
    "ssim_n": len(ssims_cg),
}

print()
print("=== N3 H1 SUMMARY ===")
print(json.dumps(summary, indent=2))
out.write_text(json.dumps(summary, indent=2))
print()
print(f"wrote {out}")
