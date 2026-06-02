#!/usr/bin/env python3
"""Harvest R2: build the CLIP (dual-encoder) vs BLIP (fused-encoder ITM rerank)
cross-paradigm comparison from the saved per-pair CSVs / summaries. rank_lift is
defined identically (rank_orig - rank_adv, positive = item lifted up), so the two
paradigms are directly comparable. Writes clip_vs_blip_compare.json."""
import json, os
import pandas as pd
import numpy as np

BASE = os.path.expanduser("${PROJECT_ROOT}")
CLIP = {
 "cogeo":    f"{BASE}/runs/T3_scaleup/run4/eval/rank_cogeo_per_pair.csv",
 "pgd_bare": f"{BASE}/runs/T3_scaleup/run4/eval/rank_pgd_bare_per_pair.csv",
 "coattack": f"{BASE}/runs/B2_full4method/eps16_1430/eval/rank_coattack_per_pair.csv",
 "advclip":  f"{BASE}/runs/B2_full4method/eps16_1430/eval/rank_advclip_per_pair.csv",
}
BLIP = {m: f"{BASE}/runs/R2_blip_rerank/{m}/blip_rank_{m}_summary.json" for m in CLIP}
LAB = ["E", "S", "C", "I"]
out = {}
print(f"{'method':9s} | {'CLIP all':>8s} {'BLIP all':>8s} | per-cohort rank-lift  C/I=hard, E/S=easy")
print("-" * 78)
for m in ["cogeo", "pgd_bare", "coattack", "advclip"]:
    if not os.path.exists(CLIP[m]):
        print(f"{m}: CLIP per-pair MISSING {CLIP[m]}"); continue
    if not os.path.exists(BLIP[m]):
        print(f"{m}: BLIP summary MISSING (not done yet?)"); continue
    c = pd.read_csv(CLIP[m])
    col = "rank_lift" if "rank_lift" in c.columns else [x for x in c.columns if "rank_lift" in x][0]
    clip_all = float(c[col].mean())
    clip_pc = {l: float(c.loc[c.esci_label == l, col].mean()) for l in LAB if (c.esci_label == l).any()}
    b = json.load(open(BLIP[m]))
    blip_all = float(b["rank_lift_rerank_mean"])
    blip_pc = {k: float(v) for k, v in b["per_label_rank_lift_rerank"].items()}
    out[m] = {"clip_overall": clip_all, "blip_overall": blip_all,
              "clip_percohort": clip_pc, "blip_percohort": blip_pc,
              "blip_itm_delta_mean": b.get("itm_delta_mean")}
    print(f"{m:9s} | {clip_all:8.2f} {blip_all:8.2f} | "
          f"CLIP[" + " ".join(f"{l}:{clip_pc.get(l,float('nan')):.1f}" for l in LAB) + "]  "
          f"BLIP[" + " ".join(f"{l}:{blip_pc.get(l,float('nan')):.1f}" for l in LAB) + "]")
# cohort-heterogeneity check: is (C+I)/2 >> (E+S)/2 under BOTH paradigms?
print("-" * 78)
for m in out:
    cp, bp = out[m]["clip_percohort"], out[m]["blip_percohort"]
    def hard_easy(pc):
        hard = np.mean([pc[l] for l in ("C", "I") if l in pc])
        easy = np.mean([pc[l] for l in ("E", "S") if l in pc])
        return hard, easy
    ch, ce = hard_easy(cp); bh, be = hard_easy(bp)
    print(f"{m:9s} hard-vs-easy  CLIP {ch:.1f} vs {ce:.1f} (x{ch/ce:.1f})   BLIP {bh:.1f} vs {be:.1f} (x{bh/be:.1f})")
json.dump(out, open(f"{BASE}/runs/R2_blip_rerank/clip_vs_blip_compare.json", "w"), indent=2)
print("WROTE", f"{BASE}/runs/R2_blip_rerank/clip_vs_blip_compare.json")
print("HARVEST_OK")
