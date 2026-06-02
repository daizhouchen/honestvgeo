#!/usr/bin/env python3
"""Aggregate R4 purification-defense eval CSVs (rank_<tag>_per_pair.csv written by
eval_harness) into a per-cohort/per-encoder defended-vs-undefended rank-lift table
plus clean-retrieval cost. Output: r4_defense_summary.json + console table."""
import sys, json, os
import numpy as np
import pandas as pd

OUT = sys.argv[1]
DEFENSES = ["none", "bd5", "bd4", "rs50", "jq40", "md3"]
TRANSFER_EB = ["laion2b", "vith14", "vitb32", "siglip"]
CLEAN_EB = ["openai", "laion2b"]


def load(tag):
    p = os.path.join(OUT, f"rank_{tag}_per_pair.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def cohort_means(df):
    out = {}
    for lab in ["E", "S", "C", "I"]:
        m = df[df["esci_label"] == lab]["rank_lift"]
        out[lab] = round(float(m.mean()), 3) if len(m) else None
    out["ALL"] = round(float(df["rank_lift"].mean()), 3)
    out["n"] = int(len(df))
    return out


def efficacy(block):
    """undefended(none) - defended  (positive = rank-lift suppressed by defense)."""
    base = block.get("none")
    if not base:
        return {}
    eff = {}
    for tag, vals in block.items():
        if tag == "none":
            continue
        eff[tag] = {k: (round(base[k] - vals[k], 3)
                        if (base.get(k) is not None and vals.get(k) is not None) else None)
                    for k in ["E", "S", "C", "I", "ALL"]}
    return eff


res = {"transfer": {}, "eps4_openai": {}, "clean_cost": {}}

for eb in TRANSFER_EB:
    res["transfer"][eb] = {}
    for tag in DEFENSES:
        df = load(f"transfer_{eb}_{tag}")
        if df is not None:
            res["transfer"][eb][tag] = cohort_means(df)

for tag in DEFENSES:
    df = load(f"eps4_openai_{tag}")
    if df is not None:
        res["eps4_openai"][tag] = cohort_means(df)

for eb in CLEAN_EB:
    res["clean_cost"][eb] = {}
    for tag in DEFENSES:
        if tag == "none":
            continue
        df = load(f"clean_{eb}_{tag}")
        if df is not None:
            res["clean_cost"][eb][tag] = cohort_means(df)

res["transfer_efficacy"] = {eb: efficacy(res["transfer"].get(eb, {})) for eb in TRANSFER_EB}
res["eps4_efficacy"] = efficacy(res["eps4_openai"])

with open(os.path.join(OUT, "r4_defense_summary.json"), "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=2))
print("written", os.path.join(OUT, "r4_defense_summary.json"))
