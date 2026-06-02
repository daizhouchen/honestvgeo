#!/usr/bin/env python3
"""R5 AE-CoGEO source-encoder scaling-law aggregation.

Pools the held-out per-pair rank_lift CSVs across source-encoder counts n in {1,2,3,4},
all evaluated on the SAME held-out set {laion2b, vith14, vitb32, siglip}, and tests whether
mean held-out transfer rank-lift rises monotonically with n.

Run on the remote (has pandas+scipy):
  cd ${PROJECT_ROOT}/n3 && python code/agg_r5_scaling.py
Writes runs/ae_s2_ens2/r5_scaling_law.json.
"""
import json, os
import numpy as np, pandas as pd

R = "${PROJECT_ROOT}/n3"
HELDOUT = ["laion2b", "vith14", "vitb32", "siglip"]

# (n_sources, dir, filename template). n=1 uses single-encoder 'ens_mean' naming; n>=2 use 'aeens'.
POINTS = [
    (1, f"{R}/runs/ae_transfer_s1", "rank_ens_mean_attacked_openai_eval_{eb}_per_pair.csv"),
    (2, f"{R}/runs/ae_s2_ens2",     "rank_aeens_eval_{eb}_per_pair.csv"),
    (3, f"{R}/runs/ae_v2_ens",      "rank_aeens_eval_{eb}_per_pair.csv"),
    (4, f"{R}/runs/ae_v3_ens4",     "rank_aeens_eval_{eb}_per_pair.csv"),
]
KEY = ["product_id", "query_id"]


def load_point(d, tmpl):
    """Return {eb: Series(rank_lift indexed by (product_id,query_id))} and a pooled long DF."""
    per_enc, frames = {}, []
    for eb in HELDOUT:
        p = os.path.join(d, tmpl.format(eb=eb))
        if not os.path.exists(p):
            per_enc[eb] = None
            continue
        df = pd.read_csv(p)
        s = df.set_index(KEY)["rank_lift"]
        per_enc[eb] = s
        f = df[KEY + ["rank_lift"]].copy(); f["enc"] = eb
        frames.append(f)
    pooled = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return per_enc, pooled


def main():
    curve = {}        # n -> {pooled_mean, n_pairs, per_enc_mean{}}
    pooled_by_n = {}  # n -> pooled long DF (rank_lift indexed by (product_id,query_id,enc))
    for n, d, tmpl in POINTS:
        if not os.path.isdir(d):
            curve[n] = {"status": "missing_dir", "dir": d}
            continue
        per_enc, pooled = load_point(d, tmpl)
        pe_mean = {eb: (round(float(s.mean()), 4) if s is not None else None) for eb, s in per_enc.items()}
        rec = {
            "dir": d,
            "per_enc_mean": pe_mean,
            "pooled_mean": round(float(pooled["rank_lift"].mean()), 4) if len(pooled) else None,
            "n_pairs_pooled": int(len(pooled)),
        }
        curve[n] = rec
        if len(pooled):
            pooled_by_n[n] = pooled.set_index(KEY + ["enc"])["rank_lift"]

    # Monotonicity: Spearman(n, pooled_mean) over available points.
    ns = [n for n in sorted(curve) if isinstance(curve[n], dict) and curve[n].get("pooled_mean") is not None]
    means = [curve[n]["pooled_mean"] for n in ns]
    out = {"held_out_encoders": HELDOUT, "curve": curve, "available_n": ns, "pooled_means": means}
    try:
        from scipy.stats import spearmanr
        if len(ns) >= 3:
            rho, p = spearmanr(ns, means)
            out["spearman_n_vs_mean"] = {"rho": round(float(rho), 4), "p": round(float(p), 5)}
    except Exception as ex:
        out["spearman_n_vs_mean"] = {"error": str(ex)}

    # Consecutive paired Wilcoxon on the common pooled (product_id,query_id,enc) pairs.
    steps = {}
    for a, b in zip(ns, ns[1:]):
        if a in pooled_by_n and b in pooled_by_n:
            sa, sb = pooled_by_n[a], pooled_by_n[b]
            idx = sa.index.intersection(sb.index)
            x = sa.loc[idx].to_numpy(float); y = sb.loc[idx].to_numpy(float); dd = y - x
            row = {"n_pairs": int(len(idx)), "mean_diff": round(float(dd.mean()), 4),
                   "win_rate": round(float((dd > 0).mean()), 4)}
            try:
                from scipy.stats import wilcoxon
                if np.any(dd != 0):
                    row["wilcoxon_p"] = round(float(wilcoxon(y, x, alternative="greater").pvalue), 6)
            except Exception as ex:
                row["wilcoxon_p"] = f"err:{ex}"
            steps[f"{a}->{b}"] = row
    out["consecutive_steps"] = steps

    op = f"{R}/runs/ae_s2_ens2/r5_scaling_law.json"
    json.dump(out, open(op, "w"), indent=2)
    print(json.dumps(out, indent=2))
    print("written", op)


if __name__ == "__main__":
    main()
