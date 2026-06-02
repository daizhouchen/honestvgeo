#!/usr/bin/env bash
# L7 5-way attack-family transfer eval: re-score cogeo/pgd_bare/advclip/coattack/diffprior
# with one fixed encoder (ViT-L-14, the shared white-box surrogate) so all five numbers
# are mutually comparable. cogeo should reproduce the existing ~15.7 rank_lift_mean as a
# sanity check that the backbone matches the published attack-family table.
set -uo pipefail
R=${PROJECT_ROOT}/n3
cd ${PROJECT_ROOT}
source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-honestvgeo}
export HF_ENDPOINT=https://hf-mirror.com TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=5
OUT=$R/runs/L7_5way
mkdir -p "$OUT"
MAN=$R/data/esci500_thumbs/manifest.csv
IMG=$R/data/esci500_thumbs/img
BB=openai/clip-vit-large-patch14
echo "L7EVAL_START $(date -u +%FT%TZ) backbone=$BB cuda=$CUDA_VISIBLE_DEVICES"

score () {
  local tag="$1" dir="$2"
  echo "=== $tag adv-dir=runs/$dir/img $(date -u +%FT%TZ) ==="
  python "$R/code/eval_harness.py" --mode rank --manifest "$MAN" --img-dir "$IMG" \
    --adv-dir "$R/runs/$dir/img" --out-dir "$OUT" --method-tag "$tag" \
    --clip-backbone "$BB" --gpu 0 2>&1
  echo "=== ${tag}_DONE $(date -u +%FT%TZ) ==="
}

score cogeo     n3_cogeo_eps16
score pgd_bare  n3_pgd_bare_eps16
score advclip   n3_advclip_eps16
score coattack  n3_coattack_eps16
score diffprior L7_diffprior_eps16

echo "L7EVAL_AGG $(date -u +%FT%TZ)"
python - <<'PY'
import json, os
OUT = "${PROJECT_ROOT}/n3/runs/L7_5way"
rows = []
for m in ["cogeo", "pgd_bare", "advclip", "coattack", "diffprior"]:
    p = os.path.join(OUT, f"rank_{m}_summary.json")
    if os.path.exists(p):
        d = json.load(open(p))
        rows.append((m, d["rank_lift_mean"], d["rank_lift_median"], d["delta_held_out_mean"],
                     d.get("per_label_delta_mean", {})))
        print(f"{m:10s} rank_lift_mean={d['rank_lift_mean']:.3f} median={d['rank_lift_median']:.3f} delta_held_out={d['delta_held_out_mean']:.4f}")
    else:
        print(f"{m:10s} MISSING")
json.dump({"backbone": "openai/clip-vit-large-patch14",
           "rows": [{"method": r[0], "rank_lift_mean": r[1], "rank_lift_median": r[2],
                     "delta_held_out_mean": r[3], "per_label_delta_mean": r[4]} for r in rows]},
          open(os.path.join(OUT, "l7_5way_compare.json"), "w"), indent=2)
print("wrote l7_5way_compare.json")
PY
echo "L7EVAL_DONE $(date -u +%FT%TZ)"
