#!/usr/bin/env bash
# Stage 1 transfer probe: does anchor-ensemble (ens_mean) retain MORE held-out-encoder
# transfer rank-lift than faithful single-anchor CoGEO (title)?
# Pure re-evaluation of the EXISTING ae_v1_es adversarials (optimized on openai ViT-L/14,
# env-sim ON) scored with held-out backbones. No new optimization -> cheap.
set -uo pipefail
source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-honestvgeo}
export HF_ENDPOINT=https://hf-mirror.com TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${GPU:-5}"
R=${PROJECT_ROOT}/n3
CODE=$R/code
MAN=$R/data/esci500_thumbs/manifest.csv
IMG=$R/data/esci500_thumbs/img
SRC=$R/runs/ae_v1_es                 # adversarials optimized on openai ViT-L/14
OUT=$R/runs/ae_transfer_s1
mkdir -p "$OUT"
cd "$CODE"
declare -A BBN=( [laion2b]=ViT-L-14-laion2b [eva02]=EVA02-L-14 [vith14]=ViT-H-14-laion2b [vitb32]=ViT-B-32-laion2b [siglip]=ViT-B-16-SigLIP-webli )
ARMS="${ARMS:-title ens_mean}"
EVALBB="${EVALBB:-laion2b eva02 vith14 vitb32 siglip}"
echo "[s1] START $(date -u) gpu=${GPU:-5} arms='$ARMS' eval_bb='$EVALBB'"
for eb in $EVALBB; do
  bbname=${BBN[$eb]}
  for arm in $ARMS; do
    tag="${arm}_attacked_openai_eval_${eb}"
    if [ -f "$OUT/rank_${tag}_per_pair.csv" ]; then echo "[s1] skip done $tag"; continue; fi
    echo "[s1] === $arm -> $eb ($bbname) $(date -u) ==="
    python eval_harness.py --mode rank --manifest "$MAN" --img-dir "$IMG" \
      --out-dir "$OUT" --adv-dir "$SRC/$arm/img" --method-tag "$tag" \
      --clip-backbone "$bbname" --gpu 0 >>"$OUT/$eb.$arm.log" 2>&1
    echo "[s1] $arm -> $eb rc=$? $(date -u)"
  done
done
echo "[s1] ALL EVAL DONE $(date -u)"
python - "$OUT" "$EVALBB" <<'PY'
import sys, json, numpy as np, pandas as pd
OUT=sys.argv[1]; ebs=sys.argv[2].split(); res={}
for eb in ebs:
    try:
        t=pd.read_csv(f"{OUT}/rank_title_attacked_openai_eval_{eb}_per_pair.csv")
        e=pd.read_csv(f"{OUT}/rank_ens_mean_attacked_openai_eval_{eb}_per_pair.csv")
        ti=t.set_index(["product_id","query_id"])["rank_lift"]
        ei=e.set_index(["product_id","query_id"])["rank_lift"]
        idx=ti.index.intersection(ei.index)
        b=ti.loc[idx].to_numpy(float); c=ei.loc[idx].to_numpy(float); d=c-b
        row={"n":int(len(idx)),"title_transfer":round(float(b.mean()),3),
             "ens_mean_transfer":round(float(c.mean()),3),
             "mean_diff":round(float(d.mean()),3),"median_diff":round(float(np.median(d)),3),
             "win_rate":round(float((d>0).mean()),3),"tie_rate":round(float((d==0).mean()),3)}
        try:
            from scipy.stats import wilcoxon
            if np.any(d!=0): row["wilcoxon_p"]=round(float(wilcoxon(c,b).pvalue),5)
        except Exception as ex: row["wilcoxon_p"]=f"err:{ex}"
    except Exception as ex:
        row={"error":str(ex)}
    res[eb]=row
print("=== Stage1 anchor-ensemble transfer A/B (title vs ens_mean), held-out encoders ===")
print(json.dumps(res, indent=2))
json.dump(res, open(f"{OUT}/transfer_s1_compare.json","w"), indent=2)
print("written", f"{OUT}/transfer_s1_compare.json")
PY
echo "[s1] DONE $(date -u)"
