#!/usr/bin/env bash
SRC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"  # repo src/ root
# Stage 2b: 4-encoder ensemble AE-CoGEO (arm B') attack + held-out-encoder transfer eval.
# Compared against BOTH the single-encoder anchor-ensemble arm (arm A = ens_mean from
# ae_transfer_s1) AND the 3-encoder ensemble arm (arm B = ae_v2_ens), so we can read off
# the marginal lift from adding the 4th, cross-corpus source encoder.
#   SOURCE ensemble : OpenAI ViT-L/14 + OpenAI ViT-B/32 + EVA02-L-14 + DataComp-XL ViT-L/14
#   HELD-OUT eval   : LAION ViT-L/14, LAION ViT-H/14, LAION ViT-B/32, SigLIP   (none in source)
# The 4th source (DataComp ViT-L/14) shares LAION-L's architecture but a disjoint corpus,
# and is NOT one of the held-out checkpoints -> held-out set stays fully held out.
# Env knobs: GPU (default 5), MAXIMG (smoke cap), ITERS (default 200), TAG (default ae_v3_ens4)
set -uo pipefail
source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-honestvgeo}
export HF_ENDPOINT=https://hf-mirror.com TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${GPU:-5}"
R=${PROJECT_ROOT}
CODE=$R/code
MAN=$R/data/esci500_thumbs/manifest.csv
IMG=$R/data/esci500_thumbs/img
TAG=${TAG:-ae_v3_ens4}
OUT=$R/runs/$TAG
ARMA=$R/runs/ae_transfer_s1            # single-encoder ens_mean held-out per-pair csvs (arm A)
ARMB=$R/runs/ae_v2_ens                 # 3-encoder ensemble held-out per-pair csvs (arm B)
ITERS=${ITERS:-200}
MAXARG=""; [ -n "${MAXIMG:-}" ] && MAXARG="--max-images ${MAXIMG}"
SRC="openai/clip-vit-large-patch14,ViT-B-32,EVA02-L-14,ViT-L-14-datacomp"
mkdir -p "$OUT"
cd "$CODE"
echo "[ae2] START $(date -u) tag=$TAG gpu=${GPU:-5} iters=$ITERS maximg=${MAXIMG:-all}"

echo "[ae2] === ATTACK (encoder-ensemble) $(date -u) ==="
python "$SRC_ROOT/attacks/n3_attack_ae_ens.py" --clip-backbones "$SRC" \
  --anchor-mode ensemble --ensemble-agg mean --max-anchors 6 \
  --manifest "$MAN" --img-dir "$IMG" --out-dir "$OUT" \
  --eps 16 --alpha 4 --iters "$ITERS" --image-size 224 --gpu 0 --seed 42 --use-envsim $MAXARG \
  >>"$OUT/attack.log" 2>&1
echo "[ae2] attack rc=$? $(date -u)"
[ -n "${MAXIMG:-}" ] && { echo "[ae2] SMOKE done (attack only). summary:"; cat "$OUT/attack_summary.json" 2>/dev/null; echo "[ae2] SMOKE END $(date -u)"; exit 0; }

declare -A BBN=( [openai]=openai/clip-vit-large-patch14 [laion2b]=ViT-L-14-laion2b [vith14]=ViT-H-14-laion2b [vitb32]=ViT-B-32-laion2b [siglip]=ViT-B-16-SigLIP-webli )
for eb in openai laion2b vith14 vitb32 siglip; do
  tag="aeens_eval_${eb}"
  echo "[ae2] === EVAL arm=B -> $eb (${BBN[$eb]}) $(date -u) ==="
  python "$SRC_ROOT/eval/eval_harness.py" --mode rank --manifest "$MAN" --img-dir "$IMG" \
    --out-dir "$OUT" --adv-dir "$OUT/img" --method-tag "$tag" \
    --clip-backbone "${BBN[$eb]}" --gpu 0 >>"$OUT/eval.$eb.log" 2>&1
  echo "[ae2] eval $eb rc=$? $(date -u)"
done

echo "[ae2] === COMPARE arm B' (4-enc ensemble) vs arm A (single-enc) and vs arm B (3-enc) on held-out $(date -u) ==="
python - "$OUT" "$ARMA" "$ARMB" <<'PY'
import sys, json, numpy as np, pandas as pd
OUT, ARMA, ARMB = sys.argv[1], sys.argv[2], sys.argv[3]
heldout = ["laion2b","vith14","vitb32","siglip"]

def paired(ref_csv, new_csv):
    a = pd.read_csv(ref_csv); b = pd.read_csv(new_csv)
    ai = a.set_index(["product_id","query_id"])["rank_lift"]
    bi = b.set_index(["product_id","query_id"])["rank_lift"]
    idx = ai.index.intersection(bi.index)
    x = ai.loc[idx].to_numpy(float); y = bi.loc[idx].to_numpy(float); d = y - x
    row = {"n": int(len(idx)),
           "ref_transfer": round(float(x.mean()), 3),
           "new_transfer": round(float(y.mean()), 3),
           "mean_diff": round(float(d.mean()), 3),
           "median_diff": round(float(np.median(d)), 3),
           "win_rate": round(float((d > 0).mean()), 3),
           "tie_rate": round(float((d == 0).mean()), 3)}
    try:
        from scipy.stats import wilcoxon
        if np.any(d != 0): row["wilcoxon_p"] = round(float(wilcoxon(y, x).pvalue), 5)
    except Exception as ex: row["wilcoxon_p"] = f"err:{ex}"
    return row

res = {"vs_single_enc": {}, "vs_3enc_ensemble": {}}
# white-box reference for arm B' (4-source-ensemble eval on openai ViT-L/14)
try:
    wb = pd.read_csv(f"{OUT}/rank_aeens_eval_openai_per_pair.csv")["rank_lift"].mean()
    res["_armBp_whitebox_openai"] = round(float(wb), 3)
except Exception as ex:
    res["_armBp_whitebox_openai"] = f"err:{ex}"
for eb in heldout:
    # B' (4-enc) vs A (single-enc): the headline transfer-resistance significance test
    try:
        res["vs_single_enc"][eb] = paired(
            f"{ARMA}/rank_ens_mean_attacked_openai_eval_{eb}_per_pair.csv",
            f"{OUT}/rank_aeens_eval_{eb}_per_pair.csv")
    except Exception as ex:
        res["vs_single_enc"][eb] = {"error": str(ex)}
    # B' (4-enc) vs B (3-enc): marginal lift from the 4th, cross-corpus source encoder
    try:
        res["vs_3enc_ensemble"][eb] = paired(
            f"{ARMB}/rank_aeens_eval_{eb}_per_pair.csv",
            f"{OUT}/rank_aeens_eval_{eb}_per_pair.csv")
    except Exception as ex:
        res["vs_3enc_ensemble"][eb] = {"error": str(ex)}
print("=== Stage2b 4-encoder ensemble: vs single-enc (headline) and vs 3-enc (marginal), HELD-OUT rank-lift ===")
print(json.dumps(res, indent=2))
json.dump(res, open(f"{OUT}/transfer_s2b_compare.json","w"), indent=2)
print("written", f"{OUT}/transfer_s2b_compare.json")
PY
echo "[ae2] DONE $(date -u)"
