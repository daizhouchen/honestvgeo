#!/usr/bin/env bash
# R5 scaling-law: 2-source-encoder AE-CoGEO (nested subset of the 3/4-source sets).
#   SOURCE ensemble : OpenAI ViT-L/14 + ViT-B/32   (subset of ae_v2_ens=3src / ae_v3_ens4=4src)
#   HELD-OUT eval   : LAION ViT-L/14, LAION ViT-H/14, LAION ViT-B/32, SigLIP  (none in source)
# Produces rank_aeens_eval_{laion2b,vith14,vitb32,siglip}_per_pair.csv, matching n=3/n=4 naming,
# so a clean monotone #source -> held-out rank-lift curve can be aggregated across n in {1,2,3,4}.
# Env knobs: GPU (default 0), ITERS (default 200, matches the existing arms), TAG (default ae_s2_ens2).
set -uo pipefail
source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-honestvgeo}
export HF_ENDPOINT=https://hf-mirror.com TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
R=${PROJECT_ROOT}
CODE=$R/code
MAN=$R/data/esci500_thumbs/manifest.csv
IMG=$R/data/esci500_thumbs/img
TAG=${TAG:-ae_s2_ens2}
OUT=$R/runs/$TAG
ITERS=${ITERS:-200}
SRC="openai/clip-vit-large-patch14,ViT-B-32"
mkdir -p "$OUT"
cd "$CODE"
echo "[r5s2] START $(date -u) tag=$TAG gpu=${GPU:-0} iters=$ITERS src=$SRC"

echo "[r5s2] === ATTACK (2-encoder ensemble) $(date -u) ==="
python n3_attack_ae_ens.py --clip-backbones "$SRC" \
  --anchor-mode ensemble --ensemble-agg mean --max-anchors 6 \
  --manifest "$MAN" --img-dir "$IMG" --out-dir "$OUT" \
  --eps 16 --alpha 4 --iters "$ITERS" --image-size 224 --gpu 0 --seed 42 --use-envsim \
  >>"$OUT/attack.log" 2>&1
echo "[r5s2] attack rc=$? $(date -u)"

declare -A BBN=( [openai]=openai/clip-vit-large-patch14 [laion2b]=ViT-L-14-laion2b [vith14]=ViT-H-14-laion2b [vitb32]=ViT-B-32-laion2b [siglip]=ViT-B-16-SigLIP-webli )
for eb in openai laion2b vith14 vitb32 siglip; do
  tag="aeens_eval_${eb}"
  echo "[r5s2] === EVAL -> $eb (${BBN[$eb]}) $(date -u) ==="
  python eval_harness.py --mode rank --manifest "$MAN" --img-dir "$IMG" \
    --out-dir "$OUT" --adv-dir "$OUT/img" --method-tag "$tag" \
    --clip-backbone "${BBN[$eb]}" --gpu 0 >>"$OUT/eval.$eb.log" 2>&1
  echo "[r5s2] eval $eb rc=$? $(date -u)"
done
echo "[r5s2] DONE $(date -u)"
