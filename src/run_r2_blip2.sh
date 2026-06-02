#!/usr/bin/env bash
# R2 SECOND-paradigm full run: BLIP-2 (Q-Former) ITM rerank over all 4 attack
# methods on the full ESCI sample. Same adv-image dirs as run_r2_blip.sh so the
# BLIP-2 numbers are apples-to-apples with the CLIP and BLIP paradigms.
# Inference-only re-score (no attack rerun). Runs inside tmux on an idle GPU.
set -uo pipefail
mkdir -p ${PROJECT_ROOT}/runs/R2_blip2_rerank
exec > ${PROJECT_ROOT}/runs/R2_blip2_rerank/r2_blip2_full.log 2>&1

source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-honestvgeo}
export HF_ENDPOINT=https://hf-mirror.com TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${R2_GPU:-6}"
cd ${PROJECT_ROOT}

MAN=data/esci1500_thumbs/manifest.csv
IMG=data/esci1500_thumbs/img
echo "R2_BLIP2_FULL_START $(date -u +%FT%TZ) gpu=$CUDA_VISIBLE_DEVICES"

run_one () {
  local m="$1"; local adv="$2"
  echo "=== METHOD $m START $(date -u +%FT%TZ) adv=$adv ==="
  python src/eval_harness_blip2.py --manifest "$MAN" --img-dir "$IMG" \
    --adv-dir "$adv" --out-dir "runs/R2_blip2_rerank/$m" --method-tag "$m" --topk 50 \
    && echo "=== METHOD $m DONE $(date -u +%FT%TZ) ===" \
    || echo "=== METHOD $m FAIL $(date -u +%FT%TZ) ==="
}

run_one cogeo    runs/T3_scaleup/run4/cogeo/img
run_one pgd_bare runs/T3_scaleup/run4/pgd_bare/img
run_one coattack runs/B2_full4method/eps16_1430/coattack/img
run_one advclip  runs/B2_full4method/eps16_1430/advclip/img

echo "R2_BLIP2_FULL_DONE $(date -u +%FT%TZ)"
