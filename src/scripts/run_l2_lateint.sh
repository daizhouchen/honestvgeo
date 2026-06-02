#!/usr/bin/env bash
SRC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"  # repo src/ root
# L2 full cross-paradigm run: ColBERT-style late-interaction MaxSim rerank over all
# 4 attack methods on the full ESCI-1430 sample, on a HELD-OUT OpenCLIP ViT-B/32
# (laion2b) encoder. Each run also reports the global-cosine control on the SAME
# encoder+pool, so the only difference is the scoring paradigm. CPU-ONLY: never
# touches a GPU (honors "do not grab cards in use"). Runs under nohup.
set -uo pipefail
H=${PROJECT_ROOT}
OUT="$H/runs/L2_lateint_rerank"
mkdir -p "$OUT"
exec > "$OUT/l2_full.log" 2>&1

source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-honestvgeo}
export HF_ENDPOINT=https://hf-mirror.com TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""        # CPU only, no GPU contention
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
cd "$H"

MAN=data/esci1500_thumbs/manifest.csv
IMG=data/esci1500_thumbs/img
MODEL="ViT-B-32"; PRE="laion2b_s34b_b79k"
echo "L2_FULL_START $(date -u +%FT%TZ) cpu model=$MODEL/$PRE threads=$OMP_NUM_THREADS"

run_one () {
  local m="$1"; local adv="$2"
  echo "=== METHOD $m START $(date -u +%FT%TZ) adv=$adv ==="
  python "$SRC_ROOT/eval/eval_harness_lateint.py" --manifest "$MAN" --img-dir "$IMG" \
    --adv-dir "$adv" --out-dir "$OUT/$m" --method-tag "$m" \
    --clip-model "$MODEL" --pretrained "$PRE" \
    && echo "=== METHOD $m DONE $(date -u +%FT%TZ) ===" \
    || echo "=== METHOD $m FAIL $(date -u +%FT%TZ) ==="
}

run_one cogeo    runs/T3_scaleup/run4/cogeo/img
run_one pgd_bare runs/T3_scaleup/run4/pgd_bare/img
run_one coattack runs/B2_full4method/eps16_1430/coattack/img
run_one advclip  runs/B2_full4method/eps16_1430/advclip/img

echo "L2_FULL_DONE $(date -u +%FT%TZ)"
