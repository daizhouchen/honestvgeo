#!/usr/bin/env bash
# R2 cross-paradigm AE-CoGEO cell: re-score the AE-CoGEO transfer adversarial
# images (ensemble-of-4-CLIP-surrogates, eps16, 491 imgs) through the BLIP
# fused-encoder ITM rerank retriever. The attacker never accessed BLIP, so this
# measures transfer to an unseen, architecturally-different retrieval paradigm.
# Anchors already in the paper for the SAME images: white-box source (OpenAI)
# rank-lift 18.5; held-out CLIP encoders 3.3-8.6. This adds the BLIP point.
set -uo pipefail
cd ${PROJECT_ROOT}
source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-honestvgeo}
export HF_ENDPOINT=https://hf-mirror.com TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${R2_GPU:-0}"

OUT=n3/runs/R2_blip_rerank_ae
mkdir -p "$OUT"
exec > "$OUT/r2_ae.log" 2>&1

IMG=n3/data/esci500_thumbs/img
AE=n3/runs/ae_v3_ens4/img
MM="$OUT/manifest_ae.csv"

echo "R2_AE_START $(date -u +%FT%TZ) gpu=$CUDA_VISIBLE_DEVICES"

# Keep only product_ids that actually have an AE-CoGEO adversarial image.
python - <<'PY'
import pandas as pd, os
df = pd.read_csv("n3/data/esci500_thumbs/manifest.csv")
ae = "n3/runs/ae_v3_ens4/img"
m = df[df["product_id"].apply(lambda a: os.path.exists(os.path.join(ae, f"{a}.jpg")))].copy()
m.to_csv("n3/runs/R2_blip_rerank_ae/manifest_ae.csv", index=False)
print("ae_rows", len(m), "of", len(df))
PY

echo "=== AE-CoGEO (transfer) on BLIP $(date -u +%FT%TZ) ==="
python n3/code/eval_harness_blip.py --manifest "$MM" --img-dir "$IMG" \
  --adv-dir "$AE" --out-dir "$OUT/ae_cogeo" --method-tag ae_cogeo --topk 50 --gpu 0 \
  && echo "=== AE_DONE $(date -u +%FT%TZ) ===" || echo "=== AE_FAIL $(date -u +%FT%TZ) ==="

echo "R2_AE_DONE $(date -u +%FT%TZ)"
