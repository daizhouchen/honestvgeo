#!/usr/bin/env bash
# run_c_transfer.sh — C transfer attack: re-evaluate adversarial images
# optimized against one CLIP backbone using a DIFFERENT CLIP backbone for
# retrieval scoring. This is pure re-evaluation (no new attack), so each
# cell costs only the time to CLIP-encode 491 images + 491 queries.
#
# Per (ATTACK_BB, EVAL_BB, METHOD) cell:
#   adv-dir comes from the matrix at the top of this script (the EXISTING
#   attack outputs); eval is run with --clip-backbone EVAL_BB.
#
# Output:
#   $TRANSFER_ROOT/${ATTACK_BB}_to_${EVAL_BB}/rank_${METHOD}_per_pair.csv
#   $TRANSFER_ROOT/${ATTACK_BB}_to_${EVAL_BB}/rank_${METHOD}_summary.json
#   $TRANSFER_ROOT/cell_summary.json (aggregate)
#
# Inputs (env):
#   TRANSFER_ROOT  output root (default runs/C_transfer)
#   MANIFEST       ESCI manifest.csv
#   IMG_DIR        original image dir
#   GPU            CUDA index
#   METHODS        space-separated method list (default "cogeo pgd_bare")

set -euo pipefail

CODE_DIR="${CODE_DIR:-${PROJECT_ROOT}/src}"
RUNS_ROOT="${RUNS_ROOT:-${PROJECT_ROOT}/runs}"
TRANSFER_ROOT="${TRANSFER_ROOT:-$RUNS_ROOT/C_transfer}"
MANIFEST="${MANIFEST:?MANIFEST required}"
IMG_DIR="${IMG_DIR:?IMG_DIR required}"
GPU="${GPU:-0}"
METHODS="${METHODS:-cogeo pgd_bare advclip coattack}"
SKIP_DONE="${SKIP_DONE:-1}"

mkdir -p "$TRANSFER_ROOT"

# Lookup table: (ATTACK_BB -> {METHOD -> adv-image-dir}).
# OpenAI ViT-L/14 attack outputs sit in separate per-method roots
# (n3_<method>_eps16/img); laion2b / eva02 sit under E1_/E5_<bb>/<method>/img;
# the 3 additional backbones (vith14, vitb32, siglip) sit under
# F<n>_<bb>_eps16/<method>/img populated by run_baselines_new_bb.sh.
adv_dir_for () {
  local bb="$1"; local m="$2"
  case "$bb" in
    openai)
      echo "$RUNS_ROOT/n3_${m}_eps16/img"
      ;;
    laion2b)
      echo "$RUNS_ROOT/E1_laion2b_eps16/${m}/img"
      ;;
    eva02)
      echo "$RUNS_ROOT/E5_eva02_eps16/${m}/img"
      ;;
    vith14)
      echo "$RUNS_ROOT/F1_vith14_eps16/${m}/img"
      ;;
    vitb32)
      echo "$RUNS_ROOT/F2_vitb32_eps16/${m}/img"
      ;;
    siglip)
      echo "$RUNS_ROOT/F3_siglip_eps16/${m}/img"
      ;;
    *)
      echo "" ; return 1 ;;
  esac
}

# Map short backbone tag to the n3_attack.py / eval_harness.py registry name.
bb_name () {
  case "$1" in
    openai)  echo "openai/clip-vit-large-patch14" ;;
    laion2b) echo "ViT-L-14-laion2b" ;;
    eva02)   echo "EVA02-L-14" ;;
    vith14)  echo "ViT-H-14-laion2b" ;;
    vitb32)  echo "ViT-B-32-laion2b" ;;
    siglip)  echo "ViT-B-16-SigLIP-webli" ;;
    *) echo "" ; return 1 ;;
  esac
}

source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-honestvgeo}
export HF_ENDPOINT="https://hf-mirror.com"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="$GPU"

cd "$CODE_DIR"

BACKBONES=(openai laion2b eva02 vith14 vitb32 siglip)

for ATTACK_BB in "${BACKBONES[@]}"; do
  for EVAL_BB in "${BACKBONES[@]}"; do
    if [[ "$ATTACK_BB" == "$EVAL_BB" ]]; then continue; fi
    cell_dir="$TRANSFER_ROOT/${ATTACK_BB}_to_${EVAL_BB}"
    mkdir -p "$cell_dir"
    EVAL_BB_NAME="$(bb_name "$EVAL_BB")"
    for m in $METHODS; do
      adv="$(adv_dir_for "$ATTACK_BB" "$m")"
      if [[ -z "$adv" || ! -d "$adv" ]]; then
        echo "[c_transfer] SKIP $ATTACK_BB->$EVAL_BB/$m (adv-dir missing: $adv)"
        continue
      fi
      tag="${m}_attacked_${ATTACK_BB}_eval_${EVAL_BB}"
      sum_path="$cell_dir/rank_${tag}_summary.json"
      if [[ "$SKIP_DONE" == "1" && -f "$sum_path" ]]; then
        echo "[c_transfer] SKIP $ATTACK_BB->$EVAL_BB/$m (done at $sum_path)"
        continue
      fi
      echo "[c_transfer] === $ATTACK_BB -> $EVAL_BB / $m ==="
      python eval_harness.py \
        --mode rank \
        --manifest "$MANIFEST" \
        --img-dir "$IMG_DIR" \
        --out-dir "$cell_dir" \
        --adv-dir "$adv" \
        --method-tag "$tag" \
        --clip-backbone "$EVAL_BB_NAME" \
        --gpu 0
    done
  done
done

# Aggregate.
TRANSFER_ROOT="$TRANSFER_ROOT" python - <<'PY'
import json, os, glob
root = os.environ["TRANSFER_ROOT"]
cells = {}
for sum_path in sorted(glob.glob(os.path.join(root, "*_to_*", "rank_*_summary.json"))):
    cell_name = os.path.basename(os.path.dirname(sum_path))
    method_tag = os.path.basename(sum_path).replace("rank_", "").replace("_summary.json", "")
    cells.setdefault(cell_name, {})[method_tag] = json.load(open(sum_path))
open(os.path.join(root, "cell_summary.json"), "w").write(json.dumps(cells, indent=2))
print("transfer cell summary written:", os.path.join(root, "cell_summary.json"), "cells:", len(cells))
PY

echo "[c_transfer] DONE $TRANSFER_ROOT"
