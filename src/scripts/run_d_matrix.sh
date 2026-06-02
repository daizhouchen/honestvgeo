#!/usr/bin/env bash
# run_d_matrix.sh — D ablation matrix (VLM × CLIP backbone) launcher.
#
# For each VLM in $VLMS (default: qwen25 qwen2 llava blip2 blip)
# For each CLIP backbone in $BACKBONES (default: openai laion2b eva02)
#   Run CoGEO + eval (single cell). Skips if cell already complete.
#
# Cells run sequentially on $GPU (single-GPU caller); to parallelize,
# launch one process per GPU with non-overlapping (VLMS, BACKBONES) slices.
#
# Outputs: $D_ROOT/$vlm_$bb/{cogeo/, eval/, cell_meta.json, cell_summary.json}

set -euo pipefail

CODE_DIR="${CODE_DIR:-${PROJECT_ROOT}/src}"
RUNS_ROOT="${RUNS_ROOT:-${PROJECT_ROOT}/runs}"
D_ROOT="${D_ROOT:-$RUNS_ROOT/D_vlm}"
CAP_DIR="${CAP_DIR:-${PROJECT_ROOT}/data/vlm_captions}"
MANIFEST="${MANIFEST:-${PROJECT_ROOT}/data/esci500_thumbs/manifest.csv}"
IMG_DIR="${IMG_DIR:-${PROJECT_ROOT}/data/esci500_thumbs/img}"
GPU="${GPU:-0}"
EPS="${EPS:-16}"
ITERS="${ITERS:-200}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
VLMS="${VLMS:-qwen25 qwen2 llava blip2 blip}"
BACKBONES="${BACKBONES:-openai laion2b eva02 vith14 vitb32 siglip}"
SKIP_DONE="${SKIP_DONE:-1}"

bb_name () {
  case "$1" in
    openai)  echo "openai/clip-vit-large-patch14" ;;
    laion2b) echo "ViT-L-14-laion2b" ;;
    eva02)   echo "EVA02-L-14" ;;
    vith14)  echo "ViT-H-14-laion2b" ;;
    vitb32)  echo "ViT-B-32-laion2b" ;;
    siglip)  echo "ViT-B-16-SigLIP-webli" ;;
    *) echo ""; return 1 ;;
  esac
}

mkdir -p "$D_ROOT"

for vlm in $VLMS; do
  cap_json="$CAP_DIR/${vlm}.json"
  if [[ ! -f "$cap_json" ]]; then
    echo "[d_matrix] SKIP $vlm (captions json missing: $cap_json)"
    continue
  fi
  for bb in $BACKBONES; do
    cell="$D_ROOT/${vlm}_${bb}"
    sum_path="$cell/cell_summary.json"
    if [[ "$SKIP_DONE" == "1" && -f "$sum_path" ]]; then
      echo "[d_matrix] SKIP $vlm/$bb (cell done)"
      continue
    fi
    BACKBONE="$(bb_name $bb)"
    echo "[d_matrix] === $vlm × $bb ==="
    VLM="$vlm" CAPTIONS_JSON="$cap_json" BACKBONE="$BACKBONE" \
      RUN_ROOT="$cell" MANIFEST="$MANIFEST" IMG_DIR="$IMG_DIR" \
      GPU="$GPU" EPS="$EPS" ITERS="$ITERS" IMAGE_SIZE="$IMAGE_SIZE" \
      bash run_d_vlm.sh
  done
done

D_ROOT="$D_ROOT" VLMS="$VLMS" BACKBONES="$BACKBONES" python - <<'PY'
import json, os, glob
root = os.environ["D_ROOT"]
vlms = os.environ["VLMS"].split()
bbs = os.environ["BACKBONES"].split()
matrix = {}
for v in vlms:
    matrix[v] = {}
    for b in bbs:
        cell_path = os.path.join(root, f"{v}_{b}", "cell_summary.json")
        if os.path.exists(cell_path):
            matrix[v][b] = json.load(open(cell_path))
        else:
            matrix[v][b] = None
open(os.path.join(root, "matrix_summary.json"), "w").write(json.dumps(matrix, indent=2))
print("d matrix summary written:", os.path.join(root, "matrix_summary.json"))
PY

echo "[d_matrix] DONE $D_ROOT"
