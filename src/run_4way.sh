#!/usr/bin/env bash
# run_4way.sh — orchestrate cogeo + pgd_bare + advclip + coattack + 4 evals
# for one (manifest, img_dir, clip_backbone, eps, image_size, run_root, gpu)
# combination.
#
# Output layout (relative to RUN_ROOT):
#   $RUN_ROOT/cogeo/{img/,attack_log.csv,attack_summary.json}
#   $RUN_ROOT/pgd_bare/...
#   $RUN_ROOT/advclip/...
#   $RUN_ROOT/coattack/...
#   $RUN_ROOT/eval/{rank_cogeo_*,rank_pgd_bare_*,rank_advclip_*,rank_coattack_*}
#   $RUN_ROOT/cell_summary.json

set -euo pipefail

CODE_DIR="${CODE_DIR:-${PROJECT_ROOT}/src}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT required}"
MANIFEST="${MANIFEST:?MANIFEST required}"
IMG_DIR="${IMG_DIR:?IMG_DIR required}"
BACKBONE="${BACKBONE:?BACKBONE required}"
EPS="${EPS:-16}"
ALPHA="${ALPHA:-4}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
ITERS="${ITERS:-200}"
GPU="${GPU:-0}"
SKIP_DONE="${SKIP_DONE:-1}"
METHODS="${METHODS:-cogeo pgd_bare advclip coattack}"

mkdir -p "$RUN_ROOT"
META="$RUN_ROOT/cell_meta.json"
cat > "$META" <<META
{
  "run_root": "$RUN_ROOT",
  "manifest": "$MANIFEST",
  "img_dir": "$IMG_DIR",
  "clip_backbone": "$BACKBONE",
  "eps": $EPS,
  "alpha": $ALPHA,
  "image_size": $IMAGE_SIZE,
  "iters": $ITERS,
  "gpu": $GPU,
  "methods": "$METHODS",
  "started_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
META

source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-honestvgeo}
export HF_ENDPOINT="https://hf-mirror.com"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="$GPU"

cd "$CODE_DIR"

run_method () {
  local m="$1"
  local odir="$RUN_ROOT/$m"
  if [[ "$SKIP_DONE" == "1" && -f "$odir/attack_summary.json" ]]; then
    echo "[run_4way] SKIP $m (already done at $odir)"
    return 0
  fi
  echo "[run_4way] === ATTACK $m === eps=$EPS bb=$BACKBONE imsize=$IMAGE_SIZE"
  case "$m" in
    cogeo|pgd_bare)
      python n3_attack.py \
        --method "$m" \
        --manifest "$MANIFEST" \
        --img-dir "$IMG_DIR" \
        --out-dir "$odir" \
        --eps "$EPS" --alpha "$ALPHA" --iters "$ITERS" \
        --clip-backbone "$BACKBONE" \
        --image-size "$IMAGE_SIZE" \
        --gpu 0
      ;;
    advclip|coattack)
      python n3_attack_baselines.py \
        --method "$m" \
        --manifest "$MANIFEST" \
        --img-dir "$IMG_DIR" \
        --out-dir "$odir" \
        --eps "$EPS" --alpha "$ALPHA" --iters "$ITERS" \
        --clip-backbone "$BACKBONE" \
        --image-size "$IMAGE_SIZE" \
        --gpu 0
      ;;
    *) echo "unknown method $m"; exit 2 ;;
  esac
}

run_eval () {
  local m="$1"
  local odir="$RUN_ROOT/$m"
  local edir="$RUN_ROOT/eval"
  mkdir -p "$edir"
  if [[ "$SKIP_DONE" == "1" && -f "$edir/rank_${m}_summary.json" ]]; then
    echo "[run_4way] SKIP eval-$m (already done)"
    return 0
  fi
  echo "[run_4way] === EVAL $m ==="
  python eval_harness.py \
    --mode rank \
    --manifest "$MANIFEST" \
    --img-dir "$IMG_DIR" \
    --out-dir "$edir" \
    --adv-dir "$odir/img" \
    --method-tag "$m" \
    --clip-backbone "$BACKBONE" \
    --gpu 0
}

for m in $METHODS; do
  run_method "$m"
done

for m in $METHODS; do
  run_eval "$m"
done

# Build a cell-level summary that joins the 4 attack summaries + 4 rank summaries
python - <<PY
import json, os
root = os.environ["RUN_ROOT"]
methods = os.environ["METHODS"].split()
out = {"run_root": root, "methods": {}}
for m in methods:
    asum_p = os.path.join(root, m, "attack_summary.json")
    esum_p = os.path.join(root, "eval", f"rank_{m}_summary.json")
    out["methods"][m] = {
        "attack_summary": json.load(open(asum_p)) if os.path.exists(asum_p) else None,
        "rank_summary": json.load(open(esum_p)) if os.path.exists(esum_p) else None,
    }
open(os.path.join(root, "cell_summary.json"), "w").write(json.dumps(out, indent=2))
print("cell summary written:", os.path.join(root, "cell_summary.json"))
PY

echo "[run_4way] DONE $RUN_ROOT"
