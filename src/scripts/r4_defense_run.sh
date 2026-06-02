#!/usr/bin/env bash
SRC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"  # repo src/ root
# R4: working input-purification defense in the detector-weak regimes.
# Reuses the EXACT eval_harness.py rank-lift scorer (rank_lift = rank_orig - rank_adv).
#
# DEPLOYED-DEFENSE MODEL (symmetric purification): a real platform purifies EVERY
# indexed image. So the defended attack-lift is measured in a uniformly-purified
# index: rank_orig uses the PURIFIED clean gallery, rank_adv uses the PURIFIED adv.
#   undefended (none)  : --img-dir = pristine clean ($IMG),  --adv-dir = adv
#   defended (D)        : --img-dir = D(clean gallery),       --adv-dir = D(adv)
# Defense efficacy = undefended rank_lift - defended rank_lift  (positive = suppressed).
# Clean-retrieval cost is measured separately: rank shift of legitimate products when
# their own image is purified (pristine gallery vs purified candidate).
#
# Regimes (where the paper's detector is weak):
#   A) TRANSFER     : AE-CoGEO 4-encoder consensus (eps16), eval on 4 HELD-OUT encoders
#   B) SMALL-BUDGET : white-box CoGEO eps=4 on OpenAI ViT-L/14
# Defenses are OUTSIDE the attack's EnvSim EOT envelope (DIM +/-10%, Gaussian blur, JPEG q85).
#
# Env knobs: GPU (default 2), SMOKE=1 -> only bd4 on transfer/laion2b + eps4/openai.
set -uo pipefail
source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-honestvgeo}
export HF_ENDPOINT=https://hf-mirror.com TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${GPU:-2}"

R=${PROJECT_ROOT}
CODE=$R/code
MAN=$R/data/esci500_thumbs/manifest.csv
IMG=$R/data/esci500_thumbs/img
OUT=$R/runs/R4_purify
mkdir -p "$OUT"
cd "$CODE"

declare -A BBN=( [openai]=openai/clip-vit-large-patch14 [laion2b]=ViT-L-14-laion2b \
                 [vith14]=ViT-H-14-laion2b [vitb32]=ViT-B-32-laion2b [siglip]=ViT-B-16-SigLIP-webli )

# defense settings: "mode:param:tag"   (none = undefended baseline)
if [ "${SMOKE:-0}" = "1" ]; then
  DEFENSES=( "none:0:none" "bitdepth:4:bd4" )
  TRANSFER_EB=( laion2b )
  CLEAN_EB=( openai )
else
  DEFENSES=( "none:0:none" "bitdepth:5:bd5" "bitdepth:4:bd4" "resize:0.5:rs50" "jpeg:40:jq40" "median:3:md3" )
  TRANSFER_EB=( laion2b vith14 vitb32 siglip )
  CLEAN_EB=( openai laion2b )
fi

purify () { python "$SRC_ROOT/defense/purify_dir.py" --src-dir "$1" --out-dir "$2" --mode "$3" --param "$4"; }

echo "[r4] START $(date -u) gpu=${GPU:-2} smoke=${SMOKE:-0} out=$OUT"

ADV_T=$R/runs/ae_v3_ens4/img        # TRANSFER source (eps16, 4-enc consensus)
ADV_B=$R/runs/E2_openai_eps4/cogeo/img    # SMALL-BUDGET source (eps4 white-box openai CoGEO)

for d in "${DEFENSES[@]}"; do
  IFS=: read -r mode param tag <<< "$d"
  if [ "$mode" = "none" ]; then
    PT="$ADV_T"; PB="$ADV_B"; IMGDIR="$IMG"          # undefended: pristine gallery
  else
    PT=$OUT/adv_transfer_$tag; PB=$OUT/adv_eps4_$tag; IMGDIR=$OUT/clean_$tag
    purify "$ADV_T" "$PT"     "$mode" "$param"        # purify transfer adv
    purify "$ADV_B" "$PB"     "$mode" "$param"        # purify eps4 adv
    purify "$IMG"   "$IMGDIR" "$mode" "$param"        # purify clean gallery (deployed defense)
  fi
  # ---- A) TRANSFER on held-out encoders ----
  for eb in "${TRANSFER_EB[@]}"; do
    echo "[r4] transfer eb=$eb def=$tag $(date -u)"
    python "$SRC_ROOT/eval/eval_harness.py" --mode rank --manifest "$MAN" --img-dir "$IMGDIR" \
      --out-dir "$OUT" --adv-dir "$PT" --method-tag "transfer_${eb}_${tag}" \
      --clip-backbone "${BBN[$eb]}" --gpu 0 >>"$OUT/eval.transfer.$eb.$tag.log" 2>&1
    echo "[r4] transfer eb=$eb def=$tag rc=$? $(date -u)"
  done
  # ---- B) SMALL-BUDGET white-box (openai eps4) ----
  echo "[r4] eps4 openai def=$tag $(date -u)"
  python "$SRC_ROOT/eval/eval_harness.py" --mode rank --manifest "$MAN" --img-dir "$IMGDIR" \
    --out-dir "$OUT" --adv-dir "$PB" --method-tag "eps4_openai_${tag}" \
    --clip-backbone "${BBN[openai]}" --gpu 0 >>"$OUT/eval.eps4.$tag.log" 2>&1
  echo "[r4] eps4 openai def=$tag rc=$? $(date -u)"
done

# ---- Clean-retrieval cost: legitimate product image purified, pristine gallery baseline ----
# rank_lift here = rank(pristine clean) - rank(purified clean); ~0 = benign, negative = hurts retrieval.
for d in "${DEFENSES[@]}"; do
  IFS=: read -r mode param tag <<< "$d"
  [ "$mode" = "none" ] && continue
  PC=$OUT/clean_$tag      # already built above
  for eb in "${CLEAN_EB[@]}"; do
    echo "[r4] cleancost eb=$eb def=$tag $(date -u)"
    python "$SRC_ROOT/eval/eval_harness.py" --mode rank --manifest "$MAN" --img-dir "$IMG" \
      --out-dir "$OUT" --adv-dir "$PC" --method-tag "clean_${eb}_${tag}" \
      --clip-backbone "${BBN[$eb]}" --gpu 0 >>"$OUT/eval.clean.$eb.$tag.log" 2>&1
    echo "[r4] cleancost eb=$eb def=$tag rc=$? $(date -u)"
  done
done

echo "[r4] === AGGREGATE $(date -u) ==="
python "$SRC_ROOT/analysis/r4_aggregate.py" "$OUT"
echo "[r4] DONE $(date -u)"
