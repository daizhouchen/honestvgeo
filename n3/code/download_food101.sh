#!/usr/bin/env bash
# download_food101.sh — fetch Food-101 via HuggingFace datasets (mirror-friendly).
#
# Resulting layout (compatible with food101_loader.py):
#   $DEST/meta/classes.txt
#   $DEST/images/<class>/<id>.jpg
set -euo pipefail
DEST="${1:-${PROJECT_ROOT}/data/food101}"
mkdir -p "$DEST"

if [[ -f "$DEST/meta/classes.txt" && -d "$DEST/images" ]]; then
  echo "[food101] already present at $DEST"
  exit 0
fi

source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-honestvgeo}
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_DATASETS_CACHE="$DEST/_hf_cache"
mkdir -p "$HF_DATASETS_CACHE"

python - <<PY
import os, json
from pathlib import Path
DEST = Path(os.environ.get("FOOD101_DEST", "${DEST}"))
DEST.mkdir(parents=True, exist_ok=True)
import datasets  # type: ignore

print("[food101] HF_ENDPOINT=", os.environ.get("HF_ENDPOINT"))
ds = datasets.load_dataset("food101", split="validation")  # validation is 25250 imgs (250/class)
print("[food101] dataset:", ds)

classes = ds.features["label"].names
(DEST / "meta").mkdir(parents=True, exist_ok=True)
(DEST / "meta/classes.txt").write_text("\n".join(classes) + "\n")
print("[food101] wrote", DEST / "meta/classes.txt", len(classes), "classes")

img_root = DEST / "images"
img_root.mkdir(parents=True, exist_ok=True)
n = 0
per_class_n = {}
for ex in ds:
    cls = classes[ex["label"]]
    cdir = img_root / cls
    cdir.mkdir(parents=True, exist_ok=True)
    n_in_cls = per_class_n.get(cls, 0)
    if n_in_cls >= 30:  # cap at 30 per class for speed; food101_loader.py picks 5 per class
        continue
    p = cdir / f"{n_in_cls:04d}.jpg"
    if not p.exists():
        ex["image"].convert("RGB").save(p, quality=95)
    per_class_n[cls] = n_in_cls + 1
    n += 1
    if n % 1000 == 0:
        print(f"[food101] wrote {n} imgs, classes complete: {sum(1 for v in per_class_n.values() if v >= 30)}/101")
print(f"[food101] total imgs written: {n}")
PY

echo "[food101] done. layout at $DEST"
