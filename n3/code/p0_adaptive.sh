#!/usr/bin/env bash
# P0 adaptive-attacker vs JPEG-q40 purification defense.
# Waits for a GENUINELY-FREE GPU (util<15% AND free_mem>14GB) so it never preempts
# a card in active use, then runs a q40-matched adaptive CoGEO attack (deterministic
# JPEG-q40 in the EOT loop) at eps=4 on OpenAI ViT-L/14, and evals the resulting adv
# both UNDEFENDED and UNDER the real jq40 purify (symmetric deployed-defense model).
set -uo pipefail
source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-honestvgeo}
export HF_ENDPOINT=https://hf-mirror.com TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

R=${PROJECT_ROOT}/n3
CODE=$R/code
MAN=$R/data/esci500_thumbs/manifest.csv
IMG=$R/data/esci500_thumbs/img
CLEAN_JQ40=$R/runs/R4_purify/clean_jq40          # purified clean gallery (reuse R4)
OUT=$R/runs/P0_adaptive
mkdir -p "$OUT"
cd "$CODE"

pick_gpu () {
python3 - <<'PY'
import subprocess, time
def snap():
    out=subprocess.check_output(["nvidia-smi","--query-gpu=index,utilization.gpu,memory.used,memory.total","--format=csv,noheader,nounits"]).decode()
    d={}
    for ln in out.strip().splitlines():
        i,util,used,tot=[int(x.strip()) for x in ln.split(",")]
        d[i]=(util,used,tot)
    return d
# A card counts as GENUINELY FREE only if (a) resident memory is tiny (<1500MiB ->
# nobody else has allocated it) AND (b) util stays <10% across 3 samples over ~6s
# (not just an instantaneous dip in someone else's bursty job). This never grabs a
# card that is in active use, per the user's instruction.
samples=[snap()]
for _ in range(2):
    time.sleep(3); samples.append(snap())
best=None
for i in samples[0]:
    used=samples[0][i][1]; tot=samples[0][i][2]; free=tot-used
    if used>=1500: continue
    if max(s[i][0] for s in samples)>=10: continue
    if free<14000: continue
    if best is None or free>best[1]: best=(i,free)
print(best[0] if best else -1)
PY
}

echo "[p0] START $(date -u); waiting for a genuinely-free GPU (util<15%, free>14GB)"
G=-1
for t in $(seq 1 360); do          # up to ~360*30s = 3h
  G=$(pick_gpu)
  if [ "$G" != "-1" ]; then echo "[p0] acquired GPU $G $(date -u)"; break; fi
  [ $((t % 10)) -eq 0 ] && echo "[p0] still waiting (tick $t) $(date -u)"
  sleep 30
done
if [ "$G" = "-1" ]; then echo "[p0] FATAL: no free GPU within wait window $(date -u)"; exit 3; fi
export CUDA_VISIBLE_DEVICES="$G"

ADV=$OUT/cogeo_adapt_jq40/img
echo "[p0] === ATTACK adaptive q40-EOT cogeo eps4 ViT-L/14 on GPU $G === $(date -u)"
COGEO_ENVSIM_JPEG_Q=40 COGEO_ENVSIM_JPEG_P=1.0 \
python n3_attack.py --method cogeo --manifest "$MAN" --img-dir "$IMG" \
  --out-dir "$OUT/cogeo_adapt_jq40" --anchor-mode product-title \
  --eps 4 --alpha 1 --iters 200 --clip-backbone ViT-L-14 --image-size 224 \
  --use-envsim --gpu 0 >"$OUT/attack_adapt.log" 2>&1
ARC=$?; echo "[p0] attack rc=$ARC $(date -u)"
if [ $ARC -ne 0 ] || [ ! -d "$ADV" ]; then echo "[p0] FATAL attack failed"; tail -20 "$OUT/attack_adapt.log"; exit 4; fi
echo "[p0] adv images: $(ls -1 "$ADV" | wc -l)"

echo "[p0] === PURIFY adaptive adv with jq40 === $(date -u)"
python purify_dir.py --src-dir "$ADV" --out-dir "$OUT/adv_adapt_jq40" --mode jpeg --param 40

BB=openai/clip-vit-large-patch14
echo "[p0] === EVAL adaptive adv UNDEFENDED === $(date -u)"
python eval_harness.py --mode rank --manifest "$MAN" --img-dir "$IMG" \
  --out-dir "$OUT" --adv-dir "$ADV" --method-tag adapt_none \
  --clip-backbone "$BB" --gpu 0 >"$OUT/eval.adapt.none.log" 2>&1
echo "[p0] eval none rc=$? $(date -u)"

echo "[p0] === EVAL adaptive adv UNDER jq40 purify (symmetric) === $(date -u)"
python eval_harness.py --mode rank --manifest "$MAN" --img-dir "$CLEAN_JQ40" \
  --out-dir "$OUT" --adv-dir "$OUT/adv_adapt_jq40" --method-tag adapt_jq40 \
  --clip-backbone "$BB" --gpu 0 >"$OUT/eval.adapt.jq40.log" 2>&1
echo "[p0] eval jq40 rc=$? $(date -u)"

echo "[p0] === RESULTS ==="
python3 - <<PY
import json,csv,collections
OUT="$OUT"
res={}
for t in ("adapt_none","adapt_jq40"):
    try:
        s=json.load(open(f"{OUT}/rank_{t}_summary.json"))
    except Exception as e:
        print(t,"MISSING summary",e); continue
    by=collections.defaultdict(list)
    try:
        with open(f"{OUT}/rank_{t}_per_pair.csv") as f:
            for row in csv.DictReader(f):
                by[row.get("esci_label","?")].append(float(row["rank_lift"]))
    except Exception as e:
        print(t,"no per_pair",e)
    coh={lab:(round(sum(v)/len(v),3) if v else None) for lab,v in by.items()}
    res[t]={"ALL":round(s["rank_lift_mean"],3),"median":s["rank_lift_median"],
            "n":s.get("n_pairs") or s.get("n"),"E":coh.get("E"),"S":coh.get("S"),"C":coh.get("C"),"I":coh.get("I")}
    print(t, res[t])
res["_baseline_nonadaptive"]={"undef_ALL":8.50,"jq40_ALL":1.98,"note":"existing E2_openai_eps4 (no EOT)"}
json.dump(res, open(f"{OUT}/p0_summary.json","w"), indent=2)
print("WROTE", f"{OUT}/p0_summary.json")
PY
echo "[p0] DONE $(date -u)"
