#!/bin/bash
# AE-CoGEO reach-higher: Anchor-Ensemble CoGEO vs faithful single-title CoGEO.
# Identical optimizer (n3_attack_ae.py), identical eps/seed/iters/env-sim.
# The ONLY difference between arms is the anchor set (single title vs query-free
# title-derived ensemble). Query-free: every anchor depends only on x_p's title.
#
# Env knobs:
#   ARMS    space-separated arm keys among: title ens_mean ens_softmin   (default "title ens_mean")
#   MAXIMG  cap images for a smoke (default empty = all 491)
#   GPU     CUDA index (default 5)
#   ENVSIM  1 to apply CoGEO env-sim (default 1)
#   OUT_TAG output subdir under runs/ (default ae_v1)
#   EPS ALPHA ITERS BB MAXANCH TAU  (defaults 16 4 200 ViT-L-14 6 8.0)
set -uo pipefail
source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-honestvgeo}
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${GPU:-5}"

R=${PROJECT_ROOT}
CODE=$R/code
OUT_TAG=${OUT_TAG:-ae_v1}
OUT=$R/runs/$OUT_TAG
mkdir -p "$OUT/eval"
MAN=${MAN:-$R/data/esci500_thumbs/manifest.csv}
IMG=${IMG:-$R/data/esci500_thumbs/img}
BB=${BB:-openai/clip-vit-large-patch14}
EPS=${EPS:-16}; ALPHA=${ALPHA:-4}; ITERS=${ITERS:-200}
MAXANCH=${MAXANCH:-6}; TAU=${TAU:-8.0}
ENVSIM=${ENVSIM:-1}
ARMS=${ARMS:-"title ens_mean"}
ARMS=${ARMS//,/ }
MAXIMG=${MAXIMG:-}
LOG=$OUT/ae.log
ESFLAG=""; [ "$ENVSIM" = "1" ] && ESFLAG="--use-envsim"
MAXARG=""; [ -n "$MAXIMG" ] && MAXARG="--max-images $MAXIMG"

echo "[ae] START $(date -u) tag=$OUT_TAG arms='$ARMS' maximg=${MAXIMG:-all} gpu=${GPU:-5} envsim=$ENVSIM eps=$EPS iters=$ITERS" | tee -a "$LOG"
cd "$CODE"

arm_args () { # arm-key -> n3_attack_ae.py mode args
  case "$1" in
    title)       echo "--anchor-mode title" ;;
    ens_mean)    echo "--anchor-mode ensemble --ensemble-agg mean --max-anchors $MAXANCH" ;;
    ens_softmin) echo "--anchor-mode ensemble --ensemble-agg softmin --max-anchors $MAXANCH --tau $TAU" ;;
    *) echo "BAD_ARM" ;;
  esac
}

for arm in $ARMS; do
  AARGS=$(arm_args "$arm")
  if [ "$AARGS" = "BAD_ARM" ]; then echo "[ae] skip bad arm '$arm'" | tee -a "$LOG"; continue; fi
  odir="$OUT/$arm"
  echo "[ae] === ATTACK arm=$arm $(date -u) ($AARGS) ===" | tee -a "$LOG"
  python n3_attack_ae.py $AARGS \
    --manifest "$MAN" --img-dir "$IMG" --out-dir "$odir" \
    --eps "$EPS" --alpha "$ALPHA" --iters "$ITERS" \
    --clip-backbone "$BB" --image-size 224 --gpu 0 --seed 42 $ESFLAG $MAXARG \
    >>"$OUT/$arm.attack.log" 2>&1
  echo "[ae] attack arm=$arm rc=$? $(date -u)" | tee -a "$LOG"
  echo "[ae] === EVAL arm=$arm $(date -u) ===" | tee -a "$LOG"
  python eval_harness.py --mode rank \
    --manifest "$MAN" --img-dir "$IMG" --out-dir "$OUT/eval" \
    --adv-dir "$odir/img" --method-tag "$arm" \
    --clip-backbone "$BB" --gpu 0 >>"$OUT/$arm.eval.log" 2>&1
  echo "[ae] eval arm=$arm rc=$? $(date -u)" | tee -a "$LOG"
done

echo "[ae] === COMPARE $(date -u) ===" | tee -a "$LOG"
python - "$OUT" "$ARMS" <<'PY' | tee -a "$LOG"
import sys, json, numpy as np, pandas as pd
OUT=sys.argv[1]; arms=sys.argv[2].split(); ev=f"{OUT}/eval"
try:
    from scipy.stats import wilcoxon
    HAVE_SCIPY=True
except Exception:
    HAVE_SCIPY=False
def load(tag):
    return pd.read_csv(f"{ev}/rank_{tag}_per_pair.csv")
def cohort(df):
    out={}
    for lab in ["E","S","C","I"]:
        m=df[df["esci_label"]==lab]
        out[lab]=[round(float(m["rank_lift"].mean()),3), int(len(m))] if len(m) else [None,0]
    out["ALL"]=[round(float(df["rank_lift"].mean()),3), int(len(df))]
    return out
res={}; frames={}
for a in arms:
    try:
        d=load(a); frames[a]=d; res[a]=cohort(d)
    except Exception as e:
        res[a]={"error":str(e)}
print("=== AE per-cohort rank_lift (mean, n) ==="); print(json.dumps(res, indent=2))
# paired comparison of each ensemble arm vs the title arm, aligned on (product_id,query_id)
cmp={}
if "title" in frames:
    base=frames["title"].set_index(["product_id","query_id"])["rank_lift"]
    for a in arms:
        if a=="title" or a not in frames: continue
        cur=frames[a].set_index(["product_id","query_id"])["rank_lift"]
        idx=base.index.intersection(cur.index)
        b=base.loc[idx].to_numpy(float); c=cur.loc[idx].to_numpy(float)
        diff=c-b; n=len(diff)
        rng=np.random.RandomState(0)
        boot=np.array([rng.choice(diff,size=n,replace=True).mean() for _ in range(2000)]) if n>0 else np.array([0.0])
        entry={"n_pairs":int(n),
               "title_mean":round(float(b.mean()),3),"arm_mean":round(float(c.mean()),3),
               "mean_diff":round(float(diff.mean()),3),
               "median_diff":round(float(np.median(diff)),3),
               "boot_ci95":[round(float(np.percentile(boot,2.5)),3),round(float(np.percentile(boot,97.5)),3)],
               "win_rate":round(float((diff>0).mean()),3),"tie_rate":round(float((diff==0).mean()),3)}
        if HAVE_SCIPY and n>0 and np.any(diff!=0):
            try:
                st,p=wilcoxon(c,b); entry["wilcoxon_p"]=round(float(p),5)
            except Exception as e:
                entry["wilcoxon_p"]=f"err:{e}"
        cmp[a]=entry
print("=== ensemble arm vs title (paired) ==="); print(json.dumps(cmp, indent=2))
out={"per_cohort":res,"paired_vs_title":cmp,"cogeo_491_headline_ref":19.86,"have_scipy":HAVE_SCIPY}
json.dump(out, open(f"{OUT}/ae_compare.json","w"), indent=2)
print(f"written {OUT}/ae_compare.json")
PY
echo "[ae] ALL DONE $(date -u)" | tee -a "$LOG"
