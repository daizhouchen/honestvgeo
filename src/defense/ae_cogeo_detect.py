#!/usr/bin/env python3
"""AE-CoGEO transfer-perturbation detectability.

Reuses the EXACT clean-calibrated training-free Laplacian flat-region detector
from b1_detector_generalization.py / detector_v2_matched.py (supp Table V/VI).
The operating threshold tau is the (1-fpr) quantile of CLEAN ESCI lap scores
only -> the detector has NO attack-family parameter, so this is the same
detector the paper already reports, just scored against a new attack family
(AE-CoGEO, the 4-encoder consensus transfer perturbation at eps=16/255).

Sanity: also re-scores CoGEO white-box adv (must reproduce the published 0.997)
so the AE-CoGEO number sits on the identical calibrated footing.
"""
import os, json
import numpy as np
from PIL import Image
try:
    from scipy.ndimage import sobel, laplace
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False
try:
    from sklearn.metrics import roc_auc_score as _sk_auc
    HAVE_SK = True
except Exception:
    HAVE_SK = False

R = "${PROJECT_ROOT}"
IMG_SIZE = 224
FPR_TARGET = 0.05
CLEAN = f"{R}/runs/T2_detector/clean224"
ATTACKS = {
    "ae_cogeo_ens4": f"{R}/runs/ae_v3_ens4/img",   # AE-CoGEO 4-encoder consensus transfer perturbation
    "cogeo_sanity":  f"{R}/runs/n3_cogeo_eps16/img" # reproduce published 0.997
}


def _load_gray(path):
    try:
        im = Image.open(path).convert("L")
        if im.size != (IMG_SIZE, IMG_SIZE):
            im = im.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
        return np.asarray(im, dtype=np.float64) / 255.0
    except Exception:
        return None


def lap_flat(path):
    a = _load_gray(path)
    if a is None:
        return np.nan
    if HAVE_SCIPY:
        gx = sobel(a, axis=0); gy = sobel(a, axis=1)
        grad = np.hypot(gx, gy); lap = laplace(a)
    else:
        gy, gx = np.gradient(a); grad = np.hypot(gx, gy)
        lap = (np.gradient(np.gradient(a, axis=0), axis=0) +
               np.gradient(np.gradient(a, axis=1), axis=1))
    thr = np.percentile(grad, 30.0)
    flat = grad <= thr
    if flat.sum() == 0:
        return float(np.mean(lap ** 2))
    return float(np.mean(lap[flat] ** 2))


def auroc(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    ok = ~np.isnan(s); s = s[ok]; y = y[ok]
    if (y == 1).sum() == 0 or (y == 0).sum() == 0:
        return float("nan")
    if HAVE_SK:
        try:
            return float(_sk_auc(y, s))
        except Exception:
            pass
    order = np.argsort(s, kind="mergesort"); ranks = np.empty(len(s), float); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[order[j + 1]] == s[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    npos = (y == 1).sum(); nneg = (y == 0).sum()
    U = ranks[y == 1].sum() - npos * (npos + 1) / 2.0
    return float(U / (npos * nneg))


def thr_at_fpr(neg, fpr=FPR_TARGET):
    neg = np.asarray(neg, float); neg = neg[~np.isnan(neg)]
    return float(np.quantile(neg, 1.0 - fpr))


def op_at_thr(neg, pos, thr):
    neg = np.asarray(neg, float); pos = np.asarray(pos, float)
    neg = neg[~np.isnan(neg)]; pos = pos[~np.isnan(pos)]
    tp = int((pos >= thr).sum()); fp = int((neg >= thr).sum()); fn = int((pos < thr).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4),
            "fpr": round(fp / len(neg), 4), "n_pos": int(len(pos)), "n_neg": int(len(neg))}


pids = sorted(p[:-4] for p in os.listdir(CLEAN) if p.endswith(".jpg"))
clean = np.array([lap_flat(f"{CLEAN}/{p}.jpg") for p in pids], float)
tau = thr_at_fpr(clean)
out = {"_meta": {"detector": "training-free Laplacian flat-region energy (clean-calibrated tau, no attack-family param)",
                 "tau_esci_clean_5pct_fpr": round(tau, 6), "n_clean": len(pids),
                 "haves": {"scipy": HAVE_SCIPY, "sklearn": HAVE_SK}},
       "clean_lap_mean": round(float(np.nanmean(clean)), 6)}
for name, advd in ATTACKS.items():
    if not os.path.isdir(advd):
        out[name] = {"error": "missing adv dir"}
        print(f"[{name}] MISSING {advd}")
        continue
    apids = [p for p in pids if os.path.exists(f"{advd}/{p}.jpg")]
    adv = np.array([lap_flat(f"{advd}/{p}.jpg") for p in apids], float)
    labels = np.r_[np.zeros(len(clean)), np.ones(len(adv))]
    scores = np.r_[clean, adv]
    au = auroc(scores, labels)
    o = op_at_thr(clean, adv, tau)
    out[name] = {"auroc": round(au, 4), "op_at_tau_esci": o, "n_adv": len(apids),
                 "adv_lap_mean": round(float(np.nanmean(adv)), 6)}
    print(f"[{name}] AUROC={au:.4f} recall@5%FPR={o['recall']} precision={o['precision']} fpr={o['fpr']} n_adv={len(apids)}")
OUTP = f"{R}/runs/ae_cogeo_detectability/ae_cogeo_detect.json"
os.makedirs(os.path.dirname(OUTP), exist_ok=True)
json.dump(out, open(OUTP, "w"), indent=2)
print("WROTE", OUTP)
print(json.dumps(out, indent=2))
