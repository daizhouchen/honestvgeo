#!/usr/bin/env python3
"""B1 -- Detector generalization (campaign analysis-45b62125).

Question: does the training-free global Laplacian flat-region detector
(supp Table V, same-manifest AUROC 0.993-0.997 at eps=16) generalize beyond the
manifest it was characterised on, or is the near-perfect AUROC a same-manifest
artefact?

This slice reuses ONLY already-measured adversarial images (CPU; no GPU; the
remote GPUs are busy with B2). The detector is unchanged from
analysis-47bea597/detector_v2_matched.py: per-image Laplacian energy in
low-gradient (flat) regions; higher score => more adversarial; the operating
threshold tau is the (1 - fpr) quantile of CLEAN scores only -- the detector has
NO attack-specific parameters, so it cannot overfit to an attack family by
construction. The interesting generalisation axes are therefore (B) a held-out
image DISTRIBUTION and (C) a held-out perturbation BUDGET.

Three held-out probes, plus a same-manifest baseline reproduction:
  Baseline : ESCI eps16, clean224 vs each of 4 attack families (reproduce 0.99x).
  Probe A  : held-out attack family (leave-one-attack-out). Because tau is
             clean-only, each family's AUROC + recall@tau_ESCI IS its held-out
             number; we report all four and the clean-calibrated operating point.
  Probe B  : held-out DISTRIBUTION = Food-101 (food photos, a different image
             domain). Calibrate tau on ESCI clean; (1) realised FPR when that
             tau is transferred to Food-101 clean (operating-point transfer);
             (2) recall@tau_ESCI on Food-101 advs; (3) intrinsic Food-101 AUROC;
             (4) tau re-calibrated on Food-101 clean, for the transfer gap.
  Probe C  : held-out perturbation budget = eps in {4,8} (detector characterised
             at eps16). AUROC + recall@tau_ESCI for cogeo & pgd_bare. Expected
             degradation at smaller eps is the honest generalisation limit and
             reconciles with the eps-sweep (attack efficacy itself collapses at
             eps4).

Outputs: {OUT}/b1_detector_generalization.json (+ stdout summary).
"""
import os, csv, json
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
OUT = f"{R}/runs/B1_detector_generalization"
os.makedirs(OUT, exist_ok=True)
IMG_SIZE = 224
JPEG_Q = 95
FPR_TARGET = 0.05

# ---- detector functions (identical to detector_v2_matched.py) ----
_img_cache = {}
def _load_gray(path):
    if path in _img_cache:
        return _img_cache[path]
    try:
        im = Image.open(path).convert("L")
        if im.size != (IMG_SIZE, IMG_SIZE):
            im = im.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
        a = np.asarray(im, dtype=np.float64) / 255.0
    except Exception:
        a = None
    _img_cache[path] = a
    return a

def fft_highfreq(path):
    a = _load_gray(path)
    if a is None: return np.nan
    F = np.fft.fftshift(np.fft.fft2(a)); mag = np.abs(F)
    h, w = mag.shape; cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    hf = r > 0.5 * min(cy, cx)
    return float(mag[hf].sum() / (mag.sum() + 1e-12))

def lap_flat(path):
    a = _load_gray(path)
    if a is None: return np.nan
    if HAVE_SCIPY:
        gx = sobel(a, axis=0); gy = sobel(a, axis=1)
        grad = np.hypot(gx, gy); lap = laplace(a)
    else:
        gy, gx = np.gradient(a); grad = np.hypot(gx, gy)
        lap = (np.gradient(np.gradient(a, axis=0), axis=0) +
               np.gradient(np.gradient(a, axis=1), axis=1))
    thr = np.percentile(grad, 30.0)
    flat = grad <= thr
    if flat.sum() == 0: return float(np.mean(lap ** 2))
    return float(np.mean(lap[flat] ** 2))

def auroc(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    ok = ~np.isnan(s); s = s[ok]; y = y[ok]
    if (y == 1).sum() == 0 or (y == 0).sum() == 0: return float("nan")
    if HAVE_SK:
        try: return float(_sk_auc(y, s))
        except Exception: pass
    order = np.argsort(s, kind="mergesort"); ranks = np.empty(len(s), float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[order[j + 1]] == s[order[i]]: j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1): ranks[order[k]] = avg
        i = j + 1
    npos = (y == 1).sum(); nneg = (y == 0).sum()
    U = ranks[y == 1].sum() - npos * (npos + 1) / 2.0
    return float(U / (npos * nneg))

def op_at_thr(neg_scores, pos_scores, thr):
    """precision/recall/fpr of decision rule (score >= thr) given fixed thr."""
    neg = np.asarray(neg_scores, float); pos = np.asarray(pos_scores, float)
    neg = neg[~np.isnan(neg)]; pos = pos[~np.isnan(pos)]
    if len(neg) == 0 or len(pos) == 0:
        return {"precision": None, "recall": None, "fpr": None, "n_pos": int(len(pos)), "n_neg": int(len(neg))}
    tp = int((pos >= thr).sum()); fp = int((neg >= thr).sum()); fn = int((pos < thr).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4),
            "fpr": round(fp / len(neg), 4), "thr": round(float(thr), 6),
            "n_pos": int(len(pos)), "n_neg": int(len(neg))}

def thr_at_fpr(neg_scores, fpr=FPR_TARGET):
    neg = np.asarray(neg_scores, float); neg = neg[~np.isnan(neg)]
    return float(np.quantile(neg, 1.0 - fpr))

def make_clean224(native_path, out_path):
    """Replicate the attack load/save pipeline for the unperturbed clean input."""
    if os.path.exists(out_path):
        return out_path
    img = Image.open(native_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    arr = np.asarray(img).astype(np.uint8)
    Image.fromarray(arr).save(out_path, quality=JPEG_Q)
    return out_path

def lap_scores(paths):
    return np.array([lap_flat(p) for p in paths], float)

# ---------------------------------------------------------------------------
# ESCI same-manifest (detector's home) -- clean224 already regenerated by T2.
ESCI_CLEAN224 = f"{R}/runs/T2_detector/clean224"
ESCI_METHODS = ["cogeo", "pgd_bare", "advclip", "coattack"]
esci_pids = sorted(p[:-4] for p in os.listdir(ESCI_CLEAN224) if p.endswith(".jpg"))
print(f"[esci] clean224 pids: {len(esci_pids)}")

esci_clean_paths = [f"{ESCI_CLEAN224}/{pid}.jpg" for pid in esci_pids]
esci_clean_lap = lap_scores(esci_clean_paths)
tau_esci = thr_at_fpr(esci_clean_lap, FPR_TARGET)   # THE operating point (clean-only)
print(f"[esci] tau@5%FPR (clean-calibrated) = {tau_esci:.6f}")

results = {"_meta": {
    "slice": "B1-detector-generalization", "campaign": "analysis-45b62125",
    "detector": "global training-free Laplacian flat-region energy (lap_flat), identical to analysis-47bea597 detector_v2_matched.py",
    "operating_point": f"tau = {1-FPR_TARGET:.2f} quantile of CLEAN lap scores; flag if lap>=tau",
    "tau_esci_clean_5pct_fpr": round(tau_esci, 6),
    "unit": "per unique image (deduplicated)",
    "haves": {"scipy": HAVE_SCIPY, "sklearn": HAVE_SK},
    "note": "tau is clean-only => detector has NO attack-family parameter; held-out-family AUROC == same-manifest by construction. Real tests are held-out distribution (Food-101) and held-out eps."}}

# Baseline + Probe A: ESCI eps16 per-family AUROC + recall@tau_esci (leave-one-attack-out).
results["esci_eps16_same_manifest"] = {}
for m in ESCI_METHODS:
    advd = f"{R}/runs/n3_{m}_eps16/img"
    adv_pids = [pid for pid in esci_pids if os.path.exists(f"{advd}/{pid}.jpg")]
    adv_lap = lap_scores([f"{advd}/{pid}.jpg" for pid in adv_pids])
    labels = np.r_[np.zeros(len(esci_clean_lap)), np.ones(len(adv_lap))]
    scores = np.r_[esci_clean_lap, adv_lap]
    au = auroc(scores, labels)
    op = op_at_thr(esci_clean_lap, adv_lap, tau_esci)
    results["esci_eps16_same_manifest"][m] = {"auroc": round(au, 4), "op_at_tau_esci": op, "n_adv": len(adv_pids)}
    print(f"[esci eps16 {m}] AUROC {au:.4f} | recall@tau {op['recall']} fpr {op['fpr']}")

# Probe B: held-out DISTRIBUTION = Food-101.
FOOD_NATIVE = f"{R}/data/food101_thumbs/img"
FOOD_CLEAN224 = f"{OUT}/food101_clean224"
os.makedirs(FOOD_CLEAN224, exist_ok=True)
FOOD_METHODS = ["cogeo", "pgd_bare", "advclip", "coattack"]
food_pids = sorted(p[:-4] for p in os.listdir(FOOD_NATIVE) if p.endswith(".jpg"))
print(f"[food101] native pids: {len(food_pids)}")
food_clean_paths = [make_clean224(f"{FOOD_NATIVE}/{pid}.jpg", f"{FOOD_CLEAN224}/{pid}.jpg") for pid in food_pids]
food_clean_lap = lap_scores(food_clean_paths)
tau_food = thr_at_fpr(food_clean_lap, FPR_TARGET)   # re-calibrated on the new distribution

# operating-point transfer: ESCI tau applied to Food-101 clean -> realised FPR
food_clean_fpr_transfer = round(float((food_clean_lap >= tau_esci).mean()), 4)
results["food101_held_out_distribution"] = {
    "n_clean": len(food_pids),
    "tau_esci_applied_to_food_clean_realised_fpr": food_clean_fpr_transfer,
    "tau_food_recalibrated_5pct_fpr": round(tau_food, 6),
    "clean_lap_mean_esci": round(float(np.nanmean(esci_clean_lap)), 6),
    "clean_lap_mean_food": round(float(np.nanmean(food_clean_lap)), 6),
    "per_method": {}}
for m in FOOD_METHODS:
    advd = f"{R}/runs/E4_food101_eps16/{m}/img"
    if not os.path.isdir(advd):
        continue
    adv_pids = [pid for pid in food_pids if os.path.exists(f"{advd}/{pid}.jpg")]
    adv_lap = lap_scores([f"{advd}/{pid}.jpg" for pid in adv_pids])
    labels = np.r_[np.zeros(len(food_clean_lap)), np.ones(len(adv_lap))]
    scores = np.r_[food_clean_lap, adv_lap]
    au = auroc(scores, labels)
    op_transfer = op_at_thr(food_clean_lap, adv_lap, tau_esci)   # transferred tau
    op_recal = op_at_thr(food_clean_lap, adv_lap, tau_food)      # re-calibrated tau
    results["food101_held_out_distribution"]["per_method"][m] = {
        "auroc": round(au, 4),
        "op_at_tau_esci_transferred": op_transfer,
        "op_at_tau_food_recalibrated": op_recal,
        "n_adv": len(adv_pids)}
    print(f"[food101 {m}] AUROC {au:.4f} | recall@tau_esci {op_transfer['recall']} (fpr {op_transfer['fpr']}) | recall@tau_food {op_recal['recall']}")
print(f"[food101] tau_esci transferred -> realised FPR on food clean = {food_clean_fpr_transfer}")

# Probe C: held-out perturbation budget eps in {4,8} (ESCI clean shared).
results["esci_held_out_eps"] = {}
for eps in [4, 8]:
    results["esci_held_out_eps"][f"eps{eps}"] = {}
    for m in ["cogeo", "pgd_bare"]:
        advd = f"{R}/runs/E2_openai_eps{eps}/{m}/img"
        if not os.path.isdir(advd):
            continue
        adv_pids = [pid for pid in esci_pids if os.path.exists(f"{advd}/{pid}.jpg")]
        adv_lap = lap_scores([f"{advd}/{pid}.jpg" for pid in adv_pids])
        labels = np.r_[np.zeros(len(esci_clean_lap)), np.ones(len(adv_lap))]
        scores = np.r_[esci_clean_lap, adv_lap]
        au = auroc(scores, labels)
        op = op_at_thr(esci_clean_lap, adv_lap, tau_esci)
        results["esci_held_out_eps"][f"eps{eps}"][m] = {"auroc": round(au, 4), "op_at_tau_esci": op, "n_adv": len(adv_pids)}
        print(f"[esci eps{eps} {m}] AUROC {au:.4f} | recall@tau_esci {op['recall']} fpr {op['fpr']}")

json.dump(results, open(f"{OUT}/b1_detector_generalization.json", "w"), indent=2)
print("WROTE", f"{OUT}/b1_detector_generalization.json")
