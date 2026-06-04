#!/usr/bin/env python3
"""Pixel-domain Gaussian-noise stealth control for the visual-GEO benchmark.

For every image in the 491-ASIN ESCI manifest, draws a random Gaussian
perturbation projected onto the same L_inf budget eps=16/255 used by the four
adversarial baselines (CoGEO, PGD-bare, AdvCLIP, Co-Attack) and reports:

* mean / median L_inf and L_2-RMS magnitudes of the perturbation, and
* mean / median pixel-domain SSIM between the original and the Gaussian-perturbed
  image, computed at the retriever's native 224 x 224 input.

This is the directional control behind the paper's claim that an L_inf=16/255
*Gaussian* perturbation is stealthier than every attack (mean SSIM = 0.854 vs
CoGEO 0.66 / PGD-bare 0.76 / Co-Attack 0.76 / AdvCLIP 0.78) yet moves the
ranking by ~0. The "~0 rank-lift" half is an analytical first-order bound (an
isotropic L_inf=16/255 noise projected into a CLIP image embedding of dimension
d>=512 has expected held-out cosine-similarity change ~0 by symmetry, hence
expected rank-lift ~0 in the within-pool diagnostic); only the *measured* SSIM
is computed here. Deterministic (fixed seed); CPU-only; numpy + Pillow + scipy.

Run (images are downloaded per the manifest's image_path, not shipped):
  PROJECT_ROOT=. python src/analysis/gaussian_control.py \
      --manifest manifests/esci500_manifest.csv \
      --img-dir  <cache>/esci500/img \
      --out      results/ablations/gaussian_control/gaussian_control_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter

EPS_255 = 16.0       # L_inf budget in [0, 255]; matches eps=16/255 in the main paper
INPUT_SIZE = 224     # CLIP retriever native input
NUM_REPLICATES = 3   # independent Gaussian draws per image, averaged at the end
SEED = 20260512


def ssim_2d(x: np.ndarray, y: np.ndarray) -> float:
    """Single-scale SSIM in [0, 1] on a 2D float image in [0, 1] (K1=0.01, K2=0.03, L=1.0)."""
    K1, K2, L = 0.01, 0.03, 1.0
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    win = 8
    mu_x = uniform_filter(x, win)
    mu_y = uniform_filter(y, win)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x2 = uniform_filter(x * x, win) - mu_x2
    sigma_y2 = uniform_filter(y * y, win) - mu_y2
    sigma_xy = uniform_filter(x * y, win) - mu_xy
    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return float(np.mean(num / den))


def ssim_rgb(x: np.ndarray, y: np.ndarray) -> float:
    """Mean per-channel SSIM on an (H, W, 3) float image in [0, 1]."""
    return float(np.mean([ssim_2d(x[..., c], y[..., c]) for c in range(3)]))


def main() -> None:
    root = os.environ.get("PROJECT_ROOT", ".")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=Path(root) / "manifests" / "esci500_manifest.csv",
                    help="ESCI manifest CSV (columns include product_id)")
    ap.add_argument("--img-dir", type=Path, required=True,
                    help="Cache dir of original product images, named <product_id>.jpg")
    ap.add_argument("--out", type=Path,
                    default=Path(root) / "results" / "ablations" / "gaussian_control" / "gaussian_control_summary.json")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    with open(args.manifest) as f:
        rows = list(csv.DictReader(f))
    unique_asins = {row["product_id"]: row for row in rows}
    print(f"Loaded {len(rows)} rows; {len(unique_asins)} unique ASINs.")

    eps_unit = EPS_255 / 255.0
    metrics: list[dict] = []
    for idx, (asin, _row) in enumerate(sorted(unique_asins.items())):
        img_path = args.img_dir / f"{asin}.jpg"
        if not img_path.exists():
            continue
        with Image.open(img_path) as im:
            im = im.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
            x = np.asarray(im, dtype=np.float32) / 255.0
        l_inf_r, l2_r, ssim_r = [], [], []
        for _k in range(NUM_REPLICATES):
            # sigma chosen so ~99% of samples sit inside [-eps, +eps] before clipping.
            delta = rng.normal(0.0, eps_unit / 2.58, size=x.shape).astype(np.float32)
            delta = np.clip(delta, -eps_unit, eps_unit)
            x_adv = np.clip(x + delta, 0.0, 1.0)
            l_inf_r.append(float(np.max(np.abs(x_adv - x))))
            l2_r.append(float(np.sqrt(np.mean((x_adv - x) ** 2))))
            ssim_r.append(ssim_rgb(x, x_adv))
        metrics.append({"asin": asin, "l_inf_mean": float(np.mean(l_inf_r)),
                        "l2_rms_mean": float(np.mean(l2_r)), "ssim_mean": float(np.mean(ssim_r))})
        if (idx + 1) % 50 == 0:
            print(f"  processed {idx + 1} / {len(unique_asins)} ASINs")

    arr_linf = np.array([m["l_inf_mean"] for m in metrics])
    arr_l2 = np.array([m["l2_rms_mean"] for m in metrics])
    arr_ssim = np.array([m["ssim_mean"] for m in metrics])
    summary = {
        "n_unique_asins_processed": len(metrics),
        "input_size": INPUT_SIZE,
        "eps_l_inf_unit": eps_unit,
        "eps_l_inf_pixel": EPS_255,
        "num_replicates_per_asin": NUM_REPLICATES,
        "seed": SEED,
        "l_inf": {"mean": float(arr_linf.mean()), "median": float(np.median(arr_linf)),
                  "p05": float(np.percentile(arr_linf, 5)), "p95": float(np.percentile(arr_linf, 95))},
        "l2_rms": {"mean": float(arr_l2.mean()), "median": float(np.median(arr_l2))},
        "ssim_at_224": {"mean": float(arr_ssim.mean()), "median": float(np.median(arr_ssim)),
                        "p05": float(np.percentile(arr_ssim, 5)), "p95": float(np.percentile(arr_ssim, 95))},
        "comparison_at_224_eps16_255": {"advclip_ssim": 0.78, "pgd_bare_ssim": 0.76,
                                        "coattack_ssim": 0.76, "cogeo_ssim": 0.66,
                                        "gaussian_ssim": float(arr_ssim.mean())},
    }
    print(json.dumps(summary, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"Wrote summary -> {args.out}")


if __name__ == "__main__":
    main()
