#!/usr/bin/env python3
"""N3 reach-higher new method: Cohort-Adaptive Budget (CAB) attack.

Motivation (from this paper's own C2 finding): per-cohort heterogeneity means a
uniform L_inf budget over-spends on images whose rank-lift saturates and
under-spends on high-yield images. CAB keeps the CoGEO optimizer but reallocates
a *fixed mean* L_inf budget across images using a QUERY-FREE difficulty signal.

Protocol invariant (NON-NEGOTIABLE, same as n3_attack.py):
    v_anchor(x_p) = CLIP_text(product_title(x_p))  -- depends ONLY on x_p.
    The held-out ESCI query q is read by eval_harness.py only; this generator
    NEVER sees q, and the per-image budget is NEVER a function of q or of the
    human ESCI cohort label. The adaptivity signal is computed only from x_p.

Two budget modes (identical optimizer; the ONLY difference is eps_i):
    --budget-mode uniform   : eps_i = mean_eps for all i  (== CoGEO baseline)
    --budget-mode adaptive  : eps_i from a query-free headroom signal, scaled so
                              mean_i(eps_i) == mean_eps, clipped to [eps_lo,eps_hi].

Output layout matches n3_attack.py so eval_harness.py --mode rank works unchanged:
    --out-dir/img/{ASIN}.jpg
    --out-dir/attack_summary.json
    --out-dir/cab_budget_alloc.csv   (per-image headroom + assigned eps)
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

LOG = logging.getLogger("n3_attack_cab")

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def load_clip(backbone: str, device: torch.device):
    import open_clip  # type: ignore
    if backbone in ("openai/clip-vit-base-patch32", "ViT-B-32"):
        model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", device=str(device))
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
    elif backbone in ("openai/clip-vit-large-patch14", "ViT-L-14"):
        model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", device=str(device))
        tokenizer = open_clip.get_tokenizer("ViT-L-14")
    else:
        raise SystemExit(f"unsupported backbone: {backbone}")
    model.eval()
    LOG.info("loaded CLIP %s", backbone)
    return model, tokenizer


def make_clip_norm(image_size: int, device: torch.device):
    mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)

    def norm(x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != image_size or x.shape[-2] != image_size:
            x = F.interpolate(x, size=(image_size, image_size), mode="bicubic", align_corners=False, antialias=True)
        return (x - mean) / std

    return norm


def sobel_mask(x: torch.Tensor, lo: float = 0.1) -> torch.Tensor:
    sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device)
    sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device)
    sx = sx.view(1, 1, 3, 3).expand(x.shape[1], 1, 3, 3)
    sy = sy.view(1, 1, 3, 3).expand(x.shape[1], 1, 3, 3)
    gx = F.conv2d(x, sx, padding=1, groups=x.shape[1])
    gy = F.conv2d(x, sy, padding=1, groups=x.shape[1])
    g = torch.sqrt(gx ** 2 + gy ** 2 + 1e-12).mean(dim=1, keepdim=True)
    g = (g - g.amin(dim=(2, 3), keepdim=True)) / (g.amax(dim=(2, 3), keepdim=True) - g.amin(dim=(2, 3), keepdim=True) + 1e-12)
    return torch.clamp(g, min=lo)


def attack_cogeo(x, v_anchor, model, preproc_norm, eps, alpha, iters, mu=1.0, mask_lo=0.1):
    """CoGEO core: anchor-guided MI-FGSM PGD with Sobel mask. eps is a scalar float (0..1)."""
    mask = sobel_mask(x, lo=mask_lo)
    g_buf = torch.zeros_like(x)
    delta = torch.zeros_like(x, requires_grad=True)
    for _ in range(iters):
        x_adv = torch.clamp(x + delta, 0.0, 1.0)
        x_norm = preproc_norm(x_adv)
        f = model.encode_image(x_norm)
        f = f / f.norm(dim=-1, keepdim=True)
        loss = -(f * v_anchor).sum(dim=-1).mean()
        grad = torch.autograd.grad(loss, delta)[0]
        grad_norm = grad / (grad.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-12)
        g_buf = mu * g_buf + grad_norm
        step = alpha * g_buf.sign() * mask
        delta = (delta - step).detach()
        delta = torch.clamp(delta, -eps, eps)
        delta = (torch.clamp(x + delta, 0.0, 1.0) - x).detach()
        delta.requires_grad_(True)
    return torch.clamp(x + delta.detach(), 0.0, 1.0)


def load_image(p: Path, image_size: int) -> torch.Tensor:
    img = Image.open(p).convert("RGB").resize((image_size, image_size), Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def save_image(t: torch.Tensor, out_path: Path) -> None:
    arr = (t.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    Image.fromarray(arr).save(out_path, quality=95)


def allocate_budget(headroom: np.ndarray, mean_eps: float, eps_lo: float, eps_hi: float,
                    gamma: float = 1.0) -> np.ndarray:
    """Rank/percentile allocation of a query-free difficulty signal to per-image eps.

    Proportional (eps ~ headroom) collapses when headrooms cluster tightly, leaving
    eps ~= uniform and failing to TEST the hypothesis. Instead we map the headroom
    RANK percentile p_i in [0,1] linearly into [eps_lo, eps_hi] (higher headroom =>
    more room to climb => larger budget), then shift to hit mean == mean_eps exactly.
    With a band symmetric around mean_eps the shift is ~0 and the full spread is kept.
    gamma>1 sharpens the percentile (concentrates budget at the extremes)."""
    s = np.asarray(headroom, dtype=np.float64)
    n = len(s)
    if n == 1:
        return np.array([mean_eps], dtype=np.float64)
    ranks = np.argsort(np.argsort(s))            # 0..n-1, ties by stable order
    p = ranks / (n - 1)                          # percentile in [0,1]
    if gamma != 1.0:                             # optional sharpening around the median
        p = np.where(p >= 0.5, 0.5 + 0.5 * (2 * p - 1) ** gamma,
                     0.5 - 0.5 * (1 - 2 * p) ** gamma)
    eps = eps_lo + p * (eps_hi - eps_lo)
    eps = eps + (mean_eps - eps.mean())          # shift to exact target mean
    eps = np.clip(eps, 1.0, None)                # safety floor only
    if abs(eps.mean() - mean_eps) > 1e-6:        # restore mean if the floor clipped
        eps = eps * (mean_eps / eps.mean())
    return eps


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--budget-mode", choices=("uniform", "adaptive"), required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--img-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--mean-eps", type=float, default=16.0, help="mean L_inf budget in /255 units")
    p.add_argument("--eps-lo", type=float, default=8.0, help="(adaptive) min per-image eps /255")
    p.add_argument("--eps-hi", type=float, default=28.0, help="(adaptive) max per-image eps /255")
    p.add_argument("--gamma", type=float, default=1.0, help="(adaptive) headroom exponent (sharpness)")
    p.add_argument("--alpha", type=float, default=4.0, help="PGD step in /255 units")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--mu", type=float, default=1.0)
    p.add_argument("--clip-backbone", default="openai/clip-vit-large-patch14")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_img_dir = args.out_dir / "img"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(args.manifest)
    df["_img_path"] = df["product_id"].apply(lambda a: args.img_dir / f"{a}.jpg")
    have = df["_img_path"].apply(lambda p: p.exists() and p.stat().st_size > 0)
    if not have.all():
        LOG.warning("dropping %d pairs with missing image", int((~have).sum()))
        df = df[have].reset_index(drop=True)
    df = df.drop_duplicates(subset=["product_id"]).reset_index(drop=True)
    if args.max_images is not None:
        df = df.head(args.max_images).reset_index(drop=True)
    LOG.info("budget-mode=%s mean_eps=%.2f band=[%.1f,%.1f] images=%d device=%s",
             args.budget_mode, args.mean_eps, args.eps_lo, args.eps_hi, len(df), device)

    anchor_texts = df["product_title"].astype(str).tolist()
    model, tokenizer = load_clip(args.clip_backbone, device)
    norm_fn = make_clip_norm(args.image_size, device)

    with torch.no_grad():
        feats = []
        for i in range(0, len(anchor_texts), 64):
            tok = tokenizer(anchor_texts[i:i + 64]).to(device)
            f = model.encode_text(tok)
            feats.append(f / f.norm(dim=-1, keepdim=True))
        anchor_feats = torch.cat(feats, dim=0)

    # ---- PASS 1: query-free headroom = 1 - cos(CLIP_image(x_clean), v_anchor) ----
    headroom = np.ones(len(df), dtype=np.float64)
    if args.budget_mode == "adaptive":
        LOG.info("pass1: computing query-free headroom for %d images", len(df))
        with torch.no_grad():
            for idx, row in df.iterrows():
                x = load_image(row["_img_path"], args.image_size).to(device)
                f = model.encode_image(norm_fn(x))
                f = f / f.norm(dim=-1, keepdim=True)
                cos = float((f * anchor_feats[idx:idx + 1]).sum(dim=-1).item())
                headroom[idx] = max(1.0 - cos, 1e-6)
        eps_alloc = allocate_budget(headroom, args.mean_eps, args.eps_lo, args.eps_hi, gamma=args.gamma)
    else:
        eps_alloc = np.full(len(df), args.mean_eps, dtype=np.float64)
    LOG.info("eps alloc: mean=%.3f min=%.2f max=%.2f std=%.2f",
             float(eps_alloc.mean()), float(eps_alloc.min()), float(eps_alloc.max()), float(eps_alloc.std()))

    # ---- PASS 2: attack each image at its assigned budget ----
    alpha_f = args.alpha / 255.0
    rows = []
    t0_all = time.time()
    pert_l2 = []
    for idx, row in df.iterrows():
        asin = row["product_id"]
        out_path = out_img_dir / f"{asin}.jpg"
        eps_i = float(eps_alloc[idx]) / 255.0
        x = load_image(row["_img_path"], args.image_size).to(device)
        v = anchor_feats[idx:idx + 1]
        x_adv = attack_cogeo(x, v, model, norm_fn, eps_i, alpha_f, args.iters, mu=args.mu)
        save_image(x_adv, out_path)
        pert_l2.append(float(torch.norm(x_adv - x).item()))
        rows.append({"product_id": asin, "headroom": round(float(headroom[idx]), 6),
                     "eps_assigned": round(float(eps_alloc[idx]), 4)})
        if (idx + 1) % 25 == 0:
            el = time.time() - t0_all
            LOG.info("  attacked %d/%d in %.1fs (avg %.2fs/img)", idx + 1, len(df), el, el / (idx + 1))

    alloc_csv = args.out_dir / "cab_budget_alloc.csv"
    with open(alloc_csv, "w") as fh:
        w = csv.DictWriter(fh, fieldnames=["product_id", "headroom", "eps_assigned"])
        w.writeheader()
        w.writerows(rows)

    summary = {
        "method": f"cogeo_cab_{args.budget_mode}",
        "budget_mode": args.budget_mode,
        "mean_eps_target": args.mean_eps,
        "mean_eps_achieved": round(float(eps_alloc.mean()), 4),
        "eps_lo": args.eps_lo, "eps_hi": args.eps_hi, "gamma": args.gamma,
        "eps_min": round(float(eps_alloc.min()), 4), "eps_max": round(float(eps_alloc.max()), 4),
        "eps_std": round(float(eps_alloc.std()), 4),
        "mean_pert_l2": round(float(np.mean(pert_l2)), 4),
        "alpha": args.alpha, "iters": args.iters, "mu": args.mu,
        "n_images": int(len(df)), "wall_seconds": round(time.time() - t0_all, 2),
        "out_img_dir": str(out_img_dir), "clip_backbone": args.clip_backbone,
        "image_size": args.image_size, "manifest": str(args.manifest), "seed": args.seed,
    }
    (args.out_dir / "attack_summary.json").write_text(json.dumps(summary, indent=2))
    LOG.info("attack summary: %s", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
