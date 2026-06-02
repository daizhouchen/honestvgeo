#!/usr/bin/env python3
"""N3 gradient-FREE (black-box) adversarial-image generator.

Ceiling-check for the ICDM rebuttal: is CoGEO's white-box rank-lift advantage a
real upper bound, or merely an artifact internal to the gradient-attack family?

This attacker uses ONLY forward queries to CLIP's image encoder (NEVER gradients,
NEVER backprop). Optimizer = NES (Natural Evolution Strategies) antithetic
sampling, the standard score-function black-box attack (Ilyas et al. 2018).

Protocol invariant (identical to n3_attack.py CoGEO):

    v_anchor(x_p) = CLIP_text(product_title(x_p))    -- depends only on x_p
    The held-out ESCI query q is read by eval_harness.py only; this script
    never sees q. (anchor-decoupled)

Constraint: L_inf eps/255 ball, clamp to [0,1], image_size 224, optional Sobel
mask (matches CoGEO when --use-mask). Image load/save pipeline is byte-identical
to n3_attack.py (BICUBIC resize 224, uint8 jpg quality=95).

Budget: pop forward evals/iter, iters iterations => pop*iters forward queries
per image. Reported exactly in attack_summary.json as forward_queries_per_image.
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

LOG = logging.getLogger("n3_attack_gfree")

_BACKBONE_REGISTRY = {
    "openai/clip-vit-base-patch32": ("ViT-B-32", "openai"),
    "openai/clip-vit-large-patch14": ("ViT-L-14", "openai"),
    "ViT-L-14": ("ViT-L-14", "openai"),
    "ViT-L-14-openai": ("ViT-L-14", "openai"),
    "ViT-L-14-laion2b": ("ViT-L-14", "laion2b_s32b_b82k"),
}

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def load_clip(backbone, device, force_image_size=None):
    import open_clip
    if backbone not in _BACKBONE_REGISTRY:
        raise SystemExit(f"unsupported backbone: {backbone}")
    arch, pretrained = _BACKBONE_REGISTRY[backbone]
    kwargs = {}
    if force_image_size is not None and force_image_size != 224:
        kwargs["force_image_size"] = force_image_size
    model, _, preproc = open_clip.create_model_and_transforms(
        arch, pretrained=pretrained, device=str(device), **kwargs)
    tokenizer = open_clip.get_tokenizer(arch)
    model.eval()
    LOG.info("loaded CLIP %s -> arch=%s pretrained=%s", backbone, arch, pretrained)
    return model, preproc, tokenizer


def _extract_mean_std(preproc, device):
    mean, std = CLIP_MEAN, CLIP_STD
    try:
        for t in getattr(preproc, "transforms", []):
            if hasattr(t, "mean") and hasattr(t, "std"):
                mean = tuple(float(m) for m in t.mean)
                std = tuple(float(s) for s in t.std)
                break
    except Exception:
        pass
    return (torch.tensor(mean, device=device).view(1, 3, 1, 1),
            torch.tensor(std, device=device).view(1, 3, 1, 1))


def make_clip_norm(image_size, device, preproc=None):
    if preproc is not None:
        mean, std = _extract_mean_std(preproc, device)
    else:
        mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)

    def norm(x):
        if x.shape[-1] != image_size or x.shape[-2] != image_size:
            x = F.interpolate(x, size=(image_size, image_size), mode="bicubic",
                              align_corners=False, antialias=True)
        return (x - mean) / std
    return norm


def sobel_mask(x, lo=0.1):
    sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device)
    sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device)
    sx = sx.view(1, 1, 3, 3).expand(x.shape[1], 1, 3, 3)
    sy = sy.view(1, 1, 3, 3).expand(x.shape[1], 1, 3, 3)
    gx = F.conv2d(x, sx, padding=1, groups=x.shape[1])
    gy = F.conv2d(x, sy, padding=1, groups=x.shape[1])
    g = torch.sqrt(gx ** 2 + gy ** 2 + 1e-12).mean(dim=1, keepdim=True)
    g = (g - g.amin(dim=(2, 3), keepdim=True)) / (
        g.amax(dim=(2, 3), keepdim=True) - g.amin(dim=(2, 3), keepdim=True) + 1e-12)
    return torch.clamp(g, min=lo)


def load_image(p, image_size):
    img = Image.open(p).convert("RGB")
    img = img.resize((image_size, image_size), Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def save_image(t, out_path):
    arr = (t.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    Image.fromarray(arr).save(out_path, quality=95)


@torch.no_grad()
def fitness_batch(x_batch, v_anchor, model, norm_fn):
    """cos(CLIP_image(x), v_anchor) for a batch. PURE FORWARD, no_grad enforced."""
    x_norm = norm_fn(x_batch)
    f = model.encode_image(x_norm)
    f = f / f.norm(dim=-1, keepdim=True)
    return (f * v_anchor).sum(dim=-1)  # [B]


@torch.no_grad()
def attack_nes(x, v_anchor, model, norm_fn, eps, sigma, lr, pop, iters,
               mask=None, query_batch=50, query_counter=None):
    """NES black-box attack (antithetic sampling). Maximizes anchor cosine.

    pop = population per iter (must be even). antithetic pairs => pop/2 noise
    draws, pop forward evals/iter. Total forward queries = pop * iters.
    """
    assert pop % 2 == 0
    device = x.device
    delta = torch.zeros_like(x)
    half = pop // 2
    for it in range(iters):
        noise = torch.randn((half,) + tuple(x.shape[1:]), device=device)
        all_noise = torch.cat([noise, -noise], dim=0)  # [pop, C, H, W]
        cand_delta = delta + sigma * all_noise
        if mask is not None:
            cand_delta = cand_delta * mask
        cand_delta = torch.clamp(cand_delta, -eps, eps)
        x_cand = torch.clamp(x + cand_delta, 0.0, 1.0)  # [pop, C, H, W]
        fits = []
        for i in range(0, pop, query_batch):
            fb = fitness_batch(x_cand[i:i + query_batch], v_anchor, model, norm_fn)
            fits.append(fb)
        fits = torch.cat(fits, dim=0)  # [pop]
        if query_counter is not None:
            query_counter[0] += pop
        rewards = fits
        r = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        grad_est = (r.view(-1, 1, 1, 1) * all_noise).mean(dim=0, keepdim=True) / sigma
        delta = delta + lr * grad_est
        if mask is not None:
            delta = delta * mask
        delta = torch.clamp(delta, -eps, eps)
        delta = torch.clamp(x + delta, 0.0, 1.0) - x
    return torch.clamp(x + delta, 0.0, 1.0)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--img-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--pids-file", type=Path, default=None)
    p.add_argument("--anchor-mode", choices=("product-title",), default="product-title")
    p.add_argument("--eps", type=int, default=16)
    p.add_argument("--sigma", type=float, default=0.01)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--pop", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--use-mask", action="store_true")
    p.add_argument("--clip-backbone", default="openai/clip-vit-large-patch14")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--query-batch", type=int, default=50)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_img_dir = args.out_dir / "img"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    LOG.info("GFREE/NES eps=%d sigma=%.4g lr=%.4g pop=%d iters=%d fwd/img=%d mask=%s device=%s",
             args.eps, args.sigma, args.lr, args.pop, args.iters, args.pop * args.iters,
             args.use_mask, device)

    df = pd.read_csv(args.manifest)
    df = df.copy()
    df["_img_path"] = df["product_id"].apply(lambda a: args.img_dir / f"{a}.jpg")
    have = df["_img_path"].apply(lambda p: p.exists() and p.stat().st_size > 0)
    df = df[have].reset_index(drop=True)
    df = df.drop_duplicates(subset=["product_id"]).reset_index(drop=True)

    if args.pids_file is not None:
        wanted = [l.strip() for l in open(args.pids_file) if l.strip()]
        order = {pid: i for i, pid in enumerate(wanted)}
        df = df[df["product_id"].isin(set(wanted))].copy()
        df["_ord"] = df["product_id"].map(order)
        df = df.sort_values("_ord").reset_index(drop=True)
        LOG.info("restricted to %d/%d requested pids", len(df), len(wanted))
    if args.max_images is not None:
        df = df.head(args.max_images).reset_index(drop=True)
    LOG.info("attacking %d unique images", len(df))

    anchor_texts = df["product_title"].astype(str).tolist()

    model, preproc, tokenizer = load_clip(args.clip_backbone, device, force_image_size=args.image_size)
    norm_fn = make_clip_norm(args.image_size, device, preproc=preproc)

    with torch.no_grad():
        feats = []
        for i in range(0, len(anchor_texts), 64):
            tok = tokenizer(anchor_texts[i:i + 64]).to(device)
            f = model.encode_text(tok)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f)
        anchor_feats = torch.cat(feats, dim=0)

    eps_f = args.eps / 255.0
    query_counter = [0]
    rows = []
    t0_all = time.time()
    for idx, row in df.iterrows():
        asin = row["product_id"]
        out_path = out_img_dir / f"{asin}.jpg"
        x = load_image(row["_img_path"], args.image_size).to(device)
        v = anchor_feats[idx:idx + 1]
        mask = sobel_mask(x, lo=0.1) if args.use_mask else None
        t0 = time.time()
        x_adv = attack_nes(x, v, model, norm_fn, eps_f, args.sigma, args.lr,
                           args.pop, args.iters, mask=mask,
                           query_batch=args.query_batch, query_counter=query_counter)
        save_image(x_adv, out_path)
        rows.append({"product_id": asin, "out_path": str(out_path), "skipped": False,
                     "seconds": round(time.time() - t0, 3)})
        if (idx + 1) % 25 == 0:
            el = time.time() - t0_all
            LOG.info("  attacked %d/%d in %.1fs (avg %.2fs/img)", idx + 1, len(df), el, el / (idx + 1))

    log_csv = args.out_dir / "attack_log.csv"
    with open(log_csv, "w") as fh:
        w = csv.DictWriter(fh, fieldnames=["product_id", "out_path", "skipped", "seconds"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})

    summary = {
        "method": "gfree_nes",
        "anchor_mode": args.anchor_mode,
        "eps": args.eps,
        "sigma": args.sigma,
        "lr": args.lr,
        "pop": args.pop,
        "iters": args.iters,
        "forward_queries_per_image": args.pop * args.iters,
        "use_mask": bool(args.use_mask),
        "n_images": int(len(df)),
        "total_forward_queries": int(query_counter[0]),
        "wall_seconds": round(time.time() - t0_all, 2),
        "out_img_dir": str(out_img_dir),
        "clip_backbone": args.clip_backbone,
        "image_size": args.image_size,
        "manifest": str(args.manifest),
        "gradient_used": False,
    }
    (args.out_dir / "attack_summary.json").write_text(json.dumps(summary, indent=2))
    LOG.info("attack summary: %s", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
