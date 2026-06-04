#!/usr/bin/env python3
"""N3 adversarial-image generator.

Produces x_p* per ASIN from the ESCI manifest under one of:

  --method cogeo     -- product-title anchor + Sobel mask + MI-FGSM PGD with
                        env-sim (DIM/TIM/DiffJPEG). Reuses the existing
                        envsim.py + diffjpeg_lite.py from the remote
                        ``code/`` tree. v_anchor = CLIP_text(product_title);
                        the eval-time ESCI query is NEVER consulted.

  --method pgd_bare  -- plain ε-bounded PGD against the same product-title
                        anchor. No mask, no env-sim, no momentum. Sanity
                        comparator that isolates whether CoGEO's extra
                        ingredients matter under the held-out eval.

Protocol invariant (non-negotiable):

    v_anchor(x_p) = CLIP_text(product_title(x_p))   -- only depends on x_p
    The held-out ESCI query q is read by eval_harness.py only;
    n3_attack.py never sees q.

Layout::

    --img-dir   /data/esci500/img         (originals, named {ASIN}.jpg)
    --manifest  /data/esci500/manifest.csv
    --out-dir   /runs/n3_<method>_eps<eps>/img/   (writes {ASIN}.jpg)

For full ablation modes (vlm anchor / random-text / cat-name) you can pass
``--anchor-mode``; the defaults preserve the protocol invariant.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

LOG = logging.getLogger("n3_attack")


_BACKBONE_REGISTRY = {
    "openai/clip-vit-base-patch32": ("ViT-B-32", "openai"),
    "ViT-B-32": ("ViT-B-32", "openai"),
    "openai/clip-vit-large-patch14": ("ViT-L-14", "openai"),
    "ViT-L-14": ("ViT-L-14", "openai"),
    "ViT-L-14-openai": ("ViT-L-14", "openai"),
    "openclip-ViT-L-14-laion2b": ("ViT-L-14", "laion2b_s32b_b82k"),
    "ViT-L-14-laion2b": ("ViT-L-14", "laion2b_s32b_b82k"),
    "EVA02-L-14": ("EVA02-L-14", "merged2b_s4b_b131k"),
    "EVA02-L-14-merged2b": ("EVA02-L-14", "merged2b_s4b_b131k"),
    # New for 6-backbone matrix.
    "ViT-H-14-laion2b": ("ViT-H-14", "laion2b_s32b_b79k"),
    "ViT-B-32-laion2b": ("ViT-B-32", "laion2b_s34b_b79k"),
    "ViT-B-16-SigLIP-webli": ("ViT-B-16-SigLIP", "webli"),
}


def load_clip(backbone: str, device: torch.device, force_image_size: int | None = None):
    import open_clip  # type: ignore
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
    LOG.info("loaded CLIP %s -> arch=%s pretrained=%s force_image_size=%s",
             backbone, arch, pretrained, force_image_size)
    return model, preproc, tokenizer


# --- env-sim stubs (mirrors of remote envsim.py contract) -------------------

def _try_import_envsim():
    try:
        # Allow run from a directory that includes envsim.py / diffjpeg_lite.py
        import envsim  # type: ignore
        return envsim
    except Exception as exc:
        LOG.warning("envsim.py not importable (%s); CoGEO env-sim will be a no-op", exc)
        return None


def sobel_mask(x: torch.Tensor, lo: float = 0.1) -> torch.Tensor:
    """Texture mask via Sobel gradient magnitude, min-max normalized then clamped."""
    sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device)
    sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device)
    sx = sx.view(1, 1, 3, 3).expand(x.shape[1], 1, 3, 3)
    sy = sy.view(1, 1, 3, 3).expand(x.shape[1], 1, 3, 3)
    gx = F.conv2d(x, sx, padding=1, groups=x.shape[1])
    gy = F.conv2d(x, sy, padding=1, groups=x.shape[1])
    g = torch.sqrt(gx ** 2 + gy ** 2 + 1e-12).mean(dim=1, keepdim=True)
    g = (g - g.amin(dim=(2, 3), keepdim=True)) / (g.amax(dim=(2, 3), keepdim=True) - g.amin(dim=(2, 3), keepdim=True) + 1e-12)
    g = torch.clamp(g, min=lo)
    return g  # [B, 1, H, W] in [lo, 1.0]


def attack_pgd_bare(
    x: torch.Tensor, v_anchor: torch.Tensor, model, preproc_norm,
    eps: float, alpha: float, iters: int,
) -> torch.Tensor:
    """Plain PGD: maximize cos(CLIP_image(x_adv), v_anchor) within an L_inf ε ball."""
    delta = torch.zeros_like(x, requires_grad=True)
    for _ in range(iters):
        x_adv = torch.clamp(x + delta, 0.0, 1.0)
        x_norm = preproc_norm(x_adv)
        f = model.encode_image(x_norm)
        f = f / f.norm(dim=-1, keepdim=True)
        loss = -(f * v_anchor).sum(dim=-1).mean()  # maximize sim => minimize -sim
        grad = torch.autograd.grad(loss, delta)[0]
        delta = (delta - alpha * grad.sign()).detach()
        delta = torch.clamp(delta, -eps, eps)
        delta = (torch.clamp(x + delta, 0.0, 1.0) - x).detach()
        delta.requires_grad_(True)
    return torch.clamp(x + delta.detach(), 0.0, 1.0)


def attack_cogeo(
    x: torch.Tensor, v_anchor: torch.Tensor, model, preproc_norm,
    eps: float, alpha: float, iters: int, mu: float,
    envsim_fn=None, mask_lo: float = 0.1,
) -> torch.Tensor:
    """CoGEO core: anchor-guided MI-FGSM PGD with Sobel mask and env-sim."""
    mask = sobel_mask(x, lo=mask_lo)  # [B, 1, H, W]
    g_buf = torch.zeros_like(x)
    delta = torch.zeros_like(x, requires_grad=True)
    for _ in range(iters):
        x_adv = torch.clamp(x + delta, 0.0, 1.0)
        if envsim_fn is not None:
            x_envsim = envsim_fn(x_adv)
        else:
            x_envsim = x_adv
        x_norm = preproc_norm(x_envsim)
        f = model.encode_image(x_norm)
        f = f / f.norm(dim=-1, keepdim=True)
        loss = -(f * v_anchor).sum(dim=-1).mean()
        grad = torch.autograd.grad(loss, delta)[0]
        # MI-FGSM momentum
        grad_norm = grad / (grad.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-12)
        g_buf = mu * g_buf + grad_norm
        step = alpha * g_buf.sign() * mask  # apply texture mask
        delta = (delta - step).detach()
        delta = torch.clamp(delta, -eps, eps)
        delta = (torch.clamp(x + delta, 0.0, 1.0) - x).detach()
        delta.requires_grad_(True)
    return torch.clamp(x + delta.detach(), 0.0, 1.0)


# --- preproc (CLIP normalize on a 0..1 RGB tensor) --------------------------

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _extract_mean_std(preproc, device):
    """Extract Normalize mean/std from a torchvision Compose preproc, falling
    back to OpenAI CLIP defaults."""
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


def make_clip_norm(image_size: int, device: torch.device, preproc=None):
    if preproc is not None:
        mean, std = _extract_mean_std(preproc, device)
    else:
        mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)
    LOG.info("clip_norm mean=%s std=%s image_size=%d",
             mean.flatten().tolist(), std.flatten().tolist(), image_size)

    def norm(x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != image_size or x.shape[-2] != image_size:
            x = F.interpolate(x, size=(image_size, image_size), mode="bicubic", align_corners=False, antialias=True)
        return (x - mean) / std

    return norm


def load_image(p: Path, image_size: int) -> torch.Tensor:
    img = Image.open(p).convert("RGB")
    img = img.resize((image_size, image_size), Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 255.0  # [H, W, 3]
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]


def save_image(t: torch.Tensor, out_path: Path) -> None:
    arr = (t.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    Image.fromarray(arr).save(out_path, quality=95)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=("cogeo", "pgd_bare"), required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--img-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--anchor-mode", choices=("product-title", "category", "random-text", "vlm"),
                   default="product-title", help="Source of v_anchor (must depend only on x_p)")
    p.add_argument("--vlm-captions-json", type=Path, default=None,
                   help="(vlm anchor only) JSON file {product_id: caption_text} pre-generated by n3_vlm_caption.py")
    p.add_argument("--eps", type=int, default=16, help="L_inf budget in /255 units")
    p.add_argument("--alpha", type=int, default=4, help="step size in /255 units")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--mu", type=float, default=1.0, help="MI-FGSM momentum (cogeo only)")
    p.add_argument("--clip-backbone", default="openai/clip-vit-base-patch32")
    p.add_argument("--image-size", type=int, default=224)  # CLIP-ViT-B/32 input size
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-images", type=int, default=None, help="Optional cap on #images for smoke test")
    p.add_argument("--use-envsim", action="store_true",
                   help="(cogeo only) try to import remote envsim.py for DIM/TIM/DiffJPEG")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_img_dir = args.out_dir / "img"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    LOG.info("method=%s eps=%d alpha=%d iters=%d device=%s anchor=%s",
             args.method, args.eps, args.alpha, args.iters, device, args.anchor_mode)

    df = pd.read_csv(args.manifest)
    df = df.copy()
    df["_img_path"] = df["product_id"].apply(lambda a: args.img_dir / f"{a}.jpg")
    have = df["_img_path"].apply(lambda p: p.exists() and p.stat().st_size > 0)
    if not have.all():
        LOG.warning("dropping %d pairs whose original image is missing", int((~have).sum()))
        df = df[have].reset_index(drop=True)
    df = df.drop_duplicates(subset=["product_id"]).reset_index(drop=True)
    if args.max_images is not None:
        df = df.head(args.max_images).reset_index(drop=True)
    LOG.info("attacking %d unique images", len(df))

    # Build per-image anchor text.
    if args.anchor_mode == "product-title":
        anchor_texts = df["product_title"].astype(str).tolist()
    elif args.anchor_mode == "category":
        if "product_brand" in df.columns:
            anchor_texts = df["product_brand"].fillna(df["product_title"]).astype(str).tolist()
        else:
            anchor_texts = df["product_title"].astype(str).tolist()
    elif args.anchor_mode == "random-text":
        # deterministic random text per ASIN; uncorrelated with q_ESCI
        rng = np.random.RandomState(args.seed)
        vocab = ["object", "item", "thing", "product", "model", "design", "device", "tool", "set", "kit"]
        anchor_texts = [" ".join(rng.choice(vocab, size=4)) for _ in range(len(df))]
    elif args.anchor_mode == "vlm":
        if args.vlm_captions_json is None or not args.vlm_captions_json.exists():
            raise SystemExit("--anchor-mode vlm requires --vlm-captions-json pointing to an existing JSON file")
        with open(args.vlm_captions_json) as fh:
            caption_map = json.load(fh)
        missing = [a for a in df["product_id"].astype(str) if a not in caption_map]
        if missing:
            raise SystemExit(f"vlm captions JSON missing {len(missing)} product_ids; first 5: {missing[:5]}")
        anchor_texts = [str(caption_map[str(a)]) for a in df["product_id"].astype(str).tolist()]
        LOG.info("loaded %d VLM-generated captions from %s", len(anchor_texts), args.vlm_captions_json)
    else:
        raise ValueError

    # Load CLIP.
    model, preproc, tokenizer = load_clip(args.clip_backbone, device, force_image_size=args.image_size)
    norm_fn = make_clip_norm(args.image_size, device, preproc=preproc)

    # Encode anchor texts -> v_anchor tensors.
    LOG.info("encoding %d anchor texts", len(anchor_texts))
    t0 = time.time()
    with torch.no_grad():
        anchor_feats = []
        for i in range(0, len(anchor_texts), 64):
            chunk = anchor_texts[i:i + 64]
            tok = tokenizer(chunk).to(device)
            f = model.encode_text(tok)
            f = f / f.norm(dim=-1, keepdim=True)
            anchor_feats.append(f)
        anchor_feats = torch.cat(anchor_feats, dim=0)
    LOG.info("anchor encode done %.2fs", time.time() - t0)

    envsim_mod = _try_import_envsim() if args.use_envsim and args.method == "cogeo" else None
    envsim_fn = None
    if envsim_mod is not None and hasattr(envsim_mod, "EnvSim"):
        es = envsim_mod.EnvSim().to(device)
        envsim_fn = es

    eps_f = args.eps / 255.0
    alpha_f = args.alpha / 255.0

    rows = []
    t0_all = time.time()
    for idx, row in df.iterrows():
        asin = row["product_id"]
        out_path = out_img_dir / f"{asin}.jpg"
        if out_path.exists() and out_path.stat().st_size > 0:
            rows.append({"product_id": asin, "out_path": str(out_path), "skipped": True})
            continue
        x = load_image(row["_img_path"], args.image_size).to(device)
        v = anchor_feats[idx:idx + 1]
        t0 = time.time()
        if args.method == "pgd_bare":
            x_adv = attack_pgd_bare(x, v, model, norm_fn, eps_f, alpha_f, args.iters)
        else:
            x_adv = attack_cogeo(x, v, model, norm_fn, eps_f, alpha_f, args.iters, args.mu, envsim_fn=envsim_fn)
        save_image(x_adv, out_path)
        dt = time.time() - t0
        rows.append({"product_id": asin, "out_path": str(out_path), "skipped": False, "seconds": round(dt, 3)})
        if (idx + 1) % 25 == 0:
            elapsed = time.time() - t0_all
            LOG.info("  attacked %d/%d in %.1fs (avg %.2fs/img)", idx + 1, len(df), elapsed, elapsed / (idx + 1))

    log_csv = args.out_dir / "attack_log.csv"
    with open(log_csv, "w") as fh:
        w = csv.DictWriter(fh, fieldnames=["product_id", "out_path", "skipped", "seconds"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})

    summary = {
        "method": args.method,
        "anchor_mode": args.anchor_mode,
        "vlm_captions_json": (str(args.vlm_captions_json) if args.vlm_captions_json else None),
        "eps": args.eps,
        "alpha": args.alpha,
        "iters": args.iters,
        "n_images": int(len(df)),
        "n_attacked": int(sum(1 for r in rows if not r["skipped"])),
        "n_skipped": int(sum(1 for r in rows if r["skipped"])),
        "wall_seconds": round(time.time() - t0_all, 2),
        "out_img_dir": str(out_img_dir),
        "clip_backbone": args.clip_backbone,
        "image_size": args.image_size,
        "manifest": str(args.manifest),
    }
    (args.out_dir / "attack_summary.json").write_text(json.dumps(summary, indent=2))
    LOG.info("attack summary: %s", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
