#!/usr/bin/env python3
"""N3 secondary CLIP-targeted attack baselines.

Two methods, both share the output contract of n3_attack.py
(``--out-dir/img/{ASIN}.jpg`` + ``attack_summary.json`` + ``attack_log.csv``)
so eval_harness.py can ingest them with no changes:

  --method advclip   Universal patch (Zhou+ 2023, arXiv 2308.07026):
                     train ONE delta_uap shared across the manifest images,
                     bounded by L_inf eps, that maximizes
                     avg cos(CLIP_image(x + delta_uap), v_anchor(x)).
                     The ESCI eval-time query q is never consulted during
                     training. Patch is applied unchanged to each image
                     at write time.

  --method coattack  Cross-modal white-box (Zhang+ 2022, arXiv 2206.09391):
                     per (x, v_anchor) pair, alternate between an L_inf
                     image-side perturbation and an L2 text-side
                     embedding perturbation; both cooperate to maximize
                     cos(CLIP_image(x_adv), v_anchor + delta_v_norm).
                     The held-out ESCI query q is still never seen.

Protocol invariant (preserved):

    v_anchor(x) = CLIP_text(product_title(x))   -- function of x only
    The held-out ESCI query q is read by eval_harness.py only.
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

LOG = logging.getLogger("n3_attack_baselines")

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


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
            x = F.interpolate(x, size=(image_size, image_size),
                              mode="bicubic", align_corners=False, antialias=True)
        return (x - mean) / std

    return norm


def load_image(p: Path, image_size: int) -> torch.Tensor:
    img = Image.open(p).convert("RGB")
    img = img.resize((image_size, image_size), Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def save_image(t: torch.Tensor, out_path: Path) -> None:
    arr = (t.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    Image.fromarray(arr).save(out_path, quality=95)


def encode_anchors(model, tokenizer, texts: list[str], device: torch.device,
                   chunk: int = 64) -> torch.Tensor:
    feats = []
    with torch.no_grad():
        for i in range(0, len(texts), chunk):
            tok = tokenizer(texts[i:i + chunk]).to(device)
            f = model.encode_text(tok)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f)
    return torch.cat(feats, dim=0)


# --- AdvCLIP (universal patch) ---------------------------------------------

def train_universal_patch(
    image_paths: list[Path], anchor_feats: torch.Tensor,
    model, preproc_norm, image_size: int, eps_f: float, alpha_f: float,
    epochs: int, batch_size: int, device: torch.device,
) -> torch.Tensor:
    """Train a universal L_inf-bounded delta on the ESCI training set.

    Loss = - mean cos(CLIP_image(clip(x_b + delta, 0, 1)), v_anchor_b).
    We do one signed-step PGD update of delta per minibatch, with a wraparound
    over the full set repeated `epochs` times. Per-batch inner iters = 1 to
    keep cost dominated by epochs.
    """
    n = len(image_paths)
    LOG.info("AdvCLIP UAP training: n=%d epochs=%d bs=%d eps=%.4f alpha=%.4f",
             n, epochs, batch_size, eps_f, alpha_f)

    # Pre-tile a [N, 3, H, W] tensor would blow GPU mem at 491 images x 3 x 224 x 224 = ~290 MB at fp32.
    # Cheap enough to keep on-device but also fine to load per-batch.
    delta = torch.zeros(1, 3, image_size, image_size, device=device, requires_grad=True)
    rng = np.random.RandomState(123)
    indices = np.arange(n)
    t0 = time.time()
    for ep in range(epochs):
        rng.shuffle(indices)
        ep_loss = 0.0
        ep_batches = 0
        for s in range(0, n, batch_size):
            batch_idx = indices[s:s + batch_size]
            xb = torch.cat(
                [load_image(image_paths[i], image_size) for i in batch_idx], dim=0
            ).to(device)
            ab = anchor_feats[batch_idx]
            x_adv = torch.clamp(xb + delta, 0.0, 1.0)
            x_norm = preproc_norm(x_adv)
            f = model.encode_image(x_norm)
            f = f / f.norm(dim=-1, keepdim=True)
            loss = -(f * ab).sum(dim=-1).mean()
            grad = torch.autograd.grad(loss, delta)[0]
            delta = (delta - alpha_f * grad.sign()).detach()
            delta = torch.clamp(delta, -eps_f, eps_f).detach()
            delta.requires_grad_(True)
            ep_loss += float(loss.detach())
            ep_batches += 1
        LOG.info("  epoch %d/%d: avg loss %.4f (%.1fs total)",
                 ep + 1, epochs, ep_loss / max(1, ep_batches), time.time() - t0)
    return delta.detach()


# --- Co-Attack (cross-modal) -----------------------------------------------

def attack_coattack(
    x: torch.Tensor, v_anchor: torch.Tensor, model, preproc_norm,
    eps_x: float, alpha_x: float, iters: int,
    eps_v: float = 0.1, alpha_v: float = 0.02,
) -> torch.Tensor:
    """Cross-modal bi-directional white-box attack.

    Maintains two adversarial deltas:
        delta_x: image-side L_inf within eps_x   (signed step)
        delta_v: text-anchor L2 within eps_v     (normalized step in embed space)

    Both update simultaneously each iteration via a shared loss
        L = -cos(CLIP_image(x + delta_x), normalize(v_anchor + delta_v)).
    """
    delta_x = torch.zeros_like(x, requires_grad=True)
    delta_v = torch.zeros_like(v_anchor, requires_grad=True)
    for _ in range(iters):
        v = v_anchor + delta_v
        v_n = v / (v.norm(dim=-1, keepdim=True) + 1e-12)
        x_adv = torch.clamp(x + delta_x, 0.0, 1.0)
        x_norm = preproc_norm(x_adv)
        f = model.encode_image(x_norm)
        f = f / f.norm(dim=-1, keepdim=True)
        loss = -(f * v_n).sum(dim=-1).mean()
        grads = torch.autograd.grad(loss, [delta_x, delta_v])
        gx, gv = grads[0], grads[1]
        # image-side L_inf signed step
        delta_x = (delta_x - alpha_x * gx.sign()).detach()
        delta_x = torch.clamp(delta_x, -eps_x, eps_x)
        delta_x = (torch.clamp(x + delta_x, 0.0, 1.0) - x).detach()
        delta_x.requires_grad_(True)
        # text-side L2 step on embedding
        gv_norm = gv / (gv.norm(dim=-1, keepdim=True) + 1e-12)
        delta_v_new = delta_v.detach() - alpha_v * gv_norm
        # project to L2 ball of radius eps_v
        n = delta_v_new.norm(dim=-1, keepdim=True)
        scale = torch.minimum(torch.ones_like(n), eps_v / (n + 1e-12))
        delta_v = (delta_v_new * scale).detach()
        delta_v.requires_grad_(True)
    return torch.clamp(x + delta_x.detach(), 0.0, 1.0)


# --- main -------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=("advclip", "coattack"), required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--img-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--anchor-mode", choices=("product-title", "category", "random-text"),
                   default="product-title")
    p.add_argument("--eps", type=int, default=16, help="L_inf budget on image in /255 units")
    p.add_argument("--alpha", type=int, default=4, help="image step size in /255 units")
    p.add_argument("--iters", type=int, default=200, help="(coattack) per-image iterations")
    # AdvCLIP universal patch knobs
    p.add_argument("--uap-epochs", type=int, default=20,
                   help="(advclip) epochs over the manifest for UAP training")
    p.add_argument("--uap-batch", type=int, default=16,
                   help="(advclip) batch size for UAP training")
    # Co-Attack text-side knobs
    p.add_argument("--text-eps", type=float, default=0.1,
                   help="(coattack) L2 budget on text-anchor embedding perturbation")
    p.add_argument("--text-alpha", type=float, default=0.02,
                   help="(coattack) L2 step size for text-anchor perturbation")
    p.add_argument("--clip-backbone", default="openai/clip-vit-large-patch14")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_img_dir = args.out_dir / "img"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    LOG.info("method=%s eps=%d alpha=%d iters=%d device=%s anchor=%s backbone=%s",
             args.method, args.eps, args.alpha, args.iters, device,
             args.anchor_mode, args.clip_backbone)

    df = pd.read_csv(args.manifest)
    df = df.copy()
    df["_img_path"] = df["product_id"].apply(lambda a: args.img_dir / f"{a}.jpg")
    have = df["_img_path"].apply(lambda q: q.exists() and q.stat().st_size > 0)
    if not have.all():
        LOG.warning("dropping %d pairs whose original image is missing", int((~have).sum()))
        df = df[have].reset_index(drop=True)
    df = df.drop_duplicates(subset=["product_id"]).reset_index(drop=True)
    if args.max_images is not None:
        df = df.head(args.max_images).reset_index(drop=True)
    LOG.info("attacking %d unique images", len(df))

    # Build anchor texts.
    if args.anchor_mode == "product-title":
        anchor_texts = df["product_title"].astype(str).tolist()
    elif args.anchor_mode == "category":
        if "product_brand" in df.columns:
            anchor_texts = df["product_brand"].fillna(df["product_title"]).astype(str).tolist()
        else:
            anchor_texts = df["product_title"].astype(str).tolist()
    elif args.anchor_mode == "random-text":
        rng = np.random.RandomState(args.seed)
        vocab = ["object", "item", "thing", "product", "model", "design",
                 "device", "tool", "set", "kit"]
        anchor_texts = [" ".join(rng.choice(vocab, size=4)) for _ in range(len(df))]

    # Load CLIP + encode anchors.
    model, preproc, tokenizer = load_clip(args.clip_backbone, device, force_image_size=args.image_size)
    norm_fn = make_clip_norm(args.image_size, device, preproc=preproc)
    LOG.info("encoding %d anchor texts", len(anchor_texts))
    t0 = time.time()
    anchor_feats = encode_anchors(model, tokenizer, anchor_texts, device)
    LOG.info("anchor encode done %.2fs", time.time() - t0)

    eps_f = args.eps / 255.0
    alpha_f = args.alpha / 255.0

    rows = []
    t0_all = time.time()
    extra_summary = {}

    if args.method == "advclip":
        # Train universal patch.
        image_paths = [Path(p) for p in df["_img_path"].tolist()]
        delta_uap = train_universal_patch(
            image_paths=image_paths,
            anchor_feats=anchor_feats,
            model=model,
            preproc_norm=norm_fn,
            image_size=args.image_size,
            eps_f=eps_f,
            alpha_f=alpha_f,
            epochs=args.uap_epochs,
            batch_size=args.uap_batch,
            device=device,
        )
        delta_uap_path = args.out_dir / "delta_uap.pt"
        torch.save(delta_uap.detach().cpu(), delta_uap_path)
        extra_summary["delta_uap_path"] = str(delta_uap_path)
        extra_summary["delta_uap_l_inf_max"] = float(delta_uap.abs().max())
        LOG.info("UAP saved to %s (max |delta|=%.4f)",
                 delta_uap_path, extra_summary["delta_uap_l_inf_max"])
        # Apply to each image.
        for idx, row in df.iterrows():
            asin = row["product_id"]
            out_path = out_img_dir / f"{asin}.jpg"
            if out_path.exists() and out_path.stat().st_size > 0:
                rows.append({"product_id": asin, "out_path": str(out_path), "skipped": True})
                continue
            x = load_image(row["_img_path"], args.image_size).to(device)
            x_adv = torch.clamp(x + delta_uap, 0.0, 1.0)
            save_image(x_adv, out_path)
            rows.append({"product_id": asin, "out_path": str(out_path),
                         "skipped": False, "seconds": 0.0})
            if (idx + 1) % 100 == 0:
                LOG.info("  applied UAP to %d/%d", idx + 1, len(df))
    else:  # coattack
        for idx, row in df.iterrows():
            asin = row["product_id"]
            out_path = out_img_dir / f"{asin}.jpg"
            if out_path.exists() and out_path.stat().st_size > 0:
                rows.append({"product_id": asin, "out_path": str(out_path), "skipped": True})
                continue
            x = load_image(row["_img_path"], args.image_size).to(device)
            v = anchor_feats[idx:idx + 1]
            t0 = time.time()
            x_adv = attack_coattack(
                x, v, model, norm_fn,
                eps_x=eps_f, alpha_x=alpha_f, iters=args.iters,
                eps_v=args.text_eps, alpha_v=args.text_alpha,
            )
            save_image(x_adv, out_path)
            dt = time.time() - t0
            rows.append({"product_id": asin, "out_path": str(out_path),
                         "skipped": False, "seconds": round(dt, 3)})
            if (idx + 1) % 25 == 0:
                elapsed = time.time() - t0_all
                LOG.info("  coattack %d/%d in %.1fs (avg %.2fs/img)",
                         idx + 1, len(df), elapsed, elapsed / (idx + 1))

    log_csv = args.out_dir / "attack_log.csv"
    with open(log_csv, "w") as fh:
        w = csv.DictWriter(fh, fieldnames=["product_id", "out_path", "skipped", "seconds"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})

    summary = {
        "method": args.method,
        "anchor_mode": args.anchor_mode,
        "eps": args.eps,
        "alpha": args.alpha,
        "iters": args.iters,
        "uap_epochs": args.uap_epochs if args.method == "advclip" else None,
        "uap_batch": args.uap_batch if args.method == "advclip" else None,
        "text_eps": args.text_eps if args.method == "coattack" else None,
        "text_alpha": args.text_alpha if args.method == "coattack" else None,
        "n_images": int(len(df)),
        "n_attacked": int(sum(1 for r in rows if not r["skipped"])),
        "n_skipped": int(sum(1 for r in rows if r["skipped"])),
        "wall_seconds": round(time.time() - t0_all, 2),
        "out_img_dir": str(out_img_dir),
        "clip_backbone": args.clip_backbone,
        "image_size": args.image_size,
        "manifest": str(args.manifest),
        **extra_summary,
    }
    (args.out_dir / "attack_summary.json").write_text(json.dumps(summary, indent=2))
    LOG.info("attack summary: %s", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
