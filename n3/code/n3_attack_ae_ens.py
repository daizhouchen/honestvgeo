#!/usr/bin/env python3
"""AE-CoGEO encoder-ensemble: transfer-designed visual-GEO attack.

Single-encoder CoGEO (and anchor-ensemble CoGEO) overfits the perturbation to ONE
retriever's embedding geometry, so the white-box rank-lift collapses (~97%) when the
deployed retriever uses a different CLIP backbone. This script generalizes the CoGEO
optimizer to optimize the perturbation against an ENSEMBLE OF SOURCE ENCODERS at once:
the masked MI-FGSM step descends the *mean* anchor-alignment loss across all source
models, so delta is pushed toward a direction that is adversarial across architectures /
pretraining corpora rather than to a single model's idiosyncrasies. Evaluated on a
HELD-OUT encoder (not in the source set), this is the standard ensemble-transfer recipe
(Liu et al. 2017) adapted to anchor-decoupled visual GEO.

Protocol invariant unchanged: every anchor is derived ONLY from x_p's product title;
the held-out ESCI query q is never seen here. Anchors are encoded by EACH source model's
own text encoder. Image normalization is the shared CLIP mean/std (valid for OpenAI /
LAION / EVA02 CLIP backbones; SigLIP is used ONLY as a held-out eval target, never a source).

Reuses helpers from n3_attack_ae.py so the single-encoder arm and this arm share the exact
optimizer, mask, env-sim, anchor construction, and per-image seeding (truly paired).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from n3_attack_ae import (
    title_anchors, sobel_mask, make_clip_norm, load_image, save_image,
    _try_import_envsim,
)

LOG = logging.getLogger("n3_attack_ae_ens")

# open_clip (arch, pretrained) registry — copied verbatim from eval_harness.py so the
# attack source encoders and the eval backbones name-match exactly.
_BACKBONE_REGISTRY = {
    "openai/clip-vit-base-patch32": ("ViT-B-32", "openai"),
    "ViT-B-32": ("ViT-B-32", "openai"),
    "openai/clip-vit-large-patch14": ("ViT-L-14", "openai"),
    "ViT-L-14": ("ViT-L-14", "openai"),
    "ViT-L-14-openai": ("ViT-L-14", "openai"),
    "ViT-L-14-laion2b": ("ViT-L-14", "laion2b_s32b_b82k"),
    "EVA02-L-14": ("EVA02-L-14", "merged2b_s4b_b131k"),
    "EVA02-L-14-merged2b": ("EVA02-L-14", "merged2b_s4b_b131k"),
    # DataComp-XL ViT-L/14: clean 4th SOURCE encoder. Same architecture as the held-out
    # LAION ViT-L/14, but trained on the DataComp-1B corpus (not LAION, not SigLIP, not in
    # the held-out eval set), so it broadens the cross-corpus consensus direction without
    # ever leaking a held-out checkpoint into the source ensemble.
    "ViT-L-14-datacomp": ("ViT-L-14", "datacomp_xl_s13b_b90k"),
    "ViT-H-14-laion2b": ("ViT-H-14", "laion2b_s32b_b79k"),
    "ViT-B-32-laion2b": ("ViT-B-32", "laion2b_s34b_b79k"),
}


def load_clip_one(backbone: str, device: torch.device):
    import open_clip  # type: ignore
    if backbone not in _BACKBONE_REGISTRY:
        raise SystemExit(f"unsupported source backbone: {backbone}")
    arch, pretrained = _BACKBONE_REGISTRY[backbone]
    model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained, device=str(device))
    tokenizer = open_clip.get_tokenizer(arch)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    LOG.info("loaded source CLIP %s (%s/%s)", backbone, arch, pretrained)
    return model, tokenizer


def attack_cogeo_encoder_ensemble(x, per_model_anchors, models, norms, eps, alpha, iters,
                                  mu=1.0, agg="mean", tau=8.0, envsim_fn=None, mask_lo=0.1):
    """CoGEO masked MI-FGSM generalized to a SOURCE-ENCODER ensemble.

    per_model_anchors: list (len M) of [K_m, D_m] L2-normalized text feats, one per model.
    models / norms    : aligned lists of M frozen CLIP models and their norm fns.
    The loss is the MEAN over models of each model's anchor-alignment loss (mean or
    softmin agg, identical to the single-encoder variant). Gradient flows to delta
    through all M models -> a cross-encoder-consensus perturbation."""
    mask = sobel_mask(x, lo=mask_lo)
    g_buf = torch.zeros_like(x)
    delta = torch.zeros_like(x, requires_grad=True)
    centroids = None
    if agg == "mean":
        centroids = []
        for v in per_model_anchors:
            c = v.mean(dim=0, keepdim=True)
            centroids.append(c / (c.norm(dim=-1, keepdim=True) + 1e-12))  # [1, D_m]
    for _ in range(iters):
        x_adv = torch.clamp(x + delta, 0.0, 1.0)
        x_in = envsim_fn(x_adv) if envsim_fn is not None else x_adv  # one shared EOT draw
        losses = []
        for m, (model, norm_fn) in enumerate(zip(models, norms)):
            f = model.encode_image(norm_fn(x_in))
            f = f / f.norm(dim=-1, keepdim=True)  # [1, D_m]
            if agg == "mean":
                losses.append(-(f * centroids[m]).sum(dim=-1).mean())
            else:  # softmin over this model's per-anchor cos
                cos = (f @ per_model_anchors[m].t()).squeeze(0)  # [K_m]
                losses.append((1.0 / tau) * torch.logsumexp(-tau * cos, dim=0))
        loss = torch.stack(losses).mean()  # average adversarial pressure across encoders
        grad = torch.autograd.grad(loss, delta)[0]
        grad_norm = grad / (grad.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-12)
        g_buf = mu * g_buf + grad_norm
        step = alpha * g_buf.sign() * mask
        delta = (delta - step).detach()
        delta = torch.clamp(delta, -eps, eps)
        delta = (torch.clamp(x + delta, 0.0, 1.0) - x).detach()
        delta.requires_grad_(True)
    return torch.clamp(x + delta.detach(), 0.0, 1.0)


def encode_anchor_lists(anchor_lists, model, tokenizer, device):
    """Encode all anchors for all images with ONE model. Returns (flat_feats, offsets)."""
    flat_texts, offsets = [], []
    for al in anchor_lists:
        start = len(flat_texts)
        flat_texts.extend(al)
        offsets.append((start, len(flat_texts)))
    with torch.no_grad():
        feats = []
        for i in range(0, len(flat_texts), 64):
            tok = tokenizer(flat_texts[i:i + 64]).to(device)
            f = model.encode_text(tok)
            feats.append(f / f.norm(dim=-1, keepdim=True))
        flat = torch.cat(feats, dim=0) if feats else torch.empty(0, device=device)
    return flat, offsets


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip-backbones", required=True,
                   help="comma-separated source encoders, e.g. 'openai/clip-vit-large-patch14,ViT-B-32,EVA02-L-14'")
    p.add_argument("--anchor-mode", choices=("title", "ensemble"), default="ensemble")
    p.add_argument("--ensemble-agg", choices=("mean", "softmin"), default="mean")
    p.add_argument("--max-anchors", type=int, default=6)
    p.add_argument("--tau", type=float, default=8.0)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--img-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--eps", type=float, default=16.0)
    p.add_argument("--alpha", type=float, default=4.0)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--mu", type=float, default=1.0)
    p.add_argument("--use-envsim", action="store_true")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    out_img_dir = args.out_dir / "img"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    backbones = [b.strip() for b in args.clip_backbones.split(",") if b.strip()]
    LOG.info("source encoder ensemble (M=%d): %s", len(backbones), backbones)

    df = pd.read_csv(args.manifest)
    df["_img_path"] = df["product_id"].apply(lambda a: args.img_dir / f"{a}.jpg")
    have = df["_img_path"].apply(lambda p: p.exists() and p.stat().st_size > 0)
    if not have.all():
        LOG.warning("dropping %d pairs with missing image", int((~have).sum()))
        df = df[have].reset_index(drop=True)
    df = df.drop_duplicates(subset=["product_id"]).reset_index(drop=True)
    if args.max_images is not None:
        df = df.head(args.max_images).reset_index(drop=True)
    LOG.info("anchor-mode=%s agg=%s max_anchors=%d eps=%.1f images=%d envsim=%s",
             args.anchor_mode, args.ensemble_agg, args.max_anchors, args.eps, len(df), args.use_envsim)

    # Load all source models + per-model norm fns.
    models, tokenizers, norms = [], [], []
    for bb in backbones:
        m, tk = load_clip_one(bb, device)
        models.append(m); tokenizers.append(tk)
        norms.append(make_clip_norm(args.image_size, device))

    # Build query-free anchor lists once (shared text across models; encoded per model).
    titles = df["product_title"].astype(str).tolist()
    if args.anchor_mode == "title":
        anchor_lists = [[t] for t in titles]
    else:
        anchor_lists = [title_anchors(t, max_anchors=args.max_anchors) for t in titles]

    # Encode anchors with EACH model's text encoder.
    per_model_flat, per_model_off = [], []
    for m, tk in zip(models, tokenizers):
        flat, off = encode_anchor_lists(anchor_lists, m, tk, device)
        per_model_flat.append(flat); per_model_off.append(off)
    LOG.info("encoded anchors with %d text encoders (mean %.2f anchors/img)",
             len(models), len(per_model_flat[0]) / max(1, len(df)))

    envsim_fn = None
    if args.use_envsim:
        es_mod = _try_import_envsim()
        if es_mod is not None and hasattr(es_mod, "EnvSim"):
            envsim_fn = es_mod.EnvSim().to(device)
            LOG.info("env-sim ENABLED")

    eps_f = args.eps / 255.0
    alpha_f = args.alpha / 255.0
    audit_rows, pert_l2 = [], []
    t0 = time.time()
    for idx, row in df.iterrows():
        asin = row["product_id"]
        out_path = out_img_dir / f"{asin}.jpg"
        per_model_anchors = []
        for m_i in range(len(models)):
            s, e = per_model_off[m_i][idx]
            per_model_anchors.append(per_model_flat[m_i][s:e])
        random.seed(args.seed * 100003 + idx)
        torch.manual_seed(args.seed * 100003 + idx)
        x = load_image(row["_img_path"], args.image_size).to(device)
        x_adv = attack_cogeo_encoder_ensemble(
            x, per_model_anchors, models, norms, eps_f, alpha_f, args.iters,
            mu=args.mu, agg=args.ensemble_agg, tau=args.tau, envsim_fn=envsim_fn)
        save_image(x_adv, out_path)
        pert_l2.append(float(torch.norm(x_adv - x).item()))
        audit_rows.append({"product_id": asin, "n_anchors": int(len(per_model_anchors[0])),
                           "anchors": " || ".join(anchor_lists[idx])})
        if (idx + 1) % 25 == 0:
            el = time.time() - t0
            LOG.info("  attacked %d/%d in %.1fs (avg %.2fs/img)", idx + 1, len(df), el, el / (idx + 1))

    with open(args.out_dir / "ae_anchor_audit.csv", "w") as fh:
        w = csv.DictWriter(fh, fieldnames=["product_id", "n_anchors", "anchors"])
        w.writeheader(); w.writerows(audit_rows)
    summary = {
        "method": "ae_cogeo_encoder_ensemble",
        "source_backbones": backbones, "n_source_encoders": len(backbones),
        "anchor_mode": args.anchor_mode, "ensemble_agg": args.ensemble_agg,
        "max_anchors": args.max_anchors, "eps": args.eps, "alpha": args.alpha,
        "iters": args.iters, "mu": args.mu, "use_envsim": bool(args.use_envsim), "tau": args.tau,
        "mean_pert_l2": round(float(np.mean(pert_l2)), 4) if pert_l2 else 0,
        "n_images": int(len(df)), "wall_seconds": round(time.time() - t0, 2),
        "out_img_dir": str(out_img_dir), "image_size": args.image_size, "seed": args.seed,
    }
    (args.out_dir / "attack_summary.json").write_text(json.dumps(summary, indent=2))
    LOG.info("attack summary: %s", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
