#!/usr/bin/env python3
"""N3 reach-higher new method: Anchor-Ensemble CoGEO (AE-CoGEO).

Motivation
----------
The held-out eval (eval_harness.py --mode rank) scores cos(CLIP_image(x_p*),
CLIP_text(q)) against a held-out shopper query q. The CoGEO optimizer instead
maximizes cos(image, v_anchor) where v_anchor = CLIP_text(product_title(x_p)).
Real shopper queries are SHORT core concepts ("storage basket") while product
titles are LONG and noisy ("Joyeen Small Plastic Weave Baskets, Storage Bins,
6 Pack"). A single full-title anchor therefore under-covers the space of
plausible held-out queries.

AE-CoGEO keeps the CoGEO optimizer unchanged but replaces the single title
anchor with a QUERY-FREE ENSEMBLE of title-derived sub-anchors (full title +
cleaned core noun-phrases + comma segments + a CLIP prompt template). The image
is optimized to align with the whole anchor set, which covers more of the query
space at the SAME L_inf budget -> higher held-out rank-lift. This is
expectation-over-transformation applied to the *query* axis, not the pixel axis.

Protocol invariant (NON-NEGOTIABLE, same as n3_attack.py / n3_attack_cab.py):
    Every anchor is derived ONLY from x_p's product_title. The held-out ESCI
    query q is read by eval_harness.py only; this generator NEVER sees q, and
    no per-image quantity depends on q or on the human ESCI cohort label.

Two anchor modes (identical optimizer; the ONLY difference is the anchor set):
    --anchor-mode title     : single product-title anchor  (== faithful CoGEO)
    --anchor-mode ensemble  : K query-free title-derived anchors

Ensemble aggregation in the loss:
    --ensemble-agg mean     : optimize cos(image, centroid of normalized anchors)
                              (equivalent to maximizing the mean per-anchor cos)
    --ensemble-agg softmin  : maximize a soft-min over per-anchor cos (push the
                              WORST-covered anchor up -> robustness across queries)

Output layout matches n3_attack.py so eval_harness.py --mode rank works unchanged:
    --out-dir/img/{ASIN}.jpg
    --out-dir/attack_summary.json
    --out-dir/ae_anchor_audit.csv   (per-image anchor texts + count, for audit)
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

LOG = logging.getLogger("n3_attack_ae")

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# Tokens that carry no query-relevant concept: packaging / quantity / generic.
_STOP = {
    "the", "a", "an", "of", "for", "with", "and", "or", "to", "in", "on", "by",
    "set", "pack", "packs", "count", "ct", "pcs", "pc", "piece", "pieces", "pair",
    "pairs", "kit", "bundle", "box", "pack of", "x", "size", "large", "small",
    "medium", "mini", "new", "premium", "professional", "quality", "inch", "inches",
    "cm", "mm", "ft", "oz", "lb", "lbs", "ml", "l", "g", "kg", "color", "colors",
    "assorted", "multi", "multicolor", "value", "best", "pack.", "&",
}


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


def _try_import_envsim():
    try:
        import envsim  # type: ignore
        return envsim
    except Exception as exc:
        LOG.warning("envsim.py not importable (%s); env-sim will be a no-op", exc)
        return None


def title_anchors(title: str, max_anchors: int = 6) -> list[str]:
    """Build a QUERY-FREE anchor set from the product title only.

    Returns the full title plus cleaned core phrases (comma segments and leading/
    trailing content n-grams with packaging/quantity tokens removed) and one CLIP
    prompt template. All strings depend only on x_p's title -- never on q."""
    title = str(title).strip()
    anchors: list[str] = []
    if not title:
        return ["product"]
    anchors.append(title)  # 1. full title (preserves overall semantics)

    # split into segments on punctuation that usually separates concepts
    segs = re.split(r"[,;|()/]| - ", title)
    core_words: list[str] = []
    for seg in segs:
        toks = re.findall(r"[A-Za-z][A-Za-z'\-]+", seg)
        cw = [t for t in toks if t.lower() not in _STOP and len(t) > 1 and not t.isdigit()]
        if cw:
            phrase = " ".join(cw)
            if 1 <= len(cw) <= 6:
                anchors.append(phrase)  # 2. cleaned comma segment
            core_words.extend(cw)

    # 3/4. leading and trailing content n-grams (short, query-like core concepts)
    if len(core_words) >= 2:
        anchors.append(" ".join(core_words[:3]))
        anchors.append(" ".join(core_words[-2:]))
    # 5. CLIP prompt template wrapping the core concept
    core = " ".join(core_words[:4]) if core_words else title
    anchors.append(f"a product photo of {core}")

    # dedupe (case-insensitive), drop empties, cap to max_anchors
    seen: set[str] = set()
    out: list[str] = []
    for a in anchors:
        a = a.strip()
        key = a.lower()
        if a and key not in seen:
            seen.add(key)
            out.append(a)
        if len(out) >= max_anchors:
            break
    return out or [title]


def attack_cogeo_ens(x, v_anchors, model, preproc_norm, eps, alpha, iters,
                     mu=1.0, agg="mean", tau=8.0, envsim_fn=None, mask_lo=0.1):
    """CoGEO core (anchor-guided MI-FGSM + Sobel mask + optional env-sim) generalized
    to an anchor ENSEMBLE. v_anchors: [K, D] L2-normalized text features.

    agg='mean'    : align image to the (renormalized) anchor centroid.
                    With K=1 this is byte-identical to the original single-anchor CoGEO.
    agg='softmin' : maximize a temperature-tau soft-min over per-anchor cos -> lifts
                    the worst-covered anchor (robust query coverage)."""
    mask = sobel_mask(x, lo=mask_lo)
    g_buf = torch.zeros_like(x)
    delta = torch.zeros_like(x, requires_grad=True)
    if agg == "mean":
        v_cen = v_anchors.mean(dim=0, keepdim=True)
        v_cen = v_cen / (v_cen.norm(dim=-1, keepdim=True) + 1e-12)  # [1, D]
    for _ in range(iters):
        x_adv = torch.clamp(x + delta, 0.0, 1.0)
        x_in = envsim_fn(x_adv) if envsim_fn is not None else x_adv
        x_norm = preproc_norm(x_in)
        f = model.encode_image(x_norm)
        f = f / f.norm(dim=-1, keepdim=True)  # [1, D]
        if agg == "mean":
            loss = -(f * v_cen).sum(dim=-1).mean()
        else:  # softmin: maximize soft-min cos => minimize (1/tau)*logsumexp(-tau*cos)
            cos = (f @ v_anchors.t()).squeeze(0)  # [K]
            loss = (1.0 / tau) * torch.logsumexp(-tau * cos, dim=0)
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--anchor-mode", choices=("title", "ensemble"), required=True)
    p.add_argument("--ensemble-agg", choices=("mean", "softmin"), default="mean")
    p.add_argument("--max-anchors", type=int, default=6)
    p.add_argument("--tau", type=float, default=8.0, help="(softmin) temperature")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--img-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--eps", type=float, default=16.0, help="L_inf budget in /255 units")
    p.add_argument("--alpha", type=float, default=4.0, help="PGD step in /255 units")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--mu", type=float, default=1.0)
    p.add_argument("--use-envsim", action="store_true", help="apply CoGEO env-sim (DIM/TIM/DiffJPEG)")
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
    random.seed(args.seed)
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
    LOG.info("anchor-mode=%s agg=%s max_anchors=%d eps=%.1f images=%d envsim=%s device=%s",
             args.anchor_mode, args.ensemble_agg, args.max_anchors, args.eps, len(df),
             args.use_envsim, device)

    model, tokenizer = load_clip(args.clip_backbone, device)
    norm_fn = make_clip_norm(args.image_size, device)

    # Build per-image anchor text lists (query-free, from title only).
    titles = df["product_title"].astype(str).tolist()
    if args.anchor_mode == "title":
        anchor_lists = [[t] for t in titles]
    else:
        anchor_lists = [title_anchors(t, max_anchors=args.max_anchors) for t in titles]

    # Encode every anchor once (flatten across images, then regroup).
    flat_texts: list[str] = []
    offsets: list[tuple[int, int]] = []
    for al in anchor_lists:
        start = len(flat_texts)
        flat_texts.extend(al)
        offsets.append((start, len(flat_texts)))
    LOG.info("encoding %d anchor texts (%d images, mean %.2f anchors/img)",
             len(flat_texts), len(df), len(flat_texts) / max(1, len(df)))
    with torch.no_grad():
        feats = []
        for i in range(0, len(flat_texts), 64):
            tok = tokenizer(flat_texts[i:i + 64]).to(device)
            f = model.encode_text(tok)
            feats.append(f / f.norm(dim=-1, keepdim=True))
        flat_feats = torch.cat(feats, dim=0) if feats else torch.empty(0, device=device)

    envsim_fn = None
    if args.use_envsim:
        es_mod = _try_import_envsim()
        if es_mod is not None and hasattr(es_mod, "EnvSim"):
            envsim_fn = es_mod.EnvSim().to(device)
            LOG.info("env-sim ENABLED")

    eps_f = args.eps / 255.0
    alpha_f = args.alpha / 255.0

    audit_rows = []
    t0_all = time.time()
    pert_l2 = []
    for idx, row in df.iterrows():
        asin = row["product_id"]
        out_path = out_img_dir / f"{asin}.jpg"
        s, e = offsets[idx]
        v_anchors = flat_feats[s:e]  # [K_i, D]
        # Per-image seeding so env-sim draws are IDENTICAL across anchor-mode arms
        # (truly paired A/B): same image index -> same stochastic transforms.
        random.seed(args.seed * 100003 + idx)
        torch.manual_seed(args.seed * 100003 + idx)
        x = load_image(row["_img_path"], args.image_size).to(device)
        x_adv = attack_cogeo_ens(x, v_anchors, model, norm_fn, eps_f, alpha_f, args.iters,
                                 mu=args.mu, agg=args.ensemble_agg, tau=args.tau,
                                 envsim_fn=envsim_fn)
        save_image(x_adv, out_path)
        pert_l2.append(float(torch.norm(x_adv - x).item()))
        audit_rows.append({"product_id": asin, "n_anchors": int(e - s),
                           "anchors": " || ".join(anchor_lists[idx])})
        if (idx + 1) % 25 == 0:
            el = time.time() - t0_all
            LOG.info("  attacked %d/%d in %.1fs (avg %.2fs/img)", idx + 1, len(df), el, el / (idx + 1))

    audit_csv = args.out_dir / "ae_anchor_audit.csv"
    with open(audit_csv, "w") as fh:
        w = csv.DictWriter(fh, fieldnames=["product_id", "n_anchors", "anchors"])
        w.writeheader()
        w.writerows(audit_rows)

    n_anchor_counts = [r["n_anchors"] for r in audit_rows]
    summary = {
        "method": f"ae_cogeo_{args.anchor_mode}_{args.ensemble_agg}",
        "anchor_mode": args.anchor_mode,
        "ensemble_agg": args.ensemble_agg,
        "max_anchors": args.max_anchors,
        "mean_anchors_per_image": round(float(np.mean(n_anchor_counts)), 3) if n_anchor_counts else 0,
        "eps": args.eps, "alpha": args.alpha, "iters": args.iters, "mu": args.mu,
        "use_envsim": bool(args.use_envsim), "tau": args.tau,
        "mean_pert_l2": round(float(np.mean(pert_l2)), 4) if pert_l2 else 0,
        "n_images": int(len(df)), "wall_seconds": round(time.time() - t0_all, 2),
        "out_img_dir": str(out_img_dir), "clip_backbone": args.clip_backbone,
        "image_size": args.image_size, "manifest": str(args.manifest), "seed": args.seed,
    }
    (args.out_dir / "attack_summary.json").write_text(json.dumps(summary, indent=2))
    LOG.info("attack summary: %s", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
