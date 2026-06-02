#!/usr/bin/env python3
"""N3 fifth attack family: diffusion-prior (DiffPGD) query-free CLIP-targeted attack.

Closes the self-stated Limitations gap "diffusion-prior attacks not tested". Shares the
EXACT output contract of n3_attack.py / n3_attack_baselines.py
(``--out-dir/img/{product_id}.jpg`` + ``attack_summary.json`` + ``attack_log.csv``) so
eval_harness.py (rank-lift) and the detectability harness ingest it with NO changes, and
it sits in the SAME 4->5 comparison (same eps=16/255 L_inf budget, same white-box surrogate
ViT-L-14, same protocol invariant v_anchor(x)=CLIP_text(product_title(x)); the held-out ESCI
query q is read by eval_harness.py only).

Method (DiffPGD; Xue+ NeurIPS 2023 recipe, adapted query-free):
    Alternate (a) a block of signed-gradient PGD steps that maximize
        cos(CLIP_image(x_adv), v_anchor)  s.t. ||x_adv - x||_inf <= eps,
    with (b) a diffusion-purification reprojection: push x_adv through a few forward+reverse
    steps of a pretrained latent diffusion model (Stable Diffusion v1.5 VAE+UNet, empty
    prompt) so the accumulated perturbation is re-expressed on the natural-image manifold,
    then re-clamp to the eps-ball of x. The diffusion prior is the ONLY new ingredient vs the
    plain-PGD baseline; the eps budget, surrogate, and anchor are held identical for fairness.

Compared to the existing 4 (cogeo / pgd_bare / advclip / coattack), this tests whether an
on-manifold (diffusion-prior) perturbation transfers better/worse and is more/less detectable.
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

LOG = logging.getLogger("n3_attack_diffprior")

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
# Load by repo id so diffusers fetches any missing subfolder weights via HF_ENDPOINT
# (the local cache snapshot is partial: vae/unet/text_encoder weights are not present).
# Set HF_ENDPOINT=https://hf-mirror.com in the environment (HF main is blocked on this box).
DEFAULT_SD_PATH = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def load_clip(backbone: str, device: torch.device):
    import open_clip  # type: ignore
    name = "ViT-L-14" if "large" in backbone or backbone == "ViT-L-14" else "ViT-B-32"
    model, _, _ = open_clip.create_model_and_transforms(name, pretrained="openai", device=str(device))
    tokenizer = open_clip.get_tokenizer(name)
    model.eval()
    LOG.info("loaded CLIP surrogate %s", name)
    return model, tokenizer


def make_clip_norm(image_size: int, device: torch.device):
    mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)

    def norm(x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != image_size or x.shape[-2] != image_size:
            x = F.interpolate(x, size=(image_size, image_size),
                              mode="bicubic", align_corners=False, antialias=True)
        return (x - mean) / std

    return norm


def load_image(p: Path, image_size: int) -> torch.Tensor:
    img = Image.open(p).convert("RGB").resize((image_size, image_size), Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def save_image(t: torch.Tensor, out_path: Path) -> None:
    arr = (t.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    Image.fromarray(arr).save(out_path, quality=95)


def encode_anchors(model, tokenizer, texts, device, chunk=64) -> torch.Tensor:
    feats = []
    with torch.no_grad():
        for i in range(0, len(texts), chunk):
            tok = tokenizer(texts[i:i + chunk]).to(device)
            f = model.encode_text(tok)
            feats.append(f / f.norm(dim=-1, keepdim=True))
    return torch.cat(feats, dim=0)


class DiffusionPrior:
    """Stable Diffusion v1.5 latent-diffusion purifier (DiffPure forward+reverse, empty prompt)."""

    def __init__(self, sd_path: str, device: torch.device, steps: int = 6, dtype=torch.float16):
        from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
        from transformers import CLIPTextModel, CLIPTokenizer
        self.device = device
        self.dtype = dtype
        self.vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae", torch_dtype=dtype).to(device).eval()
        self.unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet", torch_dtype=dtype).to(device).eval()
        self.sched = DDIMScheduler.from_pretrained(sd_path, subfolder="scheduler")
        tok = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
        text = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder", torch_dtype=dtype).to(device).eval()
        with torch.no_grad():
            ids = tok([""], padding="max_length", max_length=tok.model_max_length,
                      return_tensors="pt").input_ids.to(device)
            self.empty_emb = text(ids)[0]
        self.steps = steps
        self.scaling = float(getattr(self.vae.config, "scaling_factor", 0.18215))

    @torch.no_grad()
    def purify(self, x224: torch.Tensor, strength: float = 0.15) -> torch.Tensor:
        """x224: (1,3,224,224) in [0,1]. Returns purified (1,3,224,224) in [0,1]."""
        x = F.interpolate(x224, size=(512, 512), mode="bicubic", align_corners=False, antialias=True)
        x = (x * 2.0 - 1.0).to(self.dtype)
        z0 = self.vae.encode(x).latent_dist.mean * self.scaling
        self.sched.set_timesteps(self.steps, device=self.device)
        ts = self.sched.timesteps
        n_start = max(1, int(round(strength * len(ts))))
        t_start = ts[len(ts) - n_start]
        noise = torch.randn_like(z0)
        zt = self.sched.add_noise(z0, noise, t_start)
        for t in ts[len(ts) - n_start:]:
            eps = self.unet(zt, t, encoder_hidden_states=self.empty_emb).sample
            zt = self.sched.step(eps, t, zt).prev_sample
        dec = self.vae.decode(zt / self.scaling).sample.float()
        xp = ((dec + 1.0) / 2.0).clamp(0.0, 1.0)
        return F.interpolate(xp, size=(224, 224), mode="bicubic", align_corners=False, antialias=True)


def pgd_block(x_orig, x_adv, v_anchor, model, norm_fn, eps_f, alpha_f, steps):
    """In-place-ish signed-gradient PGD block maximizing cos(CLIP_img(x_adv), v_anchor)."""
    delta = (x_adv - x_orig).detach().requires_grad_(True)
    for _ in range(steps):
        xa = torch.clamp(x_orig + delta, 0.0, 1.0)
        f = model.encode_image(norm_fn(xa))
        f = f / f.norm(dim=-1, keepdim=True)
        loss = -(f * v_anchor).sum(dim=-1).mean()
        g = torch.autograd.grad(loss, delta)[0]
        delta = (delta - alpha_f * g.sign()).detach()
        delta = torch.clamp(delta, -eps_f, eps_f)
        delta = (torch.clamp(x_orig + delta, 0.0, 1.0) - x_orig).detach().requires_grad_(True)
    return torch.clamp(x_orig + delta.detach(), 0.0, 1.0)


def attack_diffprior(x, v_anchor, model, norm_fn, prior, eps_f, alpha_f,
                     blocks, steps_per_block, purify_strength):
    x_orig = x.detach()
    x_adv = x_orig.clone()
    for b in range(blocks):
        x_adv = pgd_block(x_orig, x_adv, v_anchor, model, norm_fn, eps_f, alpha_f, steps_per_block)
        if prior is not None and b < blocks - 1:  # purify between blocks, not after the last
            x_pur = prior.purify(x_adv, strength=purify_strength)
            # reproject purified image back into the eps-ball of x_orig
            x_adv = torch.clamp(x_orig + torch.clamp(x_pur - x_orig, -eps_f, eps_f), 0.0, 1.0).detach()
    return x_adv.detach()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--img-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--anchor-mode", choices=("product-title", "category", "random-text"), default="product-title")
    p.add_argument("--eps", type=int, default=16, help="L_inf budget on image in /255 units")
    p.add_argument("--alpha", type=int, default=4, help="image step size in /255 units")
    p.add_argument("--blocks", type=int, default=5, help="PGD blocks (purify happens between blocks)")
    p.add_argument("--steps-per-block", type=int, default=40, help="PGD steps per block (~200 total)")
    p.add_argument("--purify-strength", type=float, default=0.15, help="diffusion noise fraction [0,1]")
    p.add_argument("--purify-steps", type=int, default=6, help="DDIM steps in the purifier")
    p.add_argument("--no-diffusion", action="store_true", help="ablation: plain PGD (no diffusion prior)")
    p.add_argument("--sd-path", default=DEFAULT_SD_PATH)
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
    LOG.info("diffprior eps=%d alpha=%d blocks=%d steps/block=%d purify(strength=%.2f,steps=%d,enabled=%s) device=%s",
             args.eps, args.alpha, args.blocks, args.steps_per_block, args.purify_strength,
             args.purify_steps, not args.no_diffusion, device)

    df = pd.read_csv(args.manifest).copy()
    df["_img_path"] = df["product_id"].apply(lambda a: args.img_dir / f"{a}.jpg")
    have = df["_img_path"].apply(lambda q: q.exists() and q.stat().st_size > 0)
    if not have.all():
        LOG.warning("dropping %d pairs whose original image is missing", int((~have).sum()))
        df = df[have].reset_index(drop=True)
    df = df.drop_duplicates(subset=["product_id"]).reset_index(drop=True)
    if args.max_images is not None:
        df = df.head(args.max_images).reset_index(drop=True)
    LOG.info("attacking %d unique images", len(df))

    if args.anchor_mode == "product-title":
        anchor_texts = df["product_title"].astype(str).tolist()
    elif args.anchor_mode == "category":
        col = "product_brand" if "product_brand" in df.columns else "product_title"
        anchor_texts = df[col].fillna(df["product_title"]).astype(str).tolist()
    else:
        rng = np.random.RandomState(args.seed)
        vocab = ["object", "item", "thing", "product", "model", "design", "device", "tool", "set", "kit"]
        anchor_texts = [" ".join(rng.choice(vocab, size=4)) for _ in range(len(df))]

    model, tokenizer = load_clip(args.clip_backbone, device)
    norm_fn = make_clip_norm(args.image_size, device)
    anchor_feats = encode_anchors(model, tokenizer, anchor_texts, device)
    LOG.info("encoded %d anchors", len(anchor_texts))

    prior = None
    if not args.no_diffusion:
        t0 = time.time()
        prior = DiffusionPrior(args.sd_path, device, steps=args.purify_steps)
        LOG.info("loaded SD diffusion prior in %.1fs", time.time() - t0)

    eps_f = args.eps / 255.0
    alpha_f = args.alpha / 255.0
    rows = []
    t0_all = time.time()
    for idx, row in df.iterrows():
        asin = row["product_id"]
        out_path = out_img_dir / f"{asin}.jpg"
        if out_path.exists() and out_path.stat().st_size > 0:
            rows.append({"product_id": asin, "out_path": str(out_path), "skipped": True, "seconds": 0.0})
            continue
        x = load_image(row["_img_path"], args.image_size).to(device)
        v = anchor_feats[idx:idx + 1]
        t0 = time.time()
        with torch.enable_grad():
            x_adv = attack_diffprior(x, v, model, norm_fn, prior, eps_f, alpha_f,
                                     args.blocks, args.steps_per_block, args.purify_strength)
        save_image(x_adv, out_path)
        rows.append({"product_id": asin, "out_path": str(out_path), "skipped": False,
                     "seconds": round(time.time() - t0, 3)})
        if (idx + 1) % 25 == 0:
            el = time.time() - t0_all
            LOG.info("  diffprior %d/%d in %.1fs (avg %.2fs/img)", idx + 1, len(df), el, el / (idx + 1))

    log_csv = args.out_dir / "attack_log.csv"
    with open(log_csv, "w") as fh:
        w = csv.DictWriter(fh, fieldnames=["product_id", "out_path", "skipped", "seconds"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})

    summary = {
        "method": "diffprior" if not args.no_diffusion else "diffprior_ablate_plainpgd",
        "anchor_mode": args.anchor_mode, "eps": args.eps, "alpha": args.alpha,
        "blocks": args.blocks, "steps_per_block": args.steps_per_block,
        "total_pgd_steps": args.blocks * args.steps_per_block,
        "purify_strength": args.purify_strength, "purify_steps": args.purify_steps,
        "diffusion_enabled": (not args.no_diffusion), "sd_path": args.sd_path,
        "n_images": int(len(df)),
        "n_attacked": int(sum(1 for r in rows if not r["skipped"])),
        "n_skipped": int(sum(1 for r in rows if r["skipped"])),
        "wall_seconds": round(time.time() - t0_all, 2),
        "out_img_dir": str(out_img_dir), "clip_backbone": args.clip_backbone,
        "image_size": args.image_size, "manifest": str(args.manifest),
    }
    (args.out_dir / "attack_summary.json").write_text(json.dumps(summary, indent=2))
    LOG.info("attack summary: %s", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
