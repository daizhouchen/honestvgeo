"""cogeo — CoGEO Algorithm 1 (paper §4) re-implementation.

Phase 1: VLM anchor (vlm_anchor.py).
Phase 2: Sobel texture mask (here).
Phase 3: MI-FGSM PGD with environment simulation (envsim.py + here).

Output: optimized image x* + per-image metrics CSV row.

Usage:
  python cogeo.py --dataset-root <dir> --out-root <dir> --epsilon 16 --alpha 4 --iters 200 \\
                  --vlm Qwen/Qwen2.5-VL-7B-Instruct --gpu auto --batch-size 4 --resume

Inputs are expected as 800x800 PNG files inside <dataset-root>; outputs go under
<out-root>/{adv,anchor,metrics.csv}.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# Local modules
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from envsim import EnvSim  # noqa: E402
from vlm_anchor import VLMAnchor  # noqa: E402


# ----------------------- Phase 2: texture mask -----------------------

def texture_mask(x: torch.Tensor, floor: float = 0.1) -> torch.Tensor:
    """x: (B,3,H,W) in [0,1]. Returns mask M of shape (B,1,H,W) in [floor,1]."""
    # Sobel kernels
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device)
    # luminance
    lum = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    gx = F.conv2d(F.pad(lum, (1, 1, 1, 1), mode="replicate"), kx[None, None])
    gy = F.conv2d(F.pad(lum, (1, 1, 1, 1), mode="replicate"), ky[None, None])
    g = torch.sqrt(gx ** 2 + gy ** 2 + 1e-12)
    # min-max normalize per-image
    B = g.shape[0]
    g_flat = g.view(B, -1)
    gmin = g_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    gmax = g_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    gn = (g - gmin) / (gmax - gmin + 1e-12)
    return torch.clamp(gn, min=floor)


# ----------------------- CLIP image preprocessing as a torch op -----------------------
# We need a *differentiable* preprocessing that converts our 800x800 [0,1] tensor
# into the CLIP-expected 224x224 normalized tensor — without using PIL inside the loop.

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def clip_preprocess(x: torch.Tensor, size: int = 224) -> torch.Tensor:
    """x: (B,3,H,W) in [0,1] (any H,W). Output: (B,3,size,size) normalized for CLIP."""
    if x.shape[-2:] != (size, size):
        x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False, antialias=True)
    x = x.clamp(0.0, 1.0)
    mean = torch.tensor(CLIP_MEAN, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


# ----------------------- Algorithm 1 -----------------------

def cogeo_step(
    clip: CLIPModel,
    x_orig: torch.Tensor,        # (B,3,H,W) in [0,1], on device
    v_anchor: torch.Tensor,      # (B,proj_dim), L2-normalized
    epsilon: float,
    alpha: float,
    iters: int,
    envsim: EnvSim,
    momentum_mu: float = 1.0,
    sim_scales=(0.5, 0.25, 0.125, 0.0625),  # 1/2 .. 1/16
    use_sim: bool = True,
) -> torch.Tensor:
    """Returns x_adv = clamp(x_orig + delta * M, 0, 1), per Algorithm 1 line 14."""
    M = texture_mask(x_orig)  # (B,1,H,W)
    delta = torch.zeros_like(x_orig, requires_grad=True)
    momentum = torch.zeros_like(x_orig)

    for _t in range(iters):
        x_adv = (x_orig + delta * M).clamp(0.0, 1.0)
        x_sys = envsim(x_adv)

        # Multi-scale gradient averaging (SIM)
        loss_total = 0.0
        scales = list(sim_scales) if use_sim else [1.0]
        for s in scales:
            if s != 1.0:
                xs = F.interpolate(x_sys, scale_factor=s, mode="bilinear",
                                   align_corners=False, antialias=True)
            else:
                xs = x_sys
            xs_clip = clip_preprocess(xs)
            z = clip.get_image_features(pixel_values=xs_clip)
            z = F.normalize(z, dim=-1)
            cos = (z * v_anchor).sum(dim=-1)  # (B,)
            loss_s = (1.0 - cos).mean()
            loss_total = loss_total + loss_s
        loss_total = loss_total / len(scales)

        grad = torch.autograd.grad(loss_total, delta)[0]
        # L1-normalize per image
        g_norm = grad / (grad.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-8)
        momentum = momentum_mu * momentum + g_norm
        delta = (delta - alpha * momentum.sign()).detach()
        delta = delta.clamp(-epsilon, epsilon)
        delta.requires_grad_(True)

    return (x_orig + delta * M).clamp(0.0, 1.0).detach(), M.detach()


# ----------------------- IO helpers -----------------------

def load_image_tensor(path: Path, size: int = 800, device: str = "cuda") -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return t  # (1,3,H,W) in [0,1]


def save_image_tensor(t: torch.Tensor, path: Path):
    t = t.detach().clamp(0, 1).cpu()
    arr = (t.permute(0, 2, 3, 1).numpy() * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr[0]).save(path)


def cos_sim(z1: torch.Tensor, z2: torch.Tensor) -> float:
    return F.cosine_similarity(F.normalize(z1, dim=-1), z2, dim=-1).item()


# ----------------------- Main -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--epsilon", type=float, default=16.0, help="numerator of /255")
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--vlm", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--clip-id", default="openai/clip-vit-base-patch32")
    ap.add_argument("--limit", type=int, default=0, help="cap N images (0=all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="single-image smoke test")
    ap.add_argument("--no-envsim", action="store_true", help="ablate env-sim")
    ap.add_argument("--no-sim", action="store_true", help="ablate SIM multi-scale gradient avg")
    ap.add_argument("--no-mask", action="store_true", help="ablate texture mask (set M=1)")
    ap.add_argument("--anchor-mode", choices=["vlm", "category", "generic"], default="vlm")
    ap.add_argument("--category-name", default="product")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    out_root = Path(args.out_root)
    (out_root / "adv").mkdir(parents=True, exist_ok=True)
    (out_root / "anchor").mkdir(parents=True, exist_ok=True)
    # COGEO_PATCH:manifest_categories_load
    manifest_path = Path(args.dataset_root) / "manifest.json"
    manifest_categories = {}
    if manifest_path.exists():
        import json as _json
        with manifest_path.open() as _mf:
            _m = _json.load(_mf)
        for _item in _m.get("items", []):
            _cat = _item.get("category") or _item.get("category_raw")
            if _cat:
                manifest_categories[_item["image_id"]] = _cat
        print(f"[cogeo] loaded {len(manifest_categories)} per-image categories from manifest.json")

    metrics_path = out_root / "metrics.csv"

    # Resume support: collect already-done image ids
    done = set()
    if metrics_path.exists():
        with metrics_path.open() as f:
            for row in csv.DictReader(f):
                done.add(row["image_id"])

    # ---------- enumerate dataset ----------
    src = Path(args.dataset_root)
    paths = sorted([p for p in src.glob("*.png")])
    if args.limit:
        paths = paths[: args.limit]
    if args.smoke:
        paths = paths[:1]
    print(f"[cogeo] dataset={src} | n_images={len(paths)} | already_done={len(done)} | iters={args.iters}")

    device = "cuda"

    # ---------- VLM + CLIP ----------
    print(f"[cogeo] loading anchor stack: vlm={args.vlm}, clip={args.clip_id}")
    t0 = time.time()
    if args.anchor_mode == "vlm":
        anchor_stack = VLMAnchor(vlm_model_id=args.vlm, clip_model_id=args.clip_id, device=device)
    else:
        anchor_stack = None
    clip = CLIPModel.from_pretrained(args.clip_id).eval().to(device)
    clip_proc = CLIPProcessor.from_pretrained(args.clip_id)
    print(f"[cogeo] models loaded in {time.time()-t0:.1f}s")

    # ---------- env-sim ----------
    envsim = EnvSim(p=0.0 if args.no_envsim else 0.5).to(device)  # COGEO_PATCH:envsim_to_device

    # ---------- main loop ----------
    eps = args.epsilon / 255.0
    alpha = args.alpha / 255.0

    new_rows = []
    fout = metrics_path.open("a", newline="")
    writer = csv.writer(fout)
    if metrics_path.stat().st_size == 0:
        writer.writerow([
            "image_id", "anchor_mode", "anchor_text", "delta_clip_sim",
            "cos_orig_anchor", "cos_adv_anchor",
            "ssim", "lpips", "iters", "epsilon", "alpha",
            "no_envsim", "no_sim", "no_mask", "elapsed_s",
        ])

    # SSIM + LPIPS via piq (already verified installed)
    import piq

    with torch.no_grad():
        for p in paths:
            iid = p.stem
            if iid in done:
                print(f"[cogeo] skip done {iid}")
                continue

            t_start = time.time()
            x_orig = load_image_tensor(p, size=800, device=device)

            # Phase 1: anchor
            if args.anchor_mode == "vlm":
                pil = Image.open(p).convert("RGB").resize((800, 800), Image.BICUBIC)
                anchor_text = anchor_stack.caption(pil)
                v_anchor = anchor_stack.encode_anchor_text(anchor_text)  # (1, proj_dim)
                # save anchor
                with (out_root / "anchor" / f"{iid}.txt").open("w") as f:
                    f.write(anchor_text)
            elif args.anchor_mode == "category":
                anchor_text = manifest_categories.get(iid, args.category_name)  # COGEO_PATCH:manifest_categories_lookup
                toks = clip_proc.tokenizer(anchor_text, return_tensors="pt", truncation=True,
                                            max_length=77, padding=True).to(device)
                z = clip.get_text_features(**toks)
                v_anchor = F.normalize(z, dim=-1)
            else:  # generic
                anchor_text = "Recommended"
                toks = clip_proc.tokenizer(anchor_text, return_tensors="pt", truncation=True,
                                            max_length=77, padding=True).to(device)
                z = clip.get_text_features(**toks)
                v_anchor = F.normalize(z, dim=-1)

            # Phase 2/3 inside cogeo_step
            with torch.enable_grad():
                x_adv, mask = cogeo_step(
                    clip=clip,
                    x_orig=x_orig,
                    v_anchor=v_anchor,
                    epsilon=eps,
                    alpha=alpha,
                    iters=args.iters,
                    envsim=envsim,
                    use_sim=not args.no_sim,
                )
                if args.no_mask:
                    # re-run with M=1 (skip mask) — effective ablation: pass override mask of ones
                    raise NotImplementedError(
                        "no-mask ablation should be implemented by overriding texture_mask in cogeo_step"
                    )

            # ---------- evaluate ----------
            xc = clip_preprocess(x_orig)
            zc = F.normalize(clip.get_image_features(pixel_values=xc), dim=-1)
            cos_orig = (zc * v_anchor).sum(dim=-1).item()

            xa = clip_preprocess(x_adv)
            za = F.normalize(clip.get_image_features(pixel_values=xa), dim=-1)
            cos_adv = (za * v_anchor).sum(dim=-1).item()

            delta_sim = cos_adv - cos_orig

            # SSIM / LPIPS on the 800x800 raw images
            ssim_val = piq.ssim(x_orig, x_adv, data_range=1.0).item()
            try:  # COGEO_PATCH:lpips_optional
                lpips_val = piq.LPIPS()(x_orig, x_adv).item()
            except Exception as _lpips_err:
                print(f"[cogeo] LPIPS skipped: {type(_lpips_err).__name__}")
                lpips_val = float("nan")

            elapsed = time.time() - t_start
            row = [
                iid, args.anchor_mode, (anchor_text or "")[:200], f"{delta_sim:.6f}",
                f"{cos_orig:.6f}", f"{cos_adv:.6f}",
                f"{ssim_val:.6f}", f"{lpips_val:.6f}", args.iters, args.epsilon, args.alpha,
                int(args.no_envsim), int(args.no_sim), int(args.no_mask), f"{elapsed:.1f}",
            ]
            writer.writerow(row)
            fout.flush()
            print(f"[cogeo] {iid}  Δ={delta_sim:+.4f} ssim={ssim_val:.4f} lpips={lpips_val:.4f}  ({elapsed:.1f}s)")

            save_image_tensor(x_adv, out_root / "adv" / f"{iid}.png")

    fout.close()

    # ---------- final aggregate ----------
    deltas, ssims, lpipses = [], [], []
    with metrics_path.open() as f:
        for row in csv.DictReader(f):
            deltas.append(float(row["delta_clip_sim"]))
            ssims.append(float(row["ssim"]))
            lpipses.append(float(row["lpips"]))

    summary = {
        "n": len(deltas),
        "delta_clip_sim_mean": float(np.mean(deltas)) if deltas else None,
        "delta_clip_sim_ci95": float(1.96 * np.std(deltas, ddof=1) / max(len(deltas) ** 0.5, 1.0)) if len(deltas) > 1 else None,
        "ssim_mean": float(np.mean(ssims)) if ssims else None,
        "lpips_mean": float(np.mean(lpipses)) if lpipses else None,
        "anchor_mode": args.anchor_mode,
        "iters": args.iters,
        "epsilon": args.epsilon,
        "alpha": args.alpha,
        "no_envsim": args.no_envsim,
        "no_sim": args.no_sim,
        "no_mask": args.no_mask,
        "vlm_model_id": args.vlm if args.anchor_mode == "vlm" else None,
        "clip_model_id": args.clip_id,
    }
    with (out_root / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
