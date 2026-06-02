#!/usr/bin/env python3
"""NES (Natural Evolution Strategy) gradient-free black-box attack -- ceiling check.

Anchor-decoupled: fitness = cosine( CLIP_image(x+delta), CLIP_text(product_title) ).
The product TITLE is the anchor. The eval-time query q is NEVER used in the optimization
loop (held out). Forward-only: only model.encode_image is called; no CLIP gradients.

L_inf projection to eps = 16/255 in pixel space. We perturb the raw uint8 RGB pixels of
the resized-on-disk thumbnail, save the best as JPG (same format the eval harness reads).
Fitness uses a GPU re-implementation of the exact open_clip ViT-L/14 preprocessing
(Resize224 bicubic+antialias -> CenterCrop224 -> CLIP normalize), so the optimization
pipeline matches what the eval harness sees, while being fast enough on a 4090.

Budget: population = 2*pop_half (antithetic) per iter, over `iters` iters.
Default 40 x 50 = 2000 forward image-encodes per image (reported exactly).
"""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

LABELS = ("E", "S", "C", "I")

# open_clip ViT-L-14 openai normalize constants (from create_model_and_transforms)
_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def load_clip(backbone, device):
    import open_clip
    reg = {
        "openai/clip-vit-large-patch14": ("ViT-L-14", "openai"),
        "ViT-L-14": ("ViT-L-14", "openai"),
    }
    arch, pretrained = reg[backbone]
    model, _, preproc = open_clip.create_model_and_transforms(arch, pretrained=pretrained, device=str(device))
    tok = open_clip.get_tokenizer(arch)
    model.eval()
    return model, preproc, tok


@torch.no_grad()
def encode_text(model, tok, texts, device):
    t = tok(texts).to(device)
    f = model.encode_text(t)
    f = f / f.norm(dim=-1, keepdim=True)
    return f


def _gpu_preprocess(adv_bchw, device):
    """adv_bchw: float [B,3,H,W] in [0,255] on device. Returns CLIP-normalized 224x224."""
    x = adv_bchw / 255.0
    x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False, antialias=True)
    x = x.clamp(0, 1)
    return (x - _CLIP_MEAN.to(device)) / _CLIP_STD.to(device)


@torch.no_grad()
def encode_pils(model, preproc, pils, device, batch=64):
    """CPU-preproc encoder, retained only for optional sanity checks."""
    feats = []
    for i in range(0, len(pils), batch):
        chunk = pils[i:i + batch]
        x = torch.stack([preproc(im) for im in chunk], dim=0).to(device)
        f = model.encode_image(x)
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f)
    return torch.cat(feats, dim=0)


def nes_attack_one(model, device, base_u8, anchor_feat,
                   eps, sigma, pop_half, iters, lr, rng, fwd_batch=80):
    """base_u8: HxWx3 uint8. GPU-batched forward-only NES.
    Returns (best_adv_u8, n_forward, fit_orig, fit_best)."""
    H, W, C = base_u8.shape
    eps_px = eps * 255.0
    sigma_px = sigma * 255.0
    base_t = torch.from_numpy(base_u8.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)
    delta = torch.zeros((1, C, H, W), dtype=torch.float32, device=device)
    n_forward = 0

    @torch.no_grad()
    def fitness_batch(deltas):
        nonlocal n_forward
        N = deltas.shape[0]
        out = torch.empty(N, device=device)
        for s in range(0, N, fwd_batch):
            d = deltas[s:s + fwd_batch]
            adv = torch.clamp(base_t + d, 0, 255).round()  # uint8 grid
            x = _gpu_preprocess(adv, device)
            f = model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
            out[s:s + d.shape[0]] = (f @ anchor_feat.T).squeeze(-1)
        n_forward += N
        return out

    fit_orig = float(fitness_batch(torch.zeros((1, C, H, W), device=device))[0].item())

    gen = torch.Generator(device=device)
    gen.manual_seed(int(rng.integers(0, 2**31 - 1)))
    for _ in range(iters):
        noise = torch.randn((pop_half, C, H, W), generator=gen, device=device) * sigma_px
        cand = torch.empty((2 * pop_half, C, H, W), device=device)
        cand[0::2] = torch.clamp(delta + noise, -eps_px, eps_px)
        cand[1::2] = torch.clamp(delta - noise, -eps_px, eps_px)
        fits = fitness_batch(cand)
        adv_score = fits[0::2] - fits[1::2]  # antithetic advantage
        adv_norm = adv_score / (adv_score.std() + 1e-8)
        grad = (adv_norm.view(pop_half, 1, 1, 1) * noise).sum(0, keepdim=True) / (pop_half * sigma_px)
        delta = torch.clamp(delta + lr * eps_px * grad, -eps_px, eps_px)

    fit_best = float(fitness_batch(delta)[0].item())
    best_adv = torch.clamp(base_t + delta, 0, 255).round().squeeze(0).permute(1, 2, 0).byte().cpu().numpy()
    return best_adv, n_forward, fit_orig, fit_best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--out-adv-dir", required=True)
    ap.add_argument("--subset-csv", required=True)
    ap.add_argument("--clip-backbone", default="openai/clip-vit-large-patch14")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--eps", type=float, default=16/255)
    ap.add_argument("--sigma", type=float, default=8/255)
    ap.add_argument("--pop-half", type=int, default=20)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--n-subset", type=int, default=120)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--progress-json", default="")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    img_dir = Path(args.img_dir)
    out_adv = Path(args.out_adv_dir); out_adv.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.manifest)
    df["_p"] = df["product_id"].apply(lambda a: img_dir / f"{a}.jpg")
    df = df[df["_p"].apply(lambda p: p.exists() and p.stat().st_size > 0)].reset_index(drop=True)

    # cohort-balanced subset: proportional per-label, fixed seed.
    total = len(df)
    sel_idx = []
    sub_rng = np.random.default_rng(args.seed)
    for lab in LABELS:
        sub = df[df["esci_label"] == lab]
        if len(sub) == 0:
            continue
        k = min(max(1, round(args.n_subset * len(sub) / total)), len(sub))
        pick = sub_rng.choice(sub.index.to_numpy(), size=k, replace=False)
        sel_idx.extend(pick.tolist())
    sel_idx = sorted(sel_idx)
    sub_df = df.loc[sel_idx].reset_index(drop=True)
    sub_df[["example_id", "query_id", "product_id", "esci_label"]].to_csv(args.subset_csv, index=False)
    print(f"SUBSET n={len(sub_df)} labeldist={sub_df['esci_label'].value_counts().to_dict()}", flush=True)

    model, preproc, tok = load_clip(args.clip_backbone, device)
    print("CLIP loaded", flush=True)

    # Copy ALL originals into adv dir so the rank distractor pool == CoGEO eval pool,
    # then overwrite the subset ASINs with NES adv images.
    for a in df["product_id"].tolist():
        dst = out_adv / f"{a}.jpg"
        if not dst.exists():
            Image.open(img_dir / f"{a}.jpg").convert("RGB").save(dst, "JPEG", quality=95)
    print(f"copied {df['product_id'].nunique()} originals into adv pool", flush=True)

    attack_rows = sub_df.head(args.limit) if args.limit > 0 else sub_df

    log = []
    t0 = time.time()
    nforward_total = 0
    for i, row in attack_rows.iterrows():
        a = row["product_id"]; title = str(row["product_title"])
        anchor = encode_text(model, tok, [title], device)
        base_u8 = np.array(Image.open(img_dir / f"{a}.jpg").convert("RGB"), dtype=np.uint8)
        adv_u8, nfwd, fit0, fitb = nes_attack_one(
            model, device, base_u8, anchor,
            args.eps, args.sigma, args.pop_half, args.iters, args.lr, rng)
        Image.fromarray(adv_u8, "RGB").save(out_adv / f"{a}.jpg", "JPEG", quality=95)
        linf = int(np.abs(adv_u8.astype(int) - base_u8.astype(int)).max())
        nforward_total += nfwd
        log.append({"example_id": int(row["example_id"]), "product_id": a,
                    "esci_label": row["esci_label"], "fit_orig": fit0,
                    "fit_adv": fitb, "fit_gain": fitb - fit0,
                    "n_forward": nfwd, "linf_px": linf})
        if (i + 1) % 5 == 0 or i == 0:
            el = time.time() - t0
            print(f"[{i+1}/{len(attack_rows)}] {a} fit {fit0:.4f}->{fitb:.4f} (+{fitb-fit0:.4f}) "
                  f"linf={linf} nfwd={nfwd} elapsed={el:.0f}s", flush=True)
            if args.progress_json:
                Path(args.progress_json).write_text(json.dumps({"done": i+1, "total": len(attack_rows), "elapsed_s": el}))

    logdf = pd.DataFrame(log)
    logdf.to_csv(out_adv.parent / "nes_attack_log.csv", index=False)
    summ = {
        "n_attacked": len(log),
        "forward_per_image": int(2 * args.pop_half * args.iters),
        "forward_per_image_incl_baseline_and_final": int(2 * args.pop_half * args.iters + 2),
        "nforward_total": int(nforward_total),
        "eps": args.eps, "eps_px": args.eps * 255,
        "sigma": args.sigma, "pop_half": args.pop_half,
        "population": 2 * args.pop_half, "iters": args.iters, "lr": args.lr,
        "fit_gain_mean": float(logdf["fit_gain"].mean()) if len(logdf) else None,
        "fit_gain_pos_rate": float((logdf["fit_gain"] > 0).mean()) if len(logdf) else None,
        "linf_max": int(logdf["linf_px"].max()) if len(logdf) else None,
        "clip_backbone": args.clip_backbone,
        "elapsed_s": time.time() - t0,
    }
    (out_adv.parent / "nes_attack_summary.json").write_text(json.dumps(summ, indent=2))
    print("ATTACK_SUMMARY " + json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
