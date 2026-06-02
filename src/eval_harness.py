#!/usr/bin/env python3
"""ESCI eval harness for the N3 visual-GEO real-relevance experiment.

Two modes:

* ``no-op`` -- pre-flight sanity gate. Computes mean cos-similarity between
  CLIP_image(x_p) and CLIP_text(q) for every (query, product, label) pair
  in the manifest, without any adversarial perturbation. Reports mean per
  label and the gap ``mean_sim(E) - mean_sim(I)``. Healthy retrieval
  signal requires this gap >= 0.05.

* ``rank`` -- full eval. Loads adversarial images x_p* from --adv-dir
  alongside originals, computes per-pair Delta_clip_sim_held_out, and
  ranks x_p (orig) and x_p* against a fixed distractor pool. Outputs
  per-pair CSV plus per-method aggregate. (Only used after No-Op gate
  passes; current entry point implements ``no-op`` as the gate.)

Protocol invariant (non-negotiable):

    v_anchor(x_p) is computed only from x_p (NEVER from q).
    The eval query q comes from the manifest; it is held out from the
    optimization loop.

Usage::

    python eval_harness.py --mode no-op \\
        --manifest /data/esci500/manifest.csv \\
        --img-dir  /data/esci500/img \\
        --out-dir  /runs/n3_sanity \\
        --clip-backbone openai/clip-vit-base-patch32

After ``no-op`` gates, ``rank`` mode adds ``--adv-dir`` (where method/eps
subdirs hold optimized images) and ``--methods cogeo,pgd_bare,...``.
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
from PIL import Image

LOG = logging.getLogger("eval_harness")

LABELS = ("E", "S", "C", "I")


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


def load_clip(backbone: str, device: torch.device):
    """Load CLIP via openclip (preferred) or transformers."""
    try:
        import open_clip  # type: ignore
        if backbone not in _BACKBONE_REGISTRY:
            raise ValueError(f"unknown openclip backbone: {backbone}")
        arch, pretrained = _BACKBONE_REGISTRY[backbone]
        model, _, preproc = open_clip.create_model_and_transforms(
            arch, pretrained=pretrained, device=str(device))
        tokenizer = open_clip.get_tokenizer(arch)
        mode = "open_clip"
        model.eval()
        LOG.info("loaded CLIP via open_clip backbone=%s arch=%s pretrained=%s", backbone, arch, pretrained)
        return {"model": model, "preproc": preproc, "tokenizer": tokenizer, "mode": mode}
    except Exception as e:
        LOG.warning("open_clip load failed (%s); trying transformers", e)
    from transformers import CLIPModel, CLIPProcessor  # type: ignore
    model = CLIPModel.from_pretrained(backbone).to(device).eval()
    processor = CLIPProcessor.from_pretrained(backbone)
    LOG.info("loaded CLIP via transformers backbone=%s", backbone)
    return {"model": model, "processor": processor, "mode": "hf"}


def encode_images(clip_pkg, paths: list[Path], device: torch.device, batch: int = 32) -> torch.Tensor:
    feats = []
    if clip_pkg["mode"] == "open_clip":
        preproc = clip_pkg["preproc"]
        model = clip_pkg["model"]
        for i in range(0, len(paths), batch):
            chunk = paths[i:i + batch]
            tensors = []
            for p in chunk:
                img = Image.open(p).convert("RGB")
                tensors.append(preproc(img))
            x = torch.stack(tensors, dim=0).to(device)
            with torch.no_grad():
                f = model.encode_image(x)
                f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu())
    else:
        processor = clip_pkg["processor"]
        model = clip_pkg["model"]
        for i in range(0, len(paths), batch):
            chunk = paths[i:i + batch]
            imgs = [Image.open(p).convert("RGB") for p in chunk]
            inputs = processor(images=imgs, return_tensors="pt").to(device)
            with torch.no_grad():
                f = model.get_image_features(**inputs)
                f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu())
    return torch.cat(feats, dim=0)


def encode_texts(clip_pkg, texts: list[str], device: torch.device, batch: int = 64) -> torch.Tensor:
    feats = []
    if clip_pkg["mode"] == "open_clip":
        tokenizer = clip_pkg["tokenizer"]
        model = clip_pkg["model"]
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            tok = tokenizer(chunk).to(device)
            with torch.no_grad():
                f = model.encode_text(tok)
                f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu())
    else:
        processor = clip_pkg["processor"]
        model = clip_pkg["model"]
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            inputs = processor(text=chunk, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                f = model.get_text_features(**inputs)
                f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu())
    return torch.cat(feats, dim=0)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("no-op", "rank"), default="no-op")
    p.add_argument("--manifest", type=Path, required=True, help="ESCI manifest CSV (output of esci_loader.py)")
    p.add_argument("--img-dir", type=Path, required=True, help="Cache dir of original product images")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--clip-backbone", default="openai/clip-vit-base-patch32")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    # rank-mode only
    p.add_argument("--adv-dir", type=Path, default=None,
                   help="(rank-mode) directory holding adversarial images named {ASIN}.jpg")
    p.add_argument("--method-tag", type=str, default="adv",
                   help="(rank-mode) method tag for output filenames")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    LOG.info("device=%s mode=%s", device, args.mode)

    df = pd.read_csv(args.manifest)
    LOG.info("manifest rows=%d  labels=%s", len(df), df["esci_label"].value_counts().to_dict())

    # Drop pairs whose image file is missing.
    df = df.copy()
    df["_img_path"] = df["product_id"].apply(lambda a: args.img_dir / f"{a}.jpg")
    have_img = df["_img_path"].apply(lambda p: p.exists() and p.stat().st_size > 0)
    if not have_img.all():
        missing = int((~have_img).sum())
        LOG.warning("dropping %d pairs whose image is missing", missing)
        df = df[have_img].reset_index(drop=True)
    if len(df) == 0:
        LOG.error("no usable pairs after filter; aborting")
        return 2

    clip_pkg = load_clip(args.clip_backbone, device)

    # Encode each unique image and each query exactly once.
    asins = df["product_id"].drop_duplicates().tolist()
    asin2idx = {a: i for i, a in enumerate(asins)}
    img_paths = [args.img_dir / f"{a}.jpg" for a in asins]
    LOG.info("encoding %d unique images", len(img_paths))
    t0 = time.time()
    img_feats = encode_images(clip_pkg, img_paths, device, args.batch)
    LOG.info("image encode done %.1fs shape=%s", time.time() - t0, tuple(img_feats.shape))

    queries = df["query"].drop_duplicates().tolist()
    q2idx = {q: i for i, q in enumerate(queries)}
    LOG.info("encoding %d unique queries", len(queries))
    t0 = time.time()
    txt_feats = encode_texts(clip_pkg, queries, device, args.batch * 2)
    LOG.info("text encode done %.1fs shape=%s", time.time() - t0, tuple(txt_feats.shape))

    # Per-pair similarity at original image.
    img_idx = df["product_id"].map(asin2idx).to_numpy()
    txt_idx = df["query"].map(q2idx).to_numpy()
    sims_orig = (img_feats[img_idx] * txt_feats[txt_idx]).sum(dim=-1).numpy()
    df["sim_orig"] = sims_orig

    if args.mode == "no-op":
        # Pre-flight sanity gate.
        per_label_mean = {lab: float(df.loc[df["esci_label"] == lab, "sim_orig"].mean())
                          for lab in LABELS if (df["esci_label"] == lab).any()}
        per_label_n = {lab: int((df["esci_label"] == lab).sum()) for lab in LABELS}
        mean_E = per_label_mean.get("E", float("nan"))
        mean_I = per_label_mean.get("I", float("nan"))
        gap = mean_E - mean_I
        gate_pass = bool(gap >= 0.05)
        summary = {
            "mode": "no-op",
            "manifest": str(args.manifest),
            "img_dir": str(args.img_dir),
            "clip_backbone": args.clip_backbone,
            "n_pairs": int(len(df)),
            "per_label_n": per_label_n,
            "per_label_mean_sim": per_label_mean,
            "mean_sim_E": mean_E,
            "mean_sim_I": mean_I,
            "gap_E_minus_I": gap,
            "gate_threshold": 0.05,
            "gate_pass": gate_pass,
        }
        out_csv = args.out_dir / "no_op_per_pair.csv"
        df[["example_id", "query_id", "query", "product_id", "product_title",
            "esci_label", "sim_orig"]].to_csv(out_csv, index=False, quoting=csv.QUOTE_NONNUMERIC)
        out_json = args.out_dir / "no_op_summary.json"
        out_json.write_text(json.dumps(summary, indent=2))
        LOG.info("wrote %s", out_csv)
        LOG.info("wrote %s", out_json)
        LOG.info("no-op summary: %s", json.dumps(summary, indent=2))
        if gate_pass:
            LOG.info("✓ pre-flight sanity gate PASSED (gap %.4f >= 0.05) -- proceed to adversarial cells", gap)
            return 0
        else:
            LOG.warning("✗ pre-flight sanity gate FAILED (gap %.4f < 0.05) -- consider swapping CLIP backbone", gap)
            return 3  # soft-fail; caller decides

    # mode == "rank": needs --adv-dir.
    if args.adv_dir is None:
        LOG.error("rank-mode requires --adv-dir")
        return 1
    adv_paths_in_order = []
    missing_adv = []
    for a in asins:
        ap = args.adv_dir / f"{a}.jpg"
        if ap.exists() and ap.stat().st_size > 0:
            adv_paths_in_order.append(ap)
        else:
            missing_adv.append(a)
    if missing_adv:
        LOG.warning("missing %d adversarial images for ASINs (first 5: %s)",
                    len(missing_adv), missing_adv[:5])
        # filter df to ASINs whose adv exists
        keep = ~df["product_id"].isin(missing_adv)
        df = df[keep].reset_index(drop=True)
        asins = df["product_id"].drop_duplicates().tolist()
        asin2idx = {a: i for i, a in enumerate(asins)}
        adv_paths_in_order = [args.adv_dir / f"{a}.jpg" for a in asins]
        img_paths_in_order = [args.img_dir / f"{a}.jpg" for a in asins]
        img_feats = encode_images(clip_pkg, img_paths_in_order, device, args.batch)

    LOG.info("encoding %d adversarial images", len(adv_paths_in_order))
    adv_feats = encode_images(clip_pkg, adv_paths_in_order, device, args.batch)

    img_idx = df["product_id"].map(asin2idx).to_numpy()
    txt_idx = df["query"].map(q2idx).to_numpy()

    sims_orig = (img_feats[img_idx] * txt_feats[txt_idx]).sum(dim=-1).numpy()
    sims_adv = (adv_feats[img_idx] * txt_feats[txt_idx]).sum(dim=-1).numpy()
    df["sim_orig"] = sims_orig
    df["sim_adv"] = sims_adv
    df["delta_held_out"] = sims_adv - sims_orig

    # Rank within fixed distractor pool: every other unique image in this batch
    # is a distractor for ranking. (Cheap; cross-query so a real product image
    # competes with all others in the sample.)
    sims_pool_orig = (img_feats @ txt_feats.T).numpy()  # [N_img, N_query]
    sims_pool_adv = (adv_feats @ txt_feats.T).numpy()
    rank_orig = np.empty(len(df), dtype=int)
    rank_adv = np.empty(len(df), dtype=int)
    for r in range(len(df)):
        i = img_idx[r]
        q = txt_idx[r]
        orig_col = sims_pool_orig[:, q].copy()
        adv_col = sims_pool_orig[:, q].copy()
        adv_col[i] = sims_pool_adv[i, q]  # only this row swapped to adv
        rank_orig[r] = (orig_col >= orig_col[i]).sum()
        rank_adv[r] = (adv_col >= adv_col[i]).sum()
    df["rank_orig"] = rank_orig
    df["rank_adv"] = rank_adv
    df["rank_lift"] = rank_orig - rank_adv  # positive = lifted up

    out_csv = args.out_dir / f"rank_{args.method_tag}_per_pair.csv"
    df[["example_id", "query_id", "query", "product_id", "product_title",
        "esci_label", "sim_orig", "sim_adv", "delta_held_out",
        "rank_orig", "rank_adv", "rank_lift"]].to_csv(out_csv, index=False, quoting=csv.QUOTE_NONNUMERIC)

    summary = {
        "mode": "rank",
        "method_tag": args.method_tag,
        "n_pairs": int(len(df)),
        "delta_held_out_mean": float(df["delta_held_out"].mean()),
        "delta_held_out_ci95_half": float(1.96 * df["delta_held_out"].std(ddof=1) / np.sqrt(max(1, len(df)))),
        "rank_lift_mean": float(df["rank_lift"].mean()),
        "rank_lift_median": float(df["rank_lift"].median()),
        "i_label_top3_promotion_rate_orig": float(((df["esci_label"] == "I") & (df["rank_orig"] <= 3)).mean()),
        "i_label_top3_promotion_rate_adv": float(((df["esci_label"] == "I") & (df["rank_adv"] <= 3)).mean()),
        "per_label_delta_mean": {lab: float(df.loc[df["esci_label"] == lab, "delta_held_out"].mean())
                                 for lab in LABELS if (df["esci_label"] == lab).any()},
    }
    (args.out_dir / f"rank_{args.method_tag}_summary.json").write_text(json.dumps(summary, indent=2))
    LOG.info("rank summary: %s", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
