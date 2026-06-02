#!/usr/bin/env python3
"""Food-101 manifest builder for the cross-dataset (E4) ablation.

Builds the same manifest schema that esci_loader.py emits so eval_harness.py +
n3_attack.py + n3_attack_baselines.py can ingest Food-101 with no code changes:

    columns = [
        "example_id", "query_id", "query",
        "product_id", "product_title", "product_brand",
        "esci_label",
    ]
    image directory layout: <img-dir>/{product_id}.jpg

Cohort assignment (E / S / C / I) on Food-101 is synthetic but deterministic:

  * Every image belongs to one true class c0.
  * We compute a 101x101 CLIP_text(class_name_i) similarity matrix once.
  * For each image, the four cohorts are drawn from the same image's true class
    ordered by class-name CLIP similarity to c0:
        E -> q = class_name(c0)                   (same class, exact match)
        S -> q = class_name(c) for c in rank(1..25)   (substitute, similar)
        C -> q = class_name(c) for c in rank(26..50)  (complement-ish)
        I -> q = class_name(c) for c in rank(76..100) (irrelevant, far class)
  * Concrete choice within each rank-band uses a deterministic round-robin
    seeded by image_id so cohorts have balanced query coverage.

This is honest synthetic ESCI-style cohorting: it does not claim human
relevance graders, it claims a CLIP-text-distance proxy. That distinction is
recorded in the column ``cohort_source`` of the manifest.

Usage::

    python food101_loader.py \
        --root ${PROJECT_ROOT}/data/food101 \
        --out-img-dir ${PROJECT_ROOT}/n3/data/food101_thumbs/img \
        --out-manifest ${PROJECT_ROOT}/n3/data/food101_thumbs/manifest.csv \
        --per-class 5 --image-size 224 \
        --clip-backbone ViT-L-14
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

LOG = logging.getLogger("food101_loader")


def _norm_class_name(c: str) -> str:
    return c.replace("_", " ").strip().lower()


def _format_query(class_name: str) -> str:
    # match the ESCI shopper-style phrasing
    return _norm_class_name(class_name)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True,
                    help="Path to Food-101 root (directory containing images/<class>/<id>.jpg + meta/classes.txt)")
    ap.add_argument("--out-img-dir", type=Path, required=True)
    ap.add_argument("--out-manifest", type=Path, required=True)
    ap.add_argument("--per-class", type=int, default=5,
                    help="Images per Food-101 class to sample (balanced)")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--clip-backbone", default="ViT-L-14",
                    help="OpenCLIP backbone for class-name similarity (uses openai pretrained)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    args.out_img_dir.mkdir(parents=True, exist_ok=True)
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)

    # Discover Food-101 layout
    classes_file = args.root / "meta" / "classes.txt"
    if not classes_file.exists():
        # Fallback: derive from images/ subfolders
        img_root = args.root / "images"
        if not img_root.exists():
            LOG.error("cannot find Food-101 layout under %s (need meta/classes.txt or images/<class>/)", args.root)
            return 2
        classes = sorted([p.name for p in img_root.iterdir() if p.is_dir()])
    else:
        classes = [c.strip() for c in classes_file.read_text().splitlines() if c.strip()]
    LOG.info("found %d Food-101 classes", len(classes))

    img_root = args.root / "images"
    if not img_root.exists():
        LOG.error("expected %s to exist", img_root)
        return 2

    # Sample per_class images per class (balanced)
    samples = []  # list of (class_name, src_path)
    for cls in classes:
        d = img_root / cls
        if not d.exists():
            LOG.warning("missing class dir %s", d)
            continue
        files = sorted(d.glob("*.jpg"))
        if len(files) == 0:
            files = sorted(d.glob("*.jpeg")) + sorted(d.glob("*.png"))
        if len(files) == 0:
            LOG.warning("no images in %s", d)
            continue
        rng.shuffle(files)
        picked = files[:args.per_class]
        for p in picked:
            samples.append((cls, p))
    LOG.info("sampled %d images total (%d per class target)", len(samples), args.per_class)

    # Resize-and-write each image to flat out_img_dir keyed by product_id
    # product_id = food101-{class}-{stem}
    rows_per_image = []
    for cls, src in samples:
        stem = src.stem  # e.g. "1234567"
        product_id = f"food101-{cls}-{stem}"
        out_path = args.out_img_dir / f"{product_id}.jpg"
        if not out_path.exists() or out_path.stat().st_size == 0:
            try:
                im = Image.open(src).convert("RGB")
                im = im.resize((args.image_size, args.image_size), Image.BICUBIC)
                im.save(out_path, quality=95)
            except Exception as exc:
                LOG.warning("failed to write %s: %s", out_path, exc)
                continue
        rows_per_image.append((product_id, cls, out_path))
    LOG.info("wrote %d images to %s", len(rows_per_image), args.out_img_dir)

    # Build class-name CLIP similarity matrix
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    LOG.info("encoding %d class names with %s on %s", len(classes), args.clip_backbone, device)
    import open_clip  # type: ignore

    arch_pretrained = {
        "ViT-L-14": ("ViT-L-14", "openai"),
        "ViT-B-32": ("ViT-B-32", "openai"),
        "openai/clip-vit-large-patch14": ("ViT-L-14", "openai"),
        "openai/clip-vit-base-patch32": ("ViT-B-32", "openai"),
    }[args.clip_backbone]
    model, _, _ = open_clip.create_model_and_transforms(
        arch_pretrained[0], pretrained=arch_pretrained[1], device=str(device))
    tokenizer = open_clip.get_tokenizer(arch_pretrained[0])
    model.eval()

    class_texts = [_format_query(c) for c in classes]
    with torch.no_grad():
        tok = tokenizer(class_texts).to(device)
        feats = model.encode_text(tok)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    sim = (feats @ feats.T).cpu().numpy()  # 101x101 self-similarity
    np.fill_diagonal(sim, -1.0)  # exclude self when ranking neighbors

    # For each class c, rank the 100 OTHER classes by similarity (descending)
    rank_by_class = {}  # class_name -> list of class_names ordered most-similar-first
    for i, c in enumerate(classes):
        order = np.argsort(-sim[i])  # descending
        rank_by_class[c] = [classes[j] for j in order if classes[j] != c][:100]

    # Cohort tier definitions on rank list
    # E = self
    # S = ranks 1..25 (top quarter most similar)
    # C = ranks 26..50 (second quarter)
    # I = ranks 76..100 (least similar quarter)
    def pick_cohort_query(c0: str, cohort: str, image_seed: int) -> str:
        if cohort == "E":
            return _format_query(c0)
        rk = rank_by_class[c0]
        if cohort == "S":
            band = rk[0:25]
        elif cohort == "C":
            band = rk[25:50]
        else:  # I
            band = rk[75:100]
        idx = image_seed % len(band)
        return _format_query(band[idx])

    # Compose per-pair manifest rows: each image gets 4 rows (one per cohort)
    LABELS = ["E", "S", "C", "I"]
    out_rows = []
    seen_query_ids: dict[str, int] = {}
    qid_counter = [0]

    def query_id_for(q: str) -> str:
        if q not in seen_query_ids:
            seen_query_ids[q] = qid_counter[0]
            qid_counter[0] += 1
        return f"food101-q{seen_query_ids[q]:04d}"

    for image_idx, (product_id, cls, src_path) in enumerate(rows_per_image):
        for cohort_idx, lab in enumerate(LABELS):
            q = pick_cohort_query(cls, lab, image_idx + cohort_idx * 7919)
            qid = query_id_for(q)
            example_id = f"{product_id}__{qid}"
            out_rows.append({
                "example_id": example_id,
                "query_id": qid,
                "query": q,
                "product_id": product_id,
                "product_title": _format_query(cls),
                "product_brand": _format_query(cls).split()[0] if _format_query(cls) else cls,
                "esci_label": lab,
                "true_class": cls,
                "cohort_source": "clip_text_quartile",
            })

    df = pd.DataFrame(out_rows)
    df.to_csv(args.out_manifest, index=False, quoting=csv.QUOTE_NONNUMERIC)
    LOG.info("wrote manifest %s (%d rows, %d unique products, %d unique queries)",
             args.out_manifest, len(df), df["product_id"].nunique(), df["query"].nunique())
    LOG.info("per-cohort row counts: %s", df["esci_label"].value_counts().to_dict())

    summary = {
        "n_pairs": int(len(df)),
        "n_unique_products": int(df["product_id"].nunique()),
        "n_unique_queries": int(df["query"].nunique()),
        "n_classes": len(classes),
        "per_class_target": args.per_class,
        "per_cohort_counts": {k: int(v) for k, v in df["esci_label"].value_counts().to_dict().items()},
        "image_size": args.image_size,
        "clip_backbone": args.clip_backbone,
        "cohort_source": "clip_text_quartile",
        "seed": args.seed,
        "manifest": str(args.out_manifest),
        "img_dir": str(args.out_img_dir),
    }
    summary_path = args.out_manifest.parent / "food101_manifest_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    LOG.info("wrote manifest summary %s", summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
