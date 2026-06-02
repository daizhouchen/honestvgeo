"""dataset_food101_hf — pull `ethz/food101` via huggingface datasets (uses requests,
which works on Python 3.14 unlike urllib).

Materializes 50 images at 800x800 PNG with seed=42 random subsample, plus manifest.json
including the Food-101 class label per image (used as category-name anchor in cogeo.py).
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from PIL import Image


def main():
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--repo-id", default="ethz/food101")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[dataset_food101_hf] loading {args.repo_id} split={args.split}")
    from datasets import load_dataset
    # full (non-streaming) — first time will materialize parquet shards under HF cache
    ds = load_dataset(args.repo_id, split=args.split)
    print(f"[dataset_food101_hf] dataset loaded: len={len(ds)}; features={list(ds.features.keys())}")

    label_feat = ds.features.get("label", None)
    label_names = getattr(label_feat, "names", None)

    rng = random.Random(args.seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    picks = indices[: args.n]

    manifest = {
        "source": f"hf:{args.repo_id}:{args.split}",
        "seed": args.seed,
        "n": args.n,
        "image_size": "800x800",
        "items": []
    }

    for i, idx in enumerate(picks):
        rec = ds[idx]
        img = rec["image"] if "image" in rec else rec.get("img")
        if img is None:
            print(f"[dataset_food101_hf] skip idx={idx}: no image")
            continue
        if not isinstance(img, Image.Image):
            from io import BytesIO
            img = Image.open(BytesIO(img["bytes"])).convert("RGB")
        else:
            img = img.convert("RGB")
        img = img.resize((800, 800), Image.BICUBIC)
        iid = f"food50_{i:03d}"
        out_path = out_dir / f"{iid}.png"
        img.save(out_path, "PNG")

        label_id = rec.get("label", -1)
        category = label_names[label_id] if label_names and 0 <= label_id < len(label_names) else str(label_id)
        # Food-101 categories are e.g. "apple_pie" -> "apple pie"
        category_pretty = str(category).replace("_", " ")

        manifest["items"].append({
            "image_id": iid,
            "category": category_pretty,
            "category_raw": str(category),
            "label_id": int(label_id),
            "source_index": int(idx),
            "out_path": f"{iid}.png",
        })

    with (out_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[dataset_food101_hf] wrote {len(manifest['items'])} -> {out_dir}")


if __name__ == "__main__":
    main()
