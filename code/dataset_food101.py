"""dataset_food101 — pull Food-101 via torchvision and materialize a 50-image
random subsample (seed=42) as 800x800 PNG files under <out>.

torchvision.datasets.Food101 downloads from data.vision.ee.ethz.ch (~5 GB tarball).
We sample BEFORE saving so we don't process all 101k images.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torchvision  # noqa: F401
from torchvision.datasets import Food101
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="dest dir")
    ap.add_argument("--cache-root", default="${PROJECT_ROOT}/data/_cache_food101")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="test", choices=["train", "test"])
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Force CWD-relative for torchvision so it stores under cache-root
    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    print(f"[dataset_food101] downloading Food-101 ({args.split} split) to {cache_root}")
    ds = Food101(root=str(cache_root), split=args.split, download=True)
    print(f"[dataset_food101] dataset loaded: len={len(ds)} | classes={len(ds.classes)}")

    # Sample N indices with seed
    rng = random.Random(args.seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    picks = indices[: args.n]

    manifest = {
        "source": f"torchvision:Food101:{args.split}",
        "seed": args.seed,
        "n": args.n,
        "image_size": "800x800",
        "items": []
    }

    for i, idx in enumerate(picks):
        img, label = ds[idx]   # PIL.Image, int
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.convert("RGB").resize((800, 800), Image.BICUBIC)
        iid = f"food50_{i:03d}"
        out_path = out_dir / f"{iid}.png"
        img.save(out_path, "PNG")
        manifest["items"].append({
            "image_id": iid,
            "category": ds.classes[label],
            "source_index": int(idx),
            "out_path": f"{iid}.png",
        })

    with (out_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[dataset_food101] wrote {len(manifest['items'])} -> {out_dir}")


if __name__ == "__main__":
    main()
