"""dataset_sop — pull a 50-image random subsample of Stanford Online Products and
materialize as 800x800 PNG files under a target directory.

Strategy:
  1. Try HuggingFace datasets `Multimodal-Fatima/StanfordOnlineProducts` via hf-mirror.
  2. Fall back to direct Stanford FTP if HF version is unavailable.

This script writes:
  <out_dir>/<id>.png         (resized 800x800)
  <out_dir>/manifest.json    (image_id -> source_path/category/seed/N)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import urllib.request
from pathlib import Path

from PIL import Image


def _try_hf(out_dir: Path, n: int, seed: int) -> bool:
    try:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from datasets import load_dataset
        # Load the streaming variant first to avoid massive download if we only want N items
        ds = load_dataset("Multimodal-Fatima/StanfordOnlineProducts", split="train", streaming=True)
    except Exception as e:
        print(f"[dataset_sop] HF load failed: {type(e).__name__}: {e}")
        return False

    # Materialize: collect first ~5000 records, sample N from them deterministically.
    bucket = []
    for i, ex in enumerate(ds):
        if i >= 5000:
            break
        bucket.append(ex)

    rng = random.Random(seed)
    rng.shuffle(bucket)
    picks = bucket[:n]
    if len(picks) < n:
        print(f"[dataset_sop] HF only yielded {len(picks)} samples in first 5000; falling back")
        return False

    manifest = {"source": "hf:Multimodal-Fatima/StanfordOnlineProducts:train", "seed": seed, "n": n, "items": []}
    for i, ex in enumerate(picks):
        img = ex.get("image")
        if img is None:
            continue
        if not hasattr(img, "convert"):
            # LazyLoadedImage from datasets — convert
            from io import BytesIO
            img = Image.open(BytesIO(img["bytes"])).convert("RGB")
        else:
            img = img.convert("RGB")
        img = img.resize((800, 800), Image.BICUBIC)
        iid = f"sop50_{i:03d}"
        out = out_dir / f"{iid}.png"
        img.save(out, "PNG")
        manifest["items"].append({
            "image_id": iid,
            "category": str(ex.get("label", "")),
            "out_path": str(out.relative_to(out_dir)),
        })
    with (out_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[dataset_sop] HF path: wrote {len(manifest['items'])} -> {out_dir}")
    return True


def _try_ftp(out_dir: Path, n: int, seed: int) -> bool:
    """Stanford FTP fallback. Pulls the full Stanford_Online_Products.zip (~2.9 GB),
    samples N file paths, copies them as 800x800 PNGs. Heavy fallback; only triggered
    if HF path failed."""
    url = "ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip"
    cache = out_dir.parent / "_cache_sop_zip"
    cache.mkdir(parents=True, exist_ok=True)
    zpath = cache / "Stanford_Online_Products.zip"
    if not zpath.exists():
        print(f"[dataset_sop] downloading {url} -> {zpath} (~2.9 GB)")
        urllib.request.urlretrieve(url, zpath)
    import zipfile
    with zipfile.ZipFile(zpath) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".jpg")]
        rng = random.Random(seed)
        rng.shuffle(members)
        picks = members[:n]
        manifest = {"source": "ftp:Stanford_Online_Products.zip", "seed": seed, "n": n, "items": []}
        for i, m in enumerate(picks):
            with zf.open(m) as fh:
                img = Image.open(fh).convert("RGB").resize((800, 800), Image.BICUBIC)
            iid = f"sop50_{i:03d}"
            img.save(out_dir / f"{iid}.png", "PNG")
            manifest["items"].append({
                "image_id": iid,
                "category": str(Path(m).parts[-2]) if len(Path(m).parts) > 1 else "",
                "out_path": f"{iid}.png",
                "source_path_in_zip": m,
            })
        with (out_dir / "manifest.json").open("w") as f:
            json.dump(manifest, f, indent=2)
    print(f"[dataset_sop] FTP path: wrote {len(manifest['items'])} -> {out_dir}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="dest dir, will be created")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prefer", choices=["hf", "ftp"], default="hf")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.prefer == "hf":
        ok = _try_hf(out_dir, args.n, args.seed)
        if not ok:
            ok = _try_ftp(out_dir, args.n, args.seed)
    else:
        ok = _try_ftp(out_dir, args.n, args.seed)
        if not ok:
            ok = _try_hf(out_dir, args.n, args.seed)

    if not ok:
        print("[dataset_sop] BOTH HF and FTP paths failed — please inspect.")
        sys.exit(2)


if __name__ == "__main__":
    main()
