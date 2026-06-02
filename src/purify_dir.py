#!/usr/bin/env python3
"""R4 input-purification defense: apply a benign image transform to a directory
of (adversarial or clean) product images, writing transformed pixels so the
EXISTING eval_harness.py can re-score rank-lift on the purified images.

Transforms are deliberately chosen OUTSIDE the attack's EnvSim EOT envelope
(EnvSim hardened the attack against DIM resize +/-10%, Gaussian blur k7,
DiffJPEG q85 -- see cogeo_src/envsim.py). So the defense is NOT circular:
  - bitdepth N : feature squeezing to N bits/channel        (NOT in EOT)
  - resize F   : downsample to F*size then back (F<=0.5)     (far beyond +/-10% EOT)
  - jpeg Q     : JPEG at quality Q (use Q<=50, below q85 EOT)
  - median K   : KxK median filter                            (EOT only had Gaussian)
  - none       : passthrough (undefended baseline)

Output is saved as PNG-encoded bytes under the original "{id}.jpg" filename so
the ONLY lossy step is the intended transform (eval_harness opens by content,
not by extension).
"""
import argparse, io
from pathlib import Path
from PIL import Image, ImageFilter
import numpy as np


def t_jpeg(im, q):
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="JPEG", quality=int(q))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def t_resize(im, f):
    w, h = im.size
    nw, nh = max(8, int(round(w * float(f)))), max(8, int(round(h * float(f))))
    small = im.resize((nw, nh), Image.BICUBIC)
    return small.resize((w, h), Image.BICUBIC)


def t_bitdepth(im, bits):
    a = np.asarray(im.convert("RGB"), dtype=np.float64)
    levels = (1 << int(bits)) - 1
    a = np.round(a / 255.0 * levels) / levels * 255.0
    return Image.fromarray(a.clip(0, 255).astype(np.uint8), "RGB")


def t_median(im, k):
    return im.convert("RGB").filter(ImageFilter.MedianFilter(size=int(k)))


def apply_defense(im, mode, param):
    if mode == "jpeg":
        return t_jpeg(im, param)
    if mode == "resize":
        return t_resize(im, param)
    if mode == "bitdepth":
        return t_bitdepth(im, param)
    if mode == "median":
        return t_median(im, param)
    if mode == "none":
        return im.convert("RGB")
    raise ValueError(f"unknown mode {mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mode", required=True,
                    choices=["jpeg", "resize", "bitdepth", "median", "none"])
    ap.add_argument("--param", required=True)
    a = ap.parse_args()
    src, out = Path(a.src_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted([p for p in src.iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    n = 0
    for p in files:
        try:
            im = Image.open(p).convert("RGB")
        except Exception as e:
            print("skip", p.name, e)
            continue
        pim = apply_defense(im, a.mode, a.param)
        # save transformed pixels losslessly (PNG content) under the SAME filename
        pim.save(out / p.name, format="PNG")
        n += 1
    print(f"purified {n} images: mode={a.mode} param={a.param} {src} -> {out}")


if __name__ == "__main__":
    main()
