#!/usr/bin/env python3
"""Generate per-image VLM captions for the ESCI manifest.

Output: a JSON dict {product_id: caption_text} consumed by
``n3_attack.py --anchor-mode vlm --vlm-captions-json <FILE>``.

Five VLM backends are supported (selected via --vlm):

    qwen25  Qwen/Qwen2.5-VL-7B-Instruct   (Qwen 2.5 family)
    qwen2   Qwen/Qwen2-VL-7B-Instruct     (Qwen 2 family, older arch)
    llava   llava-hf/llava-1.5-7b-hf      (LLaVA-1.5 family)
    blip2   Salesforce/blip2-opt-2.7b     (BLIP-2 + OPT 2.7B family)
    blip    Salesforce/blip-image-captioning-base
                                          (lightweight encoder-decoder)

The captions depend only on x_p (the product image) and never read the
ESCI query, so they preserve the held-out evaluation invariant.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

LOG = logging.getLogger("n3_vlm_caption")


_PROMPT = (
    "You are a professional e-commerce product description writer. Write a concise "
    "70-120 word English description of the product in this image, focused on: "
    "(1) product category and main appearance (color, material, form); "
    "(2) key selling points and quality indicators; "
    "(3) suitable scenarios or target users. "
    "Do not invent details not visible in the image. Output a single paragraph, no bullet points."
)


def _load_qwen(model_id: str, device: str, dtype):
    from transformers import (
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
        Qwen2VLForConditionalGeneration,
    )
    if "2.5-VL" in model_id or "2_5_VL" in model_id.replace("-", "_"):
        cls = Qwen2_5_VLForConditionalGeneration
    else:
        cls = Qwen2VLForConditionalGeneration
    LOG.info("loading %s via %s", model_id, cls.__name__)
    model = cls.from_pretrained(model_id, torch_dtype=dtype, device_map=device).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


@torch.no_grad()
def _caption_qwen(model, processor, image: Image.Image, device: str, max_new_tokens: int) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": _PROMPT},
        ],
    }]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat], images=[image], return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[:, inputs["input_ids"].shape[1]:]
    text = processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
    return text


def _load_blip(model_id: str, device: str, dtype):
    from transformers import BlipForConditionalGeneration, BlipProcessor
    LOG.info("loading %s via Blip*", model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
    processor = BlipProcessor.from_pretrained(model_id)
    return model, processor


@torch.no_grad()
def _caption_blip(model, processor, image: Image.Image, device: str, max_new_tokens: int) -> str:
    inputs = processor(image, "a product photo of", return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=4)
    text = processor.decode(out[0], skip_special_tokens=True).strip()
    return text


def _load_blip2(model_id: str, device: str, dtype):
    from transformers import Blip2ForConditionalGeneration, Blip2Processor
    LOG.info("loading %s via Blip2*", model_id)
    model = Blip2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
    processor = Blip2Processor.from_pretrained(model_id)
    return model, processor


@torch.no_grad()
def _caption_blip2(model, processor, image: Image.Image, device: str, max_new_tokens: int) -> str:
    prompt = "Question: Describe this product in 70-120 words including category, appearance, key selling points, and suitable users. Answer:"
    inputs = processor(image, prompt, return_tensors="pt").to(device, dtype=model.dtype)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=4)
    text = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
    return text


def _load_llava(model_id: str, device: str, dtype):
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    LOG.info("loading %s via Llava*", model_id)
    model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype, device_map=device).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


@torch.no_grad()
def _caption_llava(model, processor, image: Image.Image, device: str, max_new_tokens: int) -> str:
    conv = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": _PROMPT},
        ],
    }]
    prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[:, inputs["input_ids"].shape[1]:]
    text = processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
    return text


_VLM_REGISTRY = {
    "qwen25": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2": "Qwen/Qwen2-VL-7B-Instruct",
    "llava": "llava-hf/llava-1.5-7b-hf",
    "blip2": "Salesforce/blip2-opt-2.7b",
    "blip": "Salesforce/blip-image-captioning-base",
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vlm", choices=tuple(_VLM_REGISTRY.keys()), required=True)
    p.add_argument("--model-id", default=None, help="Override the HF model id (default: registry lookup)")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--img-dir", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    p.add_argument("--max-new-tokens", type=int, default=180)
    p.add_argument("--max-images", type=int, default=None, help="Cap on #images for smoke test")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

    model_id = args.model_id or _VLM_REGISTRY[args.vlm]
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.manifest)
    df = df.copy()
    df["_img_path"] = df["product_id"].apply(lambda a: args.img_dir / f"{a}.jpg")
    have = df["_img_path"].apply(lambda p: p.exists() and p.stat().st_size > 0)
    if not have.all():
        LOG.warning("dropping %d pairs whose original image is missing", int((~have).sum()))
        df = df[have].reset_index(drop=True)
    df = df.drop_duplicates(subset=["product_id"]).reset_index(drop=True)
    if args.max_images is not None:
        df = df.head(args.max_images).reset_index(drop=True)
    LOG.info("captioning %d unique images with %s -> %s", len(df), args.vlm, model_id)

    # Resume from existing JSON if present (skip captions already produced).
    captions: dict[str, str] = {}
    if args.out_json.exists():
        try:
            captions = json.loads(args.out_json.read_text())
            LOG.info("resuming: %d existing captions in %s", len(captions), args.out_json)
        except Exception as exc:
            LOG.warning("could not read existing %s (%s); starting fresh", args.out_json, exc)
            captions = {}

    if args.vlm in ("qwen25", "qwen2"):
        model, processor = _load_qwen(model_id, device, dtype)
        cap_fn = _caption_qwen
    elif args.vlm == "blip":
        model, processor = _load_blip(model_id, device, dtype)
        cap_fn = _caption_blip
    elif args.vlm == "blip2":
        model, processor = _load_blip2(model_id, device, dtype)
        cap_fn = _caption_blip2
    elif args.vlm == "llava":
        model, processor = _load_llava(model_id, device, dtype)
        cap_fn = _caption_llava
    else:
        raise ValueError(args.vlm)

    t0_all = time.time()
    for idx, row in df.iterrows():
        asin = str(row["product_id"])
        if asin in captions and isinstance(captions[asin], str) and captions[asin].strip():
            continue
        try:
            img = Image.open(row["_img_path"]).convert("RGB")
        except Exception as exc:
            LOG.warning("skipping %s: cannot open image (%s)", asin, exc)
            continue
        try:
            text = cap_fn(model, processor, img, device, args.max_new_tokens)
        except Exception as exc:
            LOG.error("caption failed for %s: %s", asin, exc)
            continue
        text = " ".join(text.split())  # collapse whitespace
        captions[asin] = text
        if (idx + 1) % 25 == 0 or (idx + 1) == len(df):
            args.out_json.write_text(json.dumps(captions, indent=2, ensure_ascii=False))
            elapsed = time.time() - t0_all
            LOG.info("  captioned %d/%d in %.1fs (avg %.2fs/img); saved", idx + 1, len(df), elapsed, elapsed / (idx + 1))
    # Final flush.
    args.out_json.write_text(json.dumps(captions, indent=2, ensure_ascii=False))
    LOG.info("done -> %s (n=%d, wall=%.1fs)", args.out_json, len(captions), time.time() - t0_all)
    return 0


if __name__ == "__main__":
    sys.exit(main())
