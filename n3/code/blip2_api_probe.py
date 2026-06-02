#!/usr/bin/env python3
"""BLIP-2 ITM API probe for R2 second-paradigm closure.

Goal: resolve the exact transformers `Blip2ForImageTextRetrieval` API surface so
the full re-score harness (eval_harness_blip2.py) can be written against verified
shapes — NOT guessed. Q-Former retrieval differs from BLIP-base ITM (32 query
tokens, max-sim ITC), so we confirm:
  1) the model downloads via hf-mirror and loads on the target GPU,
  2) ITM head output: attribute name + tensor shape (expect [B,2] match logits),
  3) ITC head output: attribute name + tensor shape,
  4) whether factorizable image/text features are reachable for a full-pool matrix.

Inference-only, 2 images + 1 query. No data dependency beyond two local jpgs that
the caller passes; falls back to random tensors if none given.
"""
from __future__ import annotations
import argparse, sys, traceback
from pathlib import Path
import torch
from PIL import Image


def describe(name, out):
    print(f"--- {name}: type={type(out).__name__}")
    # dataclass-style ModelOutput: list attributes that are tensors
    keys = []
    if hasattr(out, "keys"):
        try:
            keys = list(out.keys())
        except Exception:
            keys = []
    if not keys:
        keys = [a for a in dir(out) if not a.startswith("_")]
    for k in keys:
        try:
            v = getattr(out, k)
        except Exception:
            continue
        if isinstance(v, torch.Tensor):
            print(f"    .{k}: Tensor shape={tuple(v.shape)} dtype={v.dtype}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Salesforce/blip2-itm-vit-g")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--img", type=Path, nargs="*", default=[])
    p.add_argument("--query", default="red running shoes for men")
    args = p.parse_args(argv)

    from transformers import AutoProcessor, Blip2ForImageTextRetrieval
    import transformers
    print("transformers", transformers.__version__)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print("device", device, "cuda_avail", torch.cuda.is_available())

    proc = AutoProcessor.from_pretrained(args.model)
    model = Blip2ForImageTextRetrieval.from_pretrained(
        args.model, torch_dtype=torch.float16).to(device).eval()
    if getattr(proc, "num_query_tokens", None) is None:
        proc.num_query_tokens = model.config.num_query_tokens
    print("NUM_QUERY_TOKENS", proc.num_query_tokens)
    print("MODEL_LOADED children:", [n for n, _ in model.named_children()])

    if args.img:
        imgs = [Image.open(x).convert("RGB") for x in args.img]
    else:
        imgs = [Image.new("RGB", (224, 224), (200, 100, 50)),
                Image.new("RGB", (224, 224), (50, 100, 200))]

    # ITM head
    enc = proc(images=imgs, text=[args.query] * len(imgs), return_tensors="pt",
               padding=True, truncation=True, max_length=64).to(device)
    with torch.no_grad():
        itm = model(**enc, use_image_text_matching_head=True)
    describe("ITM(use_image_text_matching_head=True)", itm)

    # ITC head
    with torch.no_grad():
        itc = model(**enc, use_image_text_matching_head=False)
    describe("ITC(use_image_text_matching_head=False)", itc)

    # Try to extract a match probability + an ITC similarity scalar robustly
    def itm_prob(o):
        for k in ("logits_per_image", "itm_score", "logits"):
            v = getattr(o, k, None)
            if isinstance(v, torch.Tensor):
                t = v
                if t.dim() == 3:
                    t = t.reshape(t.shape[0], -1)
                if t.shape[-1] == 2:
                    return torch.softmax(t.float(), dim=-1)[..., 1].flatten().tolist()
                return t.float().flatten().tolist()
        return None

    def itc_sim(o):
        for k in ("logits_per_image", "logits_per_text", "image_embeds", "logits"):
            v = getattr(o, k, None)
            if isinstance(v, torch.Tensor):
                return k, tuple(v.shape)
        return None, None

    print("ITM_PROB_SAMPLE", itm_prob(itm))
    print("ITC_SIM_ATTR", itc_sim(itc))
    print("PROBE_OK")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        print("PROBE_FAIL")
        sys.exit(1)
