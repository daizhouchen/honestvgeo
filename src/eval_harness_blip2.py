#!/usr/bin/env python3
"""R2 SECOND cross-retrieval-paradigm eval: re-score CoGEO/baseline adversarial
images with a BLIP-2 (Q-Former) ITM retriever. Mirrors eval_harness_blip.py and
eval_harness.py `rank` output schema so all paradigms are directly comparable.

Why BLIP-2 is a genuinely distinct third paradigm:
  * CLIP (eval_harness.py)      : dual-encoder, cosine of two pooled embeddings.
  * BLIP (eval_harness_blip.py) : unified encoder, ITM cross-attention rerank.
  * BLIP-2 (this file)          : 32 learned Q-Former query tokens cross-attend to
    a FROZEN ViT-g; ITC = max-sim over query tokens vs text CLS proj; ITM = a
    dedicated image-grounded-text match head. Different fusion architecture again.

Two-stage retrieval (same protocol as the BLIP harness):
  Stage 1 (ITC, factorizable via max-sim): top-K candidates per query over pool.
  Stage 2 (ITM, cross-attention): rerank top-K by match logit.

Key metric: rank_lift_rerank (full-pool, two-stage), per-cohort E/S/C/I.

Usage:
  python eval_harness_blip2.py --manifest data/esci1500_thumbs/manifest.csv \
      --img-dir data/esci1500_thumbs/img \
      --adv-dir runs/T3_scaleup/run4/cogeo/img \
      --out-dir runs/R2_blip2_rerank/cogeo --method-tag cogeo \
      --model Salesforce/blip2-itm-vit-g --topk 50 [--limit 32]
"""
from __future__ import annotations
import argparse, csv, json, logging, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

LOG = logging.getLogger("eval_harness_blip2")
LABELS = ("E", "S", "C", "I")
_DUMMY_IMG = Image.new("RGB", (224, 224), (127, 127, 127))


def load_blip2(model_id, device, dtype):
    from transformers import AutoProcessor, Blip2ForImageTextRetrieval
    proc = AutoProcessor.from_pretrained(model_id)
    model = Blip2ForImageTextRetrieval.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
    # NB: Blip2ForImageTextRetrieval encodes text in the Q-Former WITHOUT image
    # placeholder tokens. Do NOT set proc.num_query_tokens and do NOT pass
    # max_length to the processor: either would inject 32 image-placeholder ids
    # into input_ids and overflow the text-embedding gather (device-side assert).
    LOG.info("BLIP2 children: %s", [n for n, _ in model.named_children()])
    return proc, model


def _img_embeds(out):
    for k in ("image_embeds", "image_embeds_proj", "query_embeds"):
        v = getattr(out, k, None)
        if isinstance(v, torch.Tensor):
            return v
    raise RuntimeError("no image_embeds attr on ITC output")


def _txt_embeds(out):
    for k in ("text_embeds", "text_embeds_proj"):
        v = getattr(out, k, None)
        if isinstance(v, torch.Tensor):
            return v
    raise RuntimeError("no text_embeds attr on ITC output")


@torch.no_grad()
def itc_image_embeds(proc, model, pil_images, device, batch=16):
    """[N, Q, D] L2-normalized image query embeds (text-independent)."""
    feats = []
    for i in range(0, len(pil_images), batch):
        chunk = pil_images[i:i + batch]
        enc = proc(images=chunk, text=["a photo"] * len(chunk), return_tensors="pt",
                   padding=True, truncation=True).to(device)
        ie = _img_embeds(model(**enc, use_image_text_matching_head=False))
        feats.append(F.normalize(ie.float(), dim=-1).cpu())
    return torch.cat(feats, 0)


@torch.no_grad()
def itc_text_embeds(proc, model, texts, device, batch=32):
    """[M, D] L2-normalized text embeds."""
    feats = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        enc = proc(images=[_DUMMY_IMG] * len(chunk), text=chunk, return_tensors="pt",
                   padding=True, truncation=True).to(device)
        te = _txt_embeds(model(**enc, use_image_text_matching_head=False))
        feats.append(F.normalize(te.float(), dim=-1).cpu())
    return torch.cat(feats, 0)


def maxsim_pool(img_qe, txt, chunk=64):
    """img_qe [N,Q,D], txt [M,D] -> [N,M] max-sim over query tokens."""
    N, M = img_qe.shape[0], txt.shape[0]
    out = np.empty((N, M), dtype=np.float32)
    for i in range(0, N, chunk):
        ch = img_qe[i:i + chunk]
        sim = torch.einsum("bqd,md->bqm", ch, txt)
        out[i:i + ch.shape[0]] = sim.max(dim=1).values.numpy()
    return out


def maxsim_diag(img_qe_vec, txt_vec):
    """img_qe_vec [Q,D], txt_vec [D] -> scalar max-sim."""
    return float((img_qe_vec @ txt_vec).max().item())


@torch.no_grad()
def itm_match_probs(proc, model, pil_images, query, device, batch=8):
    """ITM match prob for a list of images all paired with ONE query."""
    out = []
    for i in range(0, len(pil_images), batch):
        chunk = pil_images[i:i + batch]
        enc = proc(images=chunk, text=[query] * len(chunk), return_tensors="pt",
                   padding=True, truncation=True).to(device)
        res = model(**enc, use_image_text_matching_head=True)
        logits = getattr(res, "logits_per_image", None)
        if logits is None:
            logits = res[0] if isinstance(res, (tuple, list)) else getattr(res, "logits", res)
        t = logits.float()
        if t.dim() == 3:
            t = t.reshape(t.shape[0], -1)
        p = torch.softmax(t, dim=-1)[:, 1] if t.shape[-1] == 2 else t.flatten()
        out.append(p.cpu())
    return torch.cat(out, 0).numpy()


def rank_in_pool_itc(pool_scores_q, idx):
    s = pool_scores_q[idx]
    return int((pool_scores_q >= s).sum())


def two_stage_order_positions(pool_itc_q, topk_idx, topk_itm):
    order = list(np.array(topk_idx)[np.argsort(-np.asarray(topk_itm))])
    rest = [j for j in np.argsort(-pool_itc_q) if j not in set(topk_idx)]
    full = order + rest
    return {idx: p + 1 for p, idx in enumerate(full)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--img-dir", type=Path, required=True)
    p.add_argument("--adv-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--method-tag", default="adv")
    p.add_argument("--model", default="Salesforce/blip2-itm-vit-g")
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if args.fp32 else torch.float16

    df = pd.read_csv(args.manifest)
    df["_orig"] = df["product_id"].apply(lambda a: args.img_dir / f"{a}.jpg")
    df["_adv"] = df["product_id"].apply(lambda a: args.adv_dir / f"{a}.jpg")
    keep = df["_orig"].apply(lambda q: q.exists() and q.stat().st_size > 0) & \
           df["_adv"].apply(lambda q: q.exists() and q.stat().st_size > 0)
    df = df[keep].reset_index(drop=True)
    if args.limit and len(df) > args.limit:
        df = df.iloc[:args.limit].reset_index(drop=True)
    if len(df) == 0:
        LOG.error("no usable pairs"); return 2
    LOG.info("usable pairs=%d labels=%s", len(df), df["esci_label"].value_counts().to_dict())

    proc, model = load_blip2(args.model, device, dtype)

    asins = df["product_id"].drop_duplicates().tolist()
    a2i = {a: i for i, a in enumerate(asins)}
    queries = df["query"].drop_duplicates().tolist()
    q2i = {q: i for i, q in enumerate(queries)}
    PID = df["product_id"].tolist(); QRY = df["query"].tolist()
    EXID = df["example_id"].tolist(); QID = df["query_id"].tolist(); LAB = df["esci_label"].tolist()
    orig_imgs = [Image.open(args.img_dir / f"{a}.jpg").convert("RGB") for a in asins]
    adv_imgs = [Image.open(args.adv_dir / f"{a}.jpg").convert("RGB") for a in asins]

    t0 = time.time()
    qe_orig = itc_image_embeds(proc, model, orig_imgs, device, args.batch)   # [N,Q,D]
    qe_adv = itc_image_embeds(proc, model, adv_imgs, device, args.batch)
    txt = itc_text_embeds(proc, model, queries, device, args.batch * 2)      # [M,D]
    LOG.info("ITC embeds done %.1fs N_img=%d N_q=%d qe_shape=%s txt_shape=%s",
             time.time() - t0, len(asins), len(queries), tuple(qe_orig.shape), tuple(txt.shape))

    itc_pool = maxsim_pool(qe_orig, txt)                                     # [N, M]
    itc_adv_diag = np.array([maxsim_diag(qe_adv[a2i[PID[r]]], txt[q2i[QRY[r]]]) for r in range(len(df))])
    itc_orig_diag = np.array([itc_pool[a2i[PID[r]], q2i[QRY[r]]] for r in range(len(df))])

    K = min(args.topk, len(asins))
    q_cache = {}
    for q, qi in q2i.items():
        scores = itc_pool[:, qi]
        topk_idx = list(np.argsort(-scores)[:K])
        topk_imgs = [orig_imgs[j] for j in topk_idx]
        topk_itm = itm_match_probs(proc, model, topk_imgs, q, device)
        pos = two_stage_order_positions(scores, topk_idx, list(topk_itm))
        q_cache[qi] = {"topk_idx": topk_idx, "topk_itm": dict(zip(topk_idx, topk_itm)), "pos": pos}
    LOG.info("per-query orig rerank cached for %d queries", len(q_cache))

    rows = []
    for r in range(len(df)):
        i = a2i[PID[r]]; qi = q2i[QRY[r]]; q_text = QRY[r]
        c = q_cache[qi]; scores = itc_pool[:, qi]
        rank_orig_itc = rank_in_pool_itc(scores, i)
        rank_orig_rr = c["pos"][i]
        adv_scores = scores.copy(); adv_scores[i] = itc_adv_diag[r]
        rank_adv_itc = int((adv_scores >= adv_scores[i]).sum())
        adv_topk_idx = list(np.argsort(-adv_scores)[:K])
        itm_map = dict(c["topk_itm"])
        if i in adv_topk_idx:
            itm_map[i] = itm_match_probs(proc, model, [adv_imgs[i]], q_text, device)[0]
        for j in adv_topk_idx:
            if j not in itm_map:
                itm_map[j] = itm_match_probs(proc, model, [orig_imgs[j]], q_text, device)[0]
        adv_pos = two_stage_order_positions(adv_scores, adv_topk_idx, [itm_map[j] for j in adv_topk_idx])
        rank_adv_rr = adv_pos[i]
        itm_orig_i = float(c["topk_itm"].get(i, itm_match_probs(proc, model, [orig_imgs[i]], q_text, device)[0]))
        itm_adv_i = float(itm_map.get(i, itm_match_probs(proc, model, [adv_imgs[i]], q_text, device)[0]))
        rows.append(dict(
            example_id=EXID[r], query_id=QID[r], query=q_text,
            product_id=PID[r], esci_label=LAB[r],
            itc_orig=float(itc_orig_diag[r]), itc_adv=float(itc_adv_diag[r]),
            itc_delta=float(itc_adv_diag[r] - itc_orig_diag[r]),
            itm_orig=itm_orig_i, itm_adv=itm_adv_i, itm_delta=itm_adv_i - itm_orig_i,
            rank_orig_itc=rank_orig_itc, rank_adv_itc=rank_adv_itc,
            rank_lift_itc=rank_orig_itc - rank_adv_itc,
            rank_orig_rerank=rank_orig_rr, rank_adv_rerank=rank_adv_rr,
            rank_lift_rerank=rank_orig_rr - rank_adv_rr,
        ))
    out = pd.DataFrame(rows)
    out_csv = args.out_dir / f"blip2_rank_{args.method_tag}_per_pair.csv"
    out.to_csv(out_csv, index=False, quoting=csv.QUOTE_NONNUMERIC)

    def per_label(col):
        return {lab: float(out.loc[out.esci_label == lab, col].mean())
                for lab in LABELS if (out.esci_label == lab).any()}
    summary = dict(
        paradigm="blip2_qformer_itm_rerank", model=args.model, topk=K,
        method_tag=args.method_tag, n_pairs=int(len(out)),
        itc_delta_mean=float(out.itc_delta.mean()),
        itm_delta_mean=float(out.itm_delta.mean()),
        rank_lift_itc_mean=float(out.rank_lift_itc.mean()),
        rank_lift_rerank_mean=float(out.rank_lift_rerank.mean()),
        rank_lift_rerank_median=float(out.rank_lift_rerank.median()),
        per_label_rank_lift_rerank=per_label("rank_lift_rerank"),
        per_label_itm_delta=per_label("itm_delta"),
    )
    (args.out_dir / f"blip2_rank_{args.method_tag}_summary.json").write_text(json.dumps(summary, indent=2))
    LOG.info("SUMMARY %s", json.dumps(summary, indent=2))
    print("BLIP2_RANK_OK", args.method_tag)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback; traceback.print_exc(); print("BLIP2_RANK_FAIL"); sys.exit(1)
