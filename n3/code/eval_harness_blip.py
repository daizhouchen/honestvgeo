#!/usr/bin/env python3
"""R2 cross-retrieval-paradigm eval: re-score CoGEO/baseline adversarial images
with a BLIP fused-encoder retriever, instead of the CLIP dual-encoder used in
eval_harness.py. Mirrors eval_harness.py `rank` mode output schema so the two
paradigms are directly comparable.

Why a different paradigm: CLIP scores image and text in SEPARATE encoders
(cosine of two embeddings). BLIP's ITM head jointly encodes the image+text pair
with cross-attention -> a genuinely different retrieval mechanism. The paper's
own Limitations names fused-encoder ITM as an untested paradigm; this closes it.

Two-stage retrieval (fused encoders do NOT factorize, so a full N*N ITM matrix
is intractable -> standard practice is ITC 1st-stage recall + ITM rerank):
  Stage 1 (ITC, factorizable): BLIP image/text projection cosine -> top-K
           candidates per query over the catalog pool.
  Stage 2 (ITM, cross-attention): rerank the top-K candidates by match logit.

Metrics per (query, product) pair, orig image vs adversarial image:
  itc_delta        = ITC(adv,q)        - ITC(orig,q)
  itm_delta        = ITM_prob(adv,q)   - ITM_prob(orig,q)
  rank_lift_itc    = full-pool rank lift using ITC only (BLIP 1st stage)
  rank_lift_rerank = full-pool rank lift using two-stage ITC->ITM ordering
                     (THE fused-encoder paradigm number)
A positive rank_lift_rerank means the attack still promotes the item under the
fused-encoder reranker; ~0 / negative means the threat does NOT transfer to this
paradigm (a recalibration finding either way).

Usage:
  python eval_harness_blip.py --manifest n3/data/esci1500_thumbs/manifest.csv \
      --img-dir n3/data/esci1500_thumbs/img \
      --adv-dir n3/runs/B2_full4method/eps16_1430/cogeo/img \
      --out-dir n3/runs/R2_blip_rerank/cogeo --method-tag cogeo \
      --blip-model Salesforce/blip-itm-base-coco --topk 50 [--limit 64]
"""
from __future__ import annotations
import argparse, csv, json, logging, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

LOG = logging.getLogger("eval_harness_blip")
LABELS = ("E", "S", "C", "I")


def load_blip(model_id, device):
    from transformers import BlipProcessor, BlipForImageTextRetrieval
    proc = BlipProcessor.from_pretrained(model_id)
    model = BlipForImageTextRetrieval.from_pretrained(model_id).to(device).eval()
    children = [n for n, _ in model.named_children()]
    LOG.info("BLIP children: %s", children)
    for need in ("vision_model", "text_encoder", "vision_proj", "text_proj", "itm_head"):
        if not hasattr(model, need):
            LOG.warning("model missing attribute %s (API drift?)", need)
    return proc, model


@torch.no_grad()
def itc_image_feats(proc, model, pil_images, device, batch=32):
    feats = []
    for i in range(0, len(pil_images), batch):
        chunk = pil_images[i:i + batch]
        px = proc(images=chunk, return_tensors="pt").pixel_values.to(device)
        vis = model.vision_model(px)[0]                 # [B, seq, H]
        f = F.normalize(model.vision_proj(vis[:, 0, :]), dim=-1)
        feats.append(f.cpu())
    return torch.cat(feats, 0)


@torch.no_grad()
def itc_text_feats(proc, model, texts, device, batch=64):
    feats = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        tok = proc(text=chunk, padding=True, truncation=True, max_length=40, return_tensors="pt").to(device)
        txt = model.text_encoder(input_ids=tok.input_ids, attention_mask=tok.attention_mask, return_dict=True)[0]
        f = F.normalize(model.text_proj(txt[:, 0, :]), dim=-1)
        feats.append(f.cpu())
    return torch.cat(feats, 0)


@torch.no_grad()
def itm_match_probs(proc, model, pil_images, query, device, batch=16):
    """ITM match prob (softmax[:,1]) for a list of images all paired with ONE query."""
    out = []
    for i in range(0, len(pil_images), batch):
        chunk = pil_images[i:i + batch]
        enc = proc(images=chunk, text=[query] * len(chunk), padding=True, truncation=True,
                   max_length=40, return_tensors="pt").to(device)
        res = model(**enc, use_itm_head=True)
        logits = res[0] if isinstance(res, (tuple, list)) else getattr(res, "itm_score", res)
        out.append(torch.softmax(logits, dim=1)[:, 1].cpu())
    return torch.cat(out, 0).numpy()


def rank_in_pool_itc(pool_scores_q, idx):
    """1-indexed rank of image `idx` for a query given full-pool ITC scores (desc)."""
    s = pool_scores_q[idx]
    return int((pool_scores_q >= s).sum())


def two_stage_order_positions(pool_itc_q, topk_idx, topk_itm):
    """Return a dict idx->1-indexed position under two-stage ordering:
    top-K reranked by ITM (desc) occupy ranks 1..K; the rest by ITC (desc) after."""
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
    p.add_argument("--blip-model", default="Salesforce/blip-itm-base-coco")
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--limit", type=int, default=0, help="self-test: cap manifest rows")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

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

    proc, model = load_blip(args.blip_model, device)

    asins = df["product_id"].drop_duplicates().tolist()
    a2i = {a: i for i, a in enumerate(asins)}
    queries = df["query"].drop_duplicates().tolist()
    q2i = {q: i for i, q in enumerate(queries)}
    # NB: use bracket/list access, never df.query (that is the DataFrame.query method)
    PID = df["product_id"].tolist(); QRY = df["query"].tolist()
    EXID = df["example_id"].tolist(); QID = df["query_id"].tolist(); LAB = df["esci_label"].tolist()
    orig_imgs = [Image.open(args.img_dir / f"{a}.jpg").convert("RGB") for a in asins]
    adv_imgs = [Image.open(args.adv_dir / f"{a}.jpg").convert("RGB") for a in asins]

    t0 = time.time()
    img_orig = itc_image_feats(proc, model, orig_imgs, device, args.batch)
    img_adv = itc_image_feats(proc, model, adv_imgs, device, args.batch)
    txt = itc_text_feats(proc, model, queries, device, args.batch * 2)
    LOG.info("ITC feats done %.1fs N_img=%d N_q=%d", time.time() - t0, len(asins), len(queries))

    itc_pool = (img_orig @ txt.T).numpy()          # [N, Q] orig pool ITC
    itc_adv_diag = np.array([(img_adv[a2i[PID[r]]] * txt[q2i[QRY[r]]]).sum().item()
                             for r in range(len(df))])
    itc_orig_diag = np.array([itc_pool[a2i[PID[r]], q2i[QRY[r]]] for r in range(len(df))])

    K = min(args.topk, len(asins))
    # cache per-query orig two-stage ordering + topk ITM
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
        # adv: only image i changes
        adv_scores = scores.copy(); adv_scores[i] = itc_adv_diag[r]
        rank_adv_itc = int((adv_scores >= adv_scores[i]).sum())
        adv_topk_idx = list(np.argsort(-adv_scores)[:K])
        itm_map = dict(c["topk_itm"])
        if i in adv_topk_idx:
            itm_map[i] = itm_match_probs(proc, model, [adv_imgs[i]], q_text, device)[0]
        # any newly-entered topk member other than i keeps its orig ITM if known, else compute
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
    out_csv = args.out_dir / f"blip_rank_{args.method_tag}_per_pair.csv"
    out.to_csv(out_csv, index=False, quoting=csv.QUOTE_NONNUMERIC)

    def per_label(col):
        return {lab: float(out.loc[out.esci_label == lab, col].mean())
                for lab in LABELS if (out.esci_label == lab).any()}
    summary = dict(
        paradigm="blip_fused_encoder_itm_rerank", blip_model=args.blip_model, topk=K,
        method_tag=args.method_tag, n_pairs=int(len(out)),
        itc_delta_mean=float(out.itc_delta.mean()),
        itm_delta_mean=float(out.itm_delta.mean()),
        rank_lift_itc_mean=float(out.rank_lift_itc.mean()),
        rank_lift_rerank_mean=float(out.rank_lift_rerank.mean()),
        rank_lift_rerank_median=float(out.rank_lift_rerank.median()),
        per_label_rank_lift_rerank=per_label("rank_lift_rerank"),
        per_label_itm_delta=per_label("itm_delta"),
    )
    (args.out_dir / f"blip_rank_{args.method_tag}_summary.json").write_text(json.dumps(summary, indent=2))
    LOG.info("SUMMARY %s", json.dumps(summary, indent=2))
    print("BLIP_RANK_OK", args.method_tag)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback; traceback.print_exc(); print("BLIP_RANK_FAIL"); sys.exit(1)
