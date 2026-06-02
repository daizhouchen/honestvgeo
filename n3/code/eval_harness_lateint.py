#!/usr/bin/env python3
"""L2 cross-retrieval-paradigm eval: re-score CoGEO/baseline adversarial images
with a ColBERT/ColPali-style LATE-INTERACTION retriever, instead of the CLIP
dual-encoder global-cosine used in eval_harness.py and the BLIP fused-encoder
used in eval_harness_blip.py. Mirrors those harnesses' `rank` output schema so
all three paradigms are directly comparable.

Why a different paradigm: CLIP scores by cosine of two GLOBALLY POOLED embeddings;
BLIP fuses the pair with cross-attention. A late-interaction retriever (ColBERT /
ColPali / FILIP) instead keeps per-token (per-patch / per-word) embeddings and
scores by MaxSim:  score(img, q) = sum_{word t} max_{patch p} <e_t, e_p>.
The paper's own Limitations names late-interaction retrieval as an untested
paradigm; this closes it. We build a TRAINING-FREE late-interaction reranker on a
HELD-OUT CLIP encoder (not the white-box attack source), so the test doubles as a
transfer-axis probe: same held-out encoder, but token-level MaxSim instead of the
global cosine reported in the main transfer table.

Single-stage (MaxSim is computable for the whole pool, unlike the intractable
N*N ITM matrix), so the headline metric is one number:
  rank_lift_lateint = rank_orig_lateint - rank_adv_lateint   (full pool, 1-indexed)
A positive mean rank_lift_lateint => the attack still promotes the item under a
late-interaction retriever (threat transfers to this paradigm); ~0 / negative =>
it does not (a recalibration finding either way). Per-cohort E/S/C/I means test
whether the cohort layering survives the paradigm change.

Usage:
  python eval_harness_lateint.py --manifest n3/data/esci1500_thumbs/manifest.csv \
      --img-dir n3/data/esci1500_thumbs/img \
      --adv-dir n3/runs/T3_scaleup/run4/cogeo/img \
      --out-dir n3/runs/L2_lateint_rerank/cogeo --method-tag cogeo \
      --clip-model ViT-B-32 --pretrained laion2b_s34b_b79k [--limit 64]
"""
from __future__ import annotations
import argparse, csv, json, logging, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

LOG = logging.getLogger("eval_harness_lateint")
LABELS = ("E", "S", "C", "I")


def load_clip(name, pretrained, device):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained)
    model = model.to(device).eval()
    model.visual.output_tokens = True
    tok = open_clip.get_tokenizer(name)
    assert getattr(model.visual, "proj", None) is not None, "visual.proj required to project patch tokens"
    return model, preprocess, tok


@torch.no_grad()
def encode_image(model, preprocess, pil_images, device, batch=64):
    """Returns (pooled [N,D], per-patch tokens [N,P,D]) both L2-normalized (CPU).
    pooled = the standard CLIP global image embedding (CLS @ proj); tokens = the
    per-patch embeddings used by the late-interaction MaxSim scorer."""
    g, t = [], []
    for i in range(0, len(pil_images), batch):
        chunk = pil_images[i:i + batch]
        px = torch.stack([preprocess(im) for im in chunk]).to(device)
        res = model.visual(px)
        pooled = res[0] if isinstance(res, (tuple, list)) else res   # [B, D] (already projected)
        tokens = res[1] if isinstance(res, (tuple, list)) else None  # [B, P, width]
        tokens = tokens @ model.visual.proj                          # [B, P, D]
        g.append(F.normalize(pooled, dim=-1).float().cpu())
        t.append(F.normalize(tokens, dim=-1).float().cpu())
    return torch.cat(g, 0), torch.cat(t, 0)


@torch.no_grad()
def encode_text(model, tok, texts, device, batch=128):
    """Returns (pooled [Q,D] normalized, list of per-word tokens [Lq,D] normalized) CPU.
    pooled = the standard CLIP global text embedding (EOT @ proj)."""
    g, feats = [], []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        ids = tok(chunk).to(device)
        x = model.token_embedding(ids)
        x = x + model.positional_embedding
        x = model.transformer(x, attn_mask=model.attn_mask)   # open_clip 3.x batch_first
        x = model.ln_final(x)
        x = x @ model.text_projection                         # [B, L, D]
        eot = ids.argmax(dim=-1)                              # EOT position
        for b in range(len(chunk)):
            L = int(eot[b]) + 1                               # include SOT..EOT
            g.append(F.normalize(x[b, int(eot[b])], dim=-1).float().cpu())   # global = EOT token
            feats.append(F.normalize(x[b, :L], dim=-1).float().cpu())        # late-interaction tokens
    return torch.stack(g, 0), feats


def maxsim_scores(img_tok, q_tok, chunk=512):
    """img_tok [N,P,D], q_tok [L,D] -> [N] late-interaction MaxSim (sum_l max_p <t_l,p>)."""
    N = img_tok.shape[0]
    out = torch.empty(N)
    for s in range(0, N, chunk):
        block = img_tok[s:s + chunk]                          # [b,P,D]
        sim = torch.einsum("ld,npd->nlp", q_tok, block)       # [b,L,P]
        out[s:s + chunk] = sim.max(dim=-1).values.sum(dim=-1) # [b]
    return out.numpy()


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--img-dir", type=Path, required=True)
    p.add_argument("--adv-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--method-tag", default="adv")
    p.add_argument("--clip-model", default="ViT-B-32")
    p.add_argument("--pretrained", default="laion2b_s34b_b79k")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--limit", type=int, default=0, help="self-test: cap manifest rows")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    LOG.info("device=%s model=%s/%s", device, args.clip_model, args.pretrained)

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

    model, preprocess, tok = load_clip(args.clip_model, args.pretrained, device)

    asins = df["product_id"].drop_duplicates().tolist()
    a2i = {a: i for i, a in enumerate(asins)}
    queries = df["query"].drop_duplicates().tolist()
    q2i = {q: i for i, q in enumerate(queries)}
    PID = df["product_id"].tolist(); QRY = df["query"].tolist()
    EXID = df["example_id"].tolist(); QID = df["query_id"].tolist(); LAB = df["esci_label"].tolist()
    orig_imgs = [Image.open(args.img_dir / f"{a}.jpg").convert("RGB") for a in asins]
    adv_imgs = [Image.open(args.adv_dir / f"{a}.jpg").convert("RGB") for a in asins]

    t0 = time.time()
    g_orig, tok_orig = encode_image(model, preprocess, orig_imgs, device, args.batch)   # [N,D],[N,P,D]
    g_adv, tok_adv = encode_image(model, preprocess, adv_imgs, device, args.batch)
    g_txt, q_tok = encode_text(model, tok, queries, device)
    LOG.info("feats done %.1fs N_img=%d P=%d N_q=%d", time.time() - t0,
             tok_orig.shape[0], tok_orig.shape[1], len(queries))

    # ---- global-cosine pool (standard CLIP dual-encoder) ----
    g_pool = (g_orig @ g_txt.T).numpy()                                    # [N,Q]
    g_adv_diag = np.array([float(g_adv[a2i[PID[r]]] @ g_txt[q2i[QRY[r]]]) for r in range(len(df))])
    # ---- late-interaction MaxSim pool ----
    li_pool = np.empty((len(asins), len(queries)), dtype=np.float32)
    for q, qi in q2i.items():
        li_pool[:, qi] = maxsim_scores(tok_orig, q_tok[qi])
    li_adv_diag = np.array([maxsim_scores(tok_adv[a2i[PID[r]]:a2i[PID[r]] + 1], q_tok[q2i[QRY[r]]])[0]
                            for r in range(len(df))])
    LOG.info("pool+diag scored %.1fs", time.time() - t0)

    def rank_lift(pool_mat, adv_diag, r):
        i = a2i[PID[r]]; qi = q2i[QRY[r]]
        scores = pool_mat[:, qi]
        rank_orig = int((scores >= scores[i]).sum())
        adv_scores = scores.copy(); adv_scores[i] = adv_diag[r]
        rank_adv = int((adv_scores >= adv_scores[i]).sum())
        return rank_orig, rank_adv

    rows = []
    for r in range(len(df)):
        i = a2i[PID[r]]; qi = q2i[QRY[r]]
        ro_g, ra_g = rank_lift(g_pool, g_adv_diag, r)
        ro_l, ra_l = rank_lift(li_pool, li_adv_diag, r)
        rows.append(dict(
            example_id=EXID[r], query_id=QID[r], query=QRY[r],
            product_id=PID[r], esci_label=LAB[r],
            cos_orig=float(g_pool[i, qi]), cos_adv=float(g_adv_diag[r]),
            cos_delta=float(g_adv_diag[r] - g_pool[i, qi]),
            rank_orig_global=ro_g, rank_adv_global=ra_g, rank_lift_global=ro_g - ra_g,
            lateint_orig=float(li_pool[i, qi]), lateint_adv=float(li_adv_diag[r]),
            lateint_delta=float(li_adv_diag[r] - li_pool[i, qi]),
            rank_orig_lateint=ro_l, rank_adv_lateint=ra_l, rank_lift_lateint=ro_l - ra_l,
        ))
    out = pd.DataFrame(rows)
    out_csv = args.out_dir / f"lateint_rank_{args.method_tag}_per_pair.csv"
    out.to_csv(out_csv, index=False, quoting=csv.QUOTE_NONNUMERIC)

    def per_label(col):
        return {lab: float(out.loc[out.esci_label == lab, col].mean())
                for lab in LABELS if (out.esci_label == lab).any()}
    summary = dict(
        paradigm="late_interaction_maxsim_colbert_style",
        clip_model=f"{args.clip_model}/{args.pretrained}", held_out=True,
        method_tag=args.method_tag, n_pairs=int(len(out)),
        patches=int(tok_orig.shape[1]),
        # global-cosine control (same held-out encoder + pool, standard CLIP scoring)
        rank_lift_global_mean=float(out.rank_lift_global.mean()),
        rank_lift_global_median=float(out.rank_lift_global.median()),
        per_label_rank_lift_global=per_label("rank_lift_global"),
        # late-interaction MaxSim paradigm (the headline number)
        lateint_delta_mean=float(out.lateint_delta.mean()),
        rank_lift_lateint_mean=float(out.rank_lift_lateint.mean()),
        rank_lift_lateint_median=float(out.rank_lift_lateint.median()),
        per_label_rank_lift_lateint=per_label("rank_lift_lateint"),
        per_label_lateint_delta=per_label("lateint_delta"),
    )
    (args.out_dir / f"lateint_rank_{args.method_tag}_summary.json").write_text(json.dumps(summary, indent=2))
    LOG.info("SUMMARY %s", json.dumps(summary, indent=2))
    print("LATEINT_RANK_OK", args.method_tag)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback; traceback.print_exc(); print("LATEINT_RANK_FAIL"); sys.exit(1)
