#!/usr/bin/env python3
"""Standalone rank evaluator -- SAME ESCI protocol as eval/eval_harness.py --mode rank.

Replicates eval_harness rank math EXACTLY:
  - sim = cosine( CLIP_image_feat, CLIP_text_feat(query) )  [query held out, eval-time only]
  - delta_held_out = sim_adv - sim_orig
  - distractor pool = all unique ASIN images in manifest (same 491 pool)
  - rank_orig[r] = (orig_col >= orig_col[i]).sum()  over the pool, for that pair's query col
  - rank_adv[r]  = same but row i swapped to adv sim
  - rank_lift = rank_orig - rank_adv
Only difference vs harness: image features computed with a GPU re-impl of the SAME
open_clip ViT-L/14 preprocessing (Resize224 bicubic+antialias->CenterCrop224->CLIP norm),
to avoid the harness's CPU-preproc stall on this box. Math/protocol identical.

Outputs: rank_nes_per_pair.csv + rank_nes_summary.json (same columns as harness).
"""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

LABELS = ("E", "S", "C", "I")
_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def load_clip(device):
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", device=str(device))
    tok = open_clip.get_tokenizer("ViT-L-14")
    model.eval()
    return model, tok


@torch.no_grad()
def gpu_preproc(paths, device, batch=32):
    feats = []
    import open_clip  # noqa
    for s in range(0, len(paths), batch):
        chunk = paths[s:s + batch]
        ims = []
        for p in chunk:
            a = np.array(Image.open(p).convert("RGB"), dtype=np.float32)
            ims.append(torch.from_numpy(a).permute(2, 0, 1))
        # pad to common size by resizing each to 224 directly (square thumbs)
        xs = []
        for t in ims:
            x = t.unsqueeze(0) / 255.0
            x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
            xs.append(x)
        x = torch.cat(xs, 0).to(device)
        x = (x - _MEAN.to(device)) / _STD.to(device)
        f = MODEL.encode_image(x)
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu())
    return torch.cat(feats, 0)


@torch.no_grad()
def encode_texts(tok, texts, device, batch=128):
    feats = []
    for s in range(0, len(texts), batch):
        t = tok(texts[s:s + batch]).to(device)
        f = MODEL.encode_text(t)
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu())
    return torch.cat(feats, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--adv-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--method-tag", default="nes")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    global MODEL
    MODEL, tok = load_clip(device)
    print("CLIP loaded", flush=True)

    df = pd.read_csv(args.manifest)
    img_dir = Path(args.img_dir); adv_dir = Path(args.adv_dir)
    df["_ip"] = df["product_id"].apply(lambda a: img_dir / f"{a}.jpg")
    df = df[df["_ip"].apply(lambda p: p.exists() and p.stat().st_size > 0)].reset_index(drop=True)

    asins = df["product_id"].drop_duplicates().tolist()
    # keep only ASINs whose adv exists (matches harness filter); here all 491 exist
    missing = [a for a in asins if not (adv_dir / f"{a}.jpg").exists()]
    if missing:
        df = df[~df["product_id"].isin(missing)].reset_index(drop=True)
        asins = df["product_id"].drop_duplicates().tolist()
    asin2idx = {a: i for i, a in enumerate(asins)}
    img_paths = [img_dir / f"{a}.jpg" for a in asins]
    adv_paths = [adv_dir / f"{a}.jpg" for a in asins]

    queries = df["query"].drop_duplicates().tolist()
    q2idx = {q: i for i, q in enumerate(queries)}

    t0 = time.time()
    img_feats = gpu_preproc(img_paths, device)
    print(f"orig img encode {time.time()-t0:.1f}s shape={tuple(img_feats.shape)}", flush=True)
    t0 = time.time()
    adv_feats = gpu_preproc(adv_paths, device)
    print(f"adv img encode {time.time()-t0:.1f}s shape={tuple(adv_feats.shape)}", flush=True)
    t0 = time.time()
    txt_feats = encode_texts(tok, queries, device)
    print(f"text encode {time.time()-t0:.1f}s shape={tuple(txt_feats.shape)}", flush=True)

    img_idx = df["product_id"].map(asin2idx).to_numpy()
    txt_idx = df["query"].map(q2idx).to_numpy()

    sims_orig = (img_feats[img_idx] * txt_feats[txt_idx]).sum(dim=-1).numpy()
    sims_adv = (adv_feats[img_idx] * txt_feats[txt_idx]).sum(dim=-1).numpy()
    df["sim_orig"] = sims_orig
    df["sim_adv"] = sims_adv
    df["delta_held_out"] = sims_adv - sims_orig

    sims_pool_orig = (img_feats @ txt_feats.T).numpy()  # [N_img, N_query]
    sims_pool_adv = (adv_feats @ txt_feats.T).numpy()
    n = len(df)
    rank_orig = np.empty(n, dtype=int); rank_adv = np.empty(n, dtype=int)
    for r in range(n):
        i = img_idx[r]; q = txt_idx[r]
        oc = sims_pool_orig[:, q]
        rank_orig[r] = int((oc >= oc[i]).sum())
        ac = sims_pool_orig[:, q].copy()
        ac[i] = sims_pool_adv[i, q]
        rank_adv[r] = int((ac >= ac[i]).sum())
    df["rank_orig"] = rank_orig
    df["rank_adv"] = rank_adv
    df["rank_lift"] = rank_orig - rank_adv
    print(f"rank loop done {time.time()-t0:.1f}s", flush=True)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"rank_{args.method_tag}_per_pair.csv"
    df[["example_id", "query_id", "query", "product_id", "product_title",
        "esci_label", "sim_orig", "sim_adv", "delta_held_out",
        "rank_orig", "rank_adv", "rank_lift"]].to_csv(out_csv, index=False, quoting=csv.QUOTE_NONNUMERIC)
    summ = {
        "mode": "rank", "method_tag": args.method_tag, "n_pairs": int(n),
        "delta_held_out_mean": float(df["delta_held_out"].mean()),
        "delta_held_out_ci95_half": float(1.96 * df["delta_held_out"].std(ddof=1) / np.sqrt(max(1, n))),
        "rank_lift_mean": float(df["rank_lift"].mean()),
        "rank_lift_median": float(df["rank_lift"].median()),
        "per_label_delta_mean": {lab: float(df.loc[df["esci_label"] == lab, "delta_held_out"].mean())
                                 for lab in LABELS if (df["esci_label"] == lab).any()},
    }
    (out_dir / f"rank_{args.method_tag}_summary.json").write_text(json.dumps(summ, indent=2))
    print("RANK_SUMMARY " + json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
