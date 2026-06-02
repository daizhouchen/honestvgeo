#!/usr/bin/env python3
"""ESCI subset loader for the N3 real-relevance retrieval evaluation.

Reads ESCI parquet (either the original split-form `amazon-science/esci-data`
or the pre-joined `tasksource/esci`), filters to US locale + small_version=1,
samples N pairs balanced over {E, S, C, I} relevance labels, and attempts
to fetch the product image for each ASIN from Amazon's public CDN. Writes
a manifest CSV listing only pairs whose image fetch succeeded.

Schema source (auto-detected):

* Joined: tasksource/esci style -- one parquet with columns
  ``example_id, query, query_id, product_id, product_locale, esci_label,
   small_version, large_version, product_title, ...``
* Split: amazon-science/esci-data style -- examples.parquet (without
  product_title) + products.parquet (with product_title) joined on
  (product_id, product_locale).

Usage::

    # joined (one parquet)
    python esci_loader.py \\
        --esci-parquet /data/tasksource_esci/test-00000.parquet \\
        --out-dir      /data/esci500 \\
        --target-pairs 500

    # split form (two parquets in shopping_queries_dataset/)
    python esci_loader.py \\
        --esci-root /data/esci-data \\
        --out-dir   /data/esci500
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from PIL import Image

LOG = logging.getLogger("esci_loader")

LABELS = ("E", "S", "C", "I")

# Map possible label encodings into the canonical single-letter form used
# throughout the rest of the N3 pipeline. tasksource/esci uses full English
# names; amazon-science/esci-data uses single letters.
_LABEL_NORMALIZE = {
    "E": "E", "S": "S", "C": "C", "I": "I",
    "Exact": "E", "Substitute": "S", "Complement": "C", "Irrelevant": "I",
}

# Amazon public image URL patterns (tried in order).
# `{a}` = ASIN; many ASINs return a 1x1 placeholder PNG when the slot has no
# associated image — we filter those out by min image size.
URL_PATTERNS = (
    "https://m.media-amazon.com/images/P/{a}.01._SCRMZZZZZZ_.jpg",
    "https://m.media-amazon.com/images/P/{a}.01.LZZZZZZZ.jpg",
    "https://images-na.ssl-images-amazon.com/images/P/{a}.01.L.jpg",
    "https://images-na.ssl-images-amazon.com/images/P/{a}.01.LZZZZZZZ.jpg",
)

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"


def fetch_image(asin: str, out_path: Path, timeout: float = 8.0,
                min_bytes: int = 4096, min_dim: int = 80) -> tuple[bool, str]:
    """Try several Amazon CDN URL patterns. Save first success >= thresholds."""
    if out_path.exists() and out_path.stat().st_size >= min_bytes:
        return True, "cached"
    headers = {"User-Agent": UA, "Accept": "image/avif,image/webp,image/jpeg,*/*"}
    last_err = "no_url_attempted"
    for url in URL_PATTERNS:
        full_url = url.format(a=asin)
        try:
            r = requests.get(full_url, headers=headers, timeout=timeout, stream=True)
            if r.status_code != 200:
                last_err = f"http_{r.status_code}"
                continue
            content = r.content
            if len(content) < min_bytes:
                last_err = f"too_small_{len(content)}b"
                continue
            try:
                img = Image.open(io.BytesIO(content))
                img.load()
                w, h = img.size
            except Exception as exc:
                last_err = f"decode_error:{exc.__class__.__name__}"
                continue
            if min(w, h) < min_dim:
                last_err = f"dim_{w}x{h}"
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "wb") as fh:
                fh.write(content)
            return True, f"ok:{w}x{h}"
        except requests.RequestException as exc:
            last_err = f"req_error:{exc.__class__.__name__}"
            continue
    return False, last_err


def load_esci_joined(parquet_path: Path) -> pd.DataFrame:
    """Load joined ESCI (tasksource style)."""
    LOG.info("Reading joined ESCI parquet: %s", parquet_path)
    df = pd.read_parquet(parquet_path)
    needed = {"example_id", "query", "query_id", "product_id", "product_locale",
              "esci_label", "small_version", "product_title"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"Joined parquet missing columns {missing}")
    LOG.info("joined rows=%d, columns=%d", len(df), len(df.columns))
    return df


def load_esci_split(esci_root: Path) -> pd.DataFrame:
    """Load split ESCI (amazon-science/esci-data) and join on (product_id, locale)."""
    sqd = esci_root / "shopping_queries_dataset"
    examples_path = sqd / "shopping_queries_dataset_examples.parquet"
    products_path = sqd / "shopping_queries_dataset_products.parquet"
    if not examples_path.exists() or not products_path.exists():
        raise SystemExit(
            f"ESCI parquet not found under {sqd}. "
            f"Expected {examples_path.name} and {products_path.name}."
        )
    LOG.info("Reading examples parquet: %s", examples_path)
    examples = pd.read_parquet(examples_path)
    LOG.info("Reading products parquet: %s", products_path)
    products = pd.read_parquet(products_path)
    LOG.info("examples=%d, products=%d", len(examples), len(products))
    join = examples.merge(
        products[["product_id", "product_locale", "product_title"]],
        on=["product_id", "product_locale"], how="inner",
    )
    LOG.info("split-form join rows=%d", len(join))
    return join


def sample_balanced(merged: pd.DataFrame, target: int, seed: int) -> pd.DataFrame:
    """Sample target pairs with equal allocation across {E, S, C, I}."""
    per_label = target // len(LABELS)
    parts = []
    for lab in LABELS:
        sub = merged[merged["esci_label"] == lab]
        if len(sub) == 0:
            LOG.warning("label=%s has 0 rows after filter", lab)
            continue
        n = min(per_label, len(sub))
        parts.append(sub.sample(n=n, random_state=seed))
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--esci-parquet", type=Path, default=None,
                   help="Single joined parquet (tasksource/esci) -- preferred fast path")
    p.add_argument("--esci-root", type=Path, default=None,
                   help="Path to clone of amazon-science/esci-data (split-form fallback)")
    p.add_argument("--out-dir", type=Path, required=True, help="Output dir; will write manifest.csv + img/")
    p.add_argument("--target-pairs", type=int, default=500, help="Desired #successful pairs after image fetch")
    p.add_argument("--over-sample", type=float, default=3.0,
                   help="Multiplier on target before image-fetch attempt; raises probability of meeting target")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--locale", type=str, default="us", help="Locale to filter (us|es|jp). Use 'us' for EN.")
    p.add_argument("--small-version-only", action="store_true", default=True,
                   help="Restrict to small_version==1 (the canonical ESCI-S balanced subset)")
    p.add_argument("--workers", type=int, default=8, help="Parallel image-fetch workers")
    p.add_argument("--timeout", type=float, default=8.0)
    p.add_argument("--log-level", type=str, default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = args.out_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)

    if args.esci_parquet is not None:
        join = load_esci_joined(args.esci_parquet)
    elif args.esci_root is not None:
        join = load_esci_split(args.esci_root)
    else:
        raise SystemExit("Provide either --esci-parquet (joined) or --esci-root (split form).")

    # Normalize esci_label to single-letter form so downstream code is uniform.
    join = join.copy()
    join["esci_label"] = join["esci_label"].map(_LABEL_NORMALIZE).fillna(join["esci_label"])
    bad = ~join["esci_label"].isin(LABELS)
    if bad.any():
        LOG.warning("dropping %d rows with unrecognized esci_label values", int(bad.sum()))
        join = join[~bad]

    # Filter to locale + small_version + non-empty product title.
    join = join[join["product_locale"].str.lower() == args.locale.lower()]
    if args.small_version_only and "small_version" in join.columns:
        join = join[join["small_version"] == 1]
    join = join[join["product_title"].notna() & (join["product_title"].str.len() > 0)]
    LOG.info("after locale=%s + small_version + title filter: %d rows", args.locale, len(join))

    # Sample target * over_sample, balanced across E/S/C/I.
    target_total = int(args.target_pairs * args.over_sample)
    sampled = sample_balanced(join, target_total, args.seed)
    LOG.info("sampled %d candidate pairs (will attempt image fetch in parallel)", len(sampled))

    # Image fetch in parallel, retain first success per ASIN.
    asins = sampled["product_id"].unique().tolist()
    LOG.info("unique ASINs: %d", len(asins))
    fetch_t0 = time.time()
    success: dict[str, str] = {}  # asin -> reason
    failure: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(fetch_image, a, img_dir / f"{a}.jpg", args.timeout): a for a in asins}
        for i, f in enumerate(as_completed(futures), 1):
            a = futures[f]
            try:
                ok, reason = f.result()
            except Exception as exc:
                ok, reason = False, f"future_exc:{exc.__class__.__name__}"
            (success if ok else failure)[a] = reason
            if i % 50 == 0:
                LOG.info("fetch progress: %d/%d  ok=%d  fail=%d  elapsed=%.1fs",
                         i, len(asins), len(success), len(failure), time.time() - fetch_t0)
    fetch_dt = time.time() - fetch_t0
    LOG.info("image fetch done: ok=%d / total=%d (%.1f%%) in %.1fs",
             len(success), len(asins), 100.0 * len(success) / max(1, len(asins)), fetch_dt)

    # Filter sampled rows to ASINs whose image fetch succeeded.
    survived = sampled[sampled["product_id"].isin(success)].copy()

    # Re-balance to target_pairs across labels (best-effort).
    rebalanced_parts = []
    per_label_target = args.target_pairs // len(LABELS)
    for lab in LABELS:
        sub = survived[survived["esci_label"] == lab]
        n = min(per_label_target, len(sub))
        if n < per_label_target:
            LOG.warning("label=%s only has %d pairs after image-fetch filter (target %d)",
                        lab, n, per_label_target)
        rebalanced_parts.append(sub.head(n))
    final = pd.concat(rebalanced_parts, ignore_index=True)
    LOG.info("final manifest: %d pairs (target=%d)", len(final), args.target_pairs)

    # Write manifest CSV (eval_harness reads this).
    manifest_path = args.out_dir / "manifest.csv"
    final_out = final[["example_id", "query_id", "query", "product_id",
                       "product_title", "esci_label"]].copy()
    final_out["image_path"] = final_out["product_id"].apply(lambda a: str(img_dir / f"{a}.jpg"))
    final_out.to_csv(manifest_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    LOG.info("wrote manifest: %s", manifest_path)

    # Write summary.json with provenance.
    summary = {
        "esci_root": str(args.esci_root),
        "out_dir": str(args.out_dir),
        "locale": args.locale,
        "small_version_only": args.small_version_only,
        "seed": args.seed,
        "target_pairs": args.target_pairs,
        "over_sample": args.over_sample,
        "candidates_attempted": len(asins),
        "image_fetch_ok": len(success),
        "image_fetch_failed": len(failure),
        "image_fetch_success_rate": len(success) / max(1, len(asins)),
        "image_fetch_seconds": round(fetch_dt, 2),
        "final_pairs": int(len(final)),
        "label_distribution": {lab: int((final["esci_label"] == lab).sum()) for lab in LABELS},
        "manifest_path": str(manifest_path),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    LOG.info("summary: %s", json.dumps(summary, indent=2))

    # Write fetch failure log for diagnostics.
    if failure:
        fail_log = args.out_dir / "image_fetch_failures.csv"
        with open(fail_log, "w") as fh:
            w = csv.writer(fh)
            w.writerow(["asin", "reason"])
            for a, r in failure.items():
                w.writerow([a, r])
        LOG.info("wrote failure log: %s (%d entries)", fail_log, len(failure))

    if len(final) < args.target_pairs:
        LOG.warning("final pair count %d < target %d — caller may need to switch image source",
                    len(final), args.target_pairs)
        return 2  # soft-fail: caller decides whether to continue
    return 0


if __name__ == "__main__":
    sys.exit(main())
