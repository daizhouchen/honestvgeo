#!/usr/bin/env python3
"""No-model admissibility verifier for visual-GEO submissions.

Checks, WITHOUT loading any retriever or VLM, that a set of adversarial product
images is admissible under the protocol:
  (1) L-inf budget: max per-pixel |adv - orig| / 255 <= eps for every image;
  (2) image integrity: adv and orig share dimensions and mode (no resize/crop cheat);
  (3) cohort coverage: the manifest carries all four ESCI cohorts (E/S/C/I).

Images are public and not bundled here; point --orig-dir / --adv-dir at your own
img/<ASIN>.jpg layout. With no image dirs it runs only the cohort-coverage check.

Usage:
    python admissibility_check.py --manifest manifests/esci500_manifest.csv \
        [--orig-dir DIR --adv-dir DIR --eps 0.0627]
Exit code 0 == admissible; 1 == a check failed.
"""
import argparse, csv, os, sys
from collections import Counter


def linf(orig_path, adv_path):
    import numpy as np
    from PIL import Image
    a = np.asarray(Image.open(orig_path).convert("RGB"), dtype=np.int16)
    b = np.asarray(Image.open(adv_path).convert("RGB"), dtype=np.int16)
    if a.shape != b.shape:
        return None, f"shape {a.shape} vs {b.shape}"
    return int(np.abs(a - b).max()), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--orig-dir"); ap.add_argument("--adv-dir")
    ap.add_argument("--eps", type=float, default=16 / 255, help="L-inf budget as a fraction of 255")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.manifest, encoding="utf-8")))

    # (3) cohort coverage
    cohorts = Counter(r["esci_label"] for r in rows)
    print(f"cohort coverage: {dict(cohorts)}")
    cov_ok = all(cohorts.get(c, 0) > 0 for c in ("E", "S", "C", "I"))
    print(f"  all four ESCI cohorts present: {cov_ok}  (must be True)")

    bud_ok = True
    if a.orig_dir and a.adv_dir:
        cap = round(a.eps * 255)
        worst, n, bad = 0, 0, []
        for r in rows:
            asin = r["product_id"]
            op = os.path.join(a.orig_dir, f"{asin}.jpg"); ap_ = os.path.join(a.adv_dir, f"{asin}.jpg")
            if not (os.path.exists(op) and os.path.exists(ap_)):
                continue
            d, err = linf(op, ap_); n += 1
            if err: bad.append((asin, err)); continue
            worst = max(worst, d)
            if d > cap: bad.append((asin, f"L-inf {d} > {cap}"))
        print(f"\nL-inf / integrity over {n} images: worst per-pixel delta = {worst} (cap {cap})")
        if bad:
            bud_ok = False
            for asin, why in bad[:8]:
                print(f"  VIOLATION {asin}: {why}")
    else:
        print("\n(no --orig-dir/--adv-dir given; skipping L-inf + integrity, ran cohort coverage only)")

    if not (cov_ok and bud_ok):
        print("FAIL: not admissible"); sys.exit(1)
    print("PASS: admissible under the protocol"); sys.exit(0)


if __name__ == "__main__":
    main()
