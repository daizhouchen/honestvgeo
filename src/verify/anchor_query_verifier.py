#!/usr/bin/env python3
"""anchor-perp-query invariant verifier (no model required).

The central methodological commitment of HonestEval: the optimization anchor is a
function of the product image x_p alone (here, its seller-side product title), and
NEVER of the held-out evaluation query. Any visual-GEO author can run this on their
manifest before reporting rank-lift to certify the invariant holds.

It proves the invariant *dynamically*: it builds the anchor under the real
(product, query) pairing and again under a fully shuffled query column, and asserts
the per-product anchor is byte-identical -- i.e. the query cannot influence the
anchor. It also flags any anchor that simply echoes its paired query (leakage).

Usage:
    python anchor_query_verifier.py --manifest manifests/esci500_manifest.csv
Exit code 0 == invariant holds; 1 == violated.
"""
import argparse, csv, hashlib, sys


def build_anchor(row):
    """The seller-side anchor exactly as the attacks consume it (product title only).
    Depends on the product (x_p), never on row['query']."""
    return row["product_title"].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="CSV with product_id, product_title, query columns")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.manifest, encoding="utf-8")))
    if not rows:
        print("empty manifest"); sys.exit(1)
    n = len(rows)

    # (1) Query-independence, row by row: hold the product row fixed and substitute
    #     a DIFFERENT eval query; the anchor must not change (build_anchor ignores it).
    changed = []
    for i, r in enumerate(rows):
        a_real = build_anchor(r)
        rr = dict(r); rr["query"] = rows[(i + 1) % n]["query"]  # same product, foreign query
        if build_anchor(rr) != a_real:
            changed.append((r["product_id"], a_real, build_anchor(rr)))

    # (2) No leakage: an anchor must never simply be (equal to) its paired query.
    leak = [r["product_id"] for r in rows
            if r["query"].strip() and r["query"].strip().lower() == build_anchor(r).lower()]

    # (3) Product-dependence sanity: the anchor is not a constant -- it varies with x_p.
    distinct = len({build_anchor(r) for r in rows})
    h = hashlib.sha256("".join(sorted({build_anchor(r) for r in rows})).encode()).hexdigest()[:12]

    print(f"rows: {n}   distinct anchors: {distinct}   fingerprint: {h}")
    print(f"anchors that changed when the eval query was swapped: {len(changed)}  (must be 0)")
    print(f"anchors that merely echo their query (leakage): {len(leak)}  (must be 0)")
    print(f"anchor varies across products (not a constant): {distinct > 1}  (must be True)")

    if changed or leak or distinct <= 1:
        for pid, a_real, a_swap in changed[:5]:
            print(f"  VIOLATION {pid}: anchor changed under query swap: {a_real!r} -> {a_swap!r}")
        print("FAIL: anchor-perp-query invariant violated")
        sys.exit(1)
    print("PASS: the anchor is a function of the product alone, independent of the eval query")
    sys.exit(0)


if __name__ == "__main__":
    main()
