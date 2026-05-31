#!/usr/bin/env python3
"""Un-clip figure labels by enlarging the PDF MediaBox.

Some matplotlib-exported figure PDFs were saved with a tight bounding box that
slightly under-estimated text extent, so axis-title / colorbar-label ink is
drawn in the content stream but cropped by the MediaBox. This adds margin on
the sides where text is clipped so the already-drawn text becomes fully
visible. It changes ONLY the visible canvas -- no plotted data, no marker
positions, no values.

Padding is directional per figure. Note: figures are included at a fixed
\\columnwidth/\\textwidth, so adding *width* shrinks the displayed height while
adding *height* grows it. fig_ae_scaling is therefore padded on the left/right
only (its only clipped text is the right-edge x-axis label), which keeps the
main paper at 10 pages.

Run against the pristine (git-clean) figure PDFs. Idempotent: a figure already
free of text overflow is left untouched.

Usage: python3 fix_clipped_figure_margins.py
"""
import os
import fitz  # PyMuPDF

TOL = 0.5
# (left, right, top, bottom) padding in points
PAD = {
    "fig_ae_scaling.pdf": (8.0, 8.0, 0.0, 0.0),   # clip is right-edge x-label only; width-only keeps height
    "fig_d_heatmap.pdf":  (2.0, 8.0, 2.0, 8.0),   # clip is bottom x-label (+ tiny colorbar right)
}
HERE = os.path.dirname(os.path.abspath(__file__))


def text_overflow(page):
    r = page.rect
    bad = []
    for w in page.get_text("words"):
        x0, y0, x1, y1, txt = w[0], w[1], w[2], w[3], w[4]
        if x1 > r.width + TOL or x0 < -TOL or y1 > r.height + TOL or y0 < -TOL:
            bad.append(txt)
    return bad


def main():
    for fn, (pl, pr, pt, pb) in PAD.items():
        path = os.path.join(HERE, fn)
        if not os.path.exists(path):
            print(f"SKIP {fn}: not found")
            continue
        d = fitz.open(path)
        pg = d[0]
        before = text_overflow(pg)
        if not before:
            print(f"OK   {fn}: no clipped text, left untouched")
            d.close()
            continue
        mb = pg.mediabox
        # MediaBox uses a bottom-left origin: x grows right, y grows up.
        # pb (visual bottom) lowers y0; pt (visual top) raises y1.
        pg.set_mediabox(fitz.Rect(mb.x0 - pl, mb.y0 - pb, mb.x1 + pr, mb.y1 + pt))
        tmp = path + ".tmp"
        d.save(tmp)
        d.close()
        os.replace(tmp, path)
        d2 = fitz.open(path)
        r2 = d2[0].rect
        after = text_overflow(d2[0])
        d2.close()
        print(f"FIX  {fn}: clipped={before} -> overflow={after} new_box=%.0fx%.0f" % (r2.width, r2.height))


if __name__ == "__main__":
    main()
