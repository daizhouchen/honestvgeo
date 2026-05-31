# HonestVGEO

Source and reproduction materials for **"Recalibrating Visual GEO: An Anchor-Decoupled Evaluation Protocol and a Cross-Encoder Transfer Probe on ESCI"** (ICDM 2026 submission).

## Contents
- `main.tex`, `sections/`, `references.bib`, `main.bbl` — manuscript source (IEEEtran, 10 pages).
- `build.sh` — build the PDF (`pdflatex` + `bibtex` ×2). Produces `main.pdf`.
- `main.pdf` — compiled manuscript.
- `figures/make_*.py` — figure-generation scripts; **every plotted value is hardcoded to match the manuscript tables and the underlying experiment data** so figures and tables cannot drift.
- `figures/_data/` — the experiment cell summaries (rank-lift means per cell) consumed by the composite results figure, so `make_results_composite.py` is reproducible standalone.
- `figures/*.pdf`, `figures/*.png` — rendered figures.

## Build
```bash
bash build.sh    # -> main.pdf, prints page count + warnings
```
Requires a TeX distribution with IEEEtran (TinyTeX or TeX Live).

## Notes
- This bundle is the **paper + figure code + figure data**. The full evaluation harness (anchor⊥query verifier, rank-lift scorer, the four attack implementations, the ESCI manifest, and the AE-CoGEO consensus attack) is the larger reproduction package referenced in the paper; it is maintained separately.
- All numbers in the manuscript trace to measured experiment outputs.

## Status
Private backup during peer review (triple-blind). To be opened publicly at camera-ready.
