#!/usr/bin/env bash
# Authoritative build = TinyTeX pdflatex+bibtex (the real submission engine).
# references.bib is the source of truth; bibtex regenerates main.bbl each build.
# Usage: bash build.sh   -> main.pdf + prints page count (ICDM hard limit: 10) + warnings.
set -e
export PATH=/home/zcdai/.TinyTeX/bin/x86_64-linux:$PATH
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/tmp/icdm_b1.log 2>&1 || { echo "PASS1 FAILED"; tail -25 /tmp/icdm_b1.log; exit 1; }
bibtex main >/tmp/icdm_bib.log 2>&1 || true
pdflatex -interaction=nonstopmode main.tex >/tmp/icdm_b2.log 2>&1 || true
pdflatex -interaction=nonstopmode main.tex >/tmp/icdm_b3.log 2>&1 || true
PAGES=$(python3 -c "import pypdf; print(len(pypdf.PdfReader('main.pdf').pages))" 2>/dev/null || echo '?')
echo "PAGES: $PAGES   (ICDM hard limit: 10)"
echo "undefined cite/ref: $(grep -ci 'undefined' /tmp/icdm_b3.log || true)"
echo "overfull hboxes: $(grep -c 'Overfull' /tmp/icdm_b3.log || true)"
