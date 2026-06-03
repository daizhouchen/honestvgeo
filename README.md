# HonestEval — reproduction artifact

Code, manifests, and result tables for **"Recalibrating Visual GEO:
Anchor-Decoupled Evaluation and a Transfer Probe"** (ICDM 2026 submission).

> Reproduction artifact only — the manuscript is submitted through the conference
> system, not here. Anonymized for triple-blind review. Datasets (Amazon ESCI,
> Food-101) and model weights (CLIP/OpenCLIP/EVA-02/SigLIP, BLIP/BLIP-2, VLM
> captioners) are public and fetched on demand; they are not bundled. Paths use
> the placeholders `${PROJECT_ROOT}` / `${CONDA_BASE}` / `${HOME}`.

## Layout

```
.
├── requirements.txt          # Python dependencies
├── ENVIRONMENT.md            # software / hardware the experiments ran under
├── src/                      # all reproduction code, grouped by role
│   ├── method/               # CoGEO building blocks: envsim, diffjpeg_lite, vlm caption->anchor
│   ├── attacks/              # CoGEO (n3_attack.py), baselines (PGD-bare/AdvCLIP/Co-Attack),
│   │                         #   AE-CoGEO transfer probe (n3_attack_ae*.py), diffusion-prior,
│   │                         #   gradient-free NES, cohort-adaptive-budget
│   ├── verify/               # standalone HonestEval verifiers: anchor_query_verifier.py
│   │                         #   (anchor-perp-query invariant, no model) + admissibility_check.py
│   │                         #   (no-model L-inf budget / integrity / cohort coverage)
│   ├── eval/                 # eval_harness.py = rank-lift scorer + pre-flight retriever-
│   │                         #   discrimination gate (gap_{E-I}); BLIP/BLIP-2/MaxSim harnesses; NES eval
│   ├── defense/              # input purification, detector-gated purification, training-free
│   │                         #   Laplacian detector + generalization
│   ├── data/                 # ESCI / Food-101 loaders
│   ├── analysis/             # statistics + aggregation -> the reported tables
│   └── scripts/              # per-experiment launchers (*.sh); each resolves its own paths
├── manifests/                # esci500_manifest.csv (491 imgs / 500 triples),
│                             #   esci1500_manifest.csv (1,430-pair scale-up)
│                             #   cols: example_id, query_id, query, product_id (ASIN),
│                             #   product_title, esci_label, image_path (img/<ASIN>.jpg)
├── results/                  # reference result tables, grouped by paper theme — see results/README.md
│   ├── white_box_4way/  backbones/  transfer/  scale_up/
│   ├── defense/  five_family/  ablations/
│   └── esci_full_aggregate.json
└── figures/                  # make_*.py — regenerate every plot from ../results/
```

## Documentation

Every directory has its own README with file-by-file detail:

- [`src/README.md`](src/README.md) — code organized by role (method / attacks / eval /
  defense / data / analysis / scripts), the attack→score→aggregate pipeline, and a
  table describing every module and launcher.
- [`results/README.md`](results/README.md) — the seven result themes and which paper
  table/figure each maps to.
- [`figures/README.md`](figures/README.md) — which `make_*.py` produces which figure,
  and what data it reads.
- [`manifests/README.md`](manifests/README.md) — manifest columns and how to obtain
  the (public, unbundled) images.
- [`ENVIRONMENT.md`](ENVIRONMENT.md) — software/hardware the experiments ran under.

## Where each artifact item the paper references lives

| Item | Here |
|---|---|
| anchor⊥query invariant verifier (standalone, no model) | `src/verify/anchor_query_verifier.py` |
| no-model admissibility verifier (L∞ budget / integrity / cohort coverage) | `src/verify/admissibility_check.py` |
| pre-flight retriever-discrimination gate | `src/eval/eval_harness.py` (`gap_E_minus_I`) |
| reference rank-lift scorer | `src/eval/eval_harness.py` |
| CoGEO attack | `src/attacks/n3_attack.py` (blocks in `src/method/`) |
| AE-CoGEO cross-encoder transfer probe | `src/attacks/n3_attack_ae.py`, `n3_attack_ae_ens.py` |
| three baselines | `src/attacks/n3_attack_baselines.py` |
| ESCI manifest | `manifests/` |
| reference result tables | `results/` (indexed in `results/README.md`) |
| reproduction configuration (seeds, hyper-params, hardware) | `src/scripts/run_*.sh` + `ENVIRONMENT.md` |

## Reproduce

1. `pip install -r requirements.txt`; set `PROJECT_ROOT`.
2. **Verify the protocol (no model, seconds):**
   `python src/verify/anchor_query_verifier.py --manifest manifests/esci500_manifest.csv`
   certifies the anchor⊥query invariant; `src/verify/admissibility_check.py` checks the
   L∞ budget / integrity / cohort coverage.
3. Fetch the public datasets/weights (ESCI & Food-101 via HuggingFace; CLIP-family
   + BLIP weights on demand). Images are addressed `img/<ASIN>.jpg` per manifest.
4. **Attack:** `bash src/scripts/run_4way.sh` (white-box four-way),
   `bash src/scripts/ae_ens_run.sh` (AE-CoGEO consensus transfer), etc. Each script
   resolves the code paths itself; outputs land under `runs/`.
5. **Score:** `src/eval/eval_harness.py` emits per-image rank-lift CSVs and the
   admissibility gate; `src/analysis/n3_stats*.py` aggregates to the reported tables.
6. **Figures:** `python figures/make_*.py` regenerate every plot from `results/`.

The bundled `results/` are the reference outputs; rerunning the pipeline writes fresh
outputs under `runs/`. Plotted values are hardcoded to the measured outputs so
figures and tables cannot drift.

> Naming note: some files keep an `n3_` prefix — an internal experiment-series tag,
> not an identifier; it is wired through imports and result-folder names and kept to
> preserve a self-consistent, runnable graph.

## Status

Private during peer review (triple-blind); to be opened at camera-ready.
