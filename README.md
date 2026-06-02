# HonestEval — Recalibrating Visual GEO

Reproduction artifact for **"Recalibrating Visual GEO: Anchor-Decoupled Evaluation and a Transfer Probe"** (ICDM 2026 submission).

> Anonymized for triple-blind review. Datasets (Amazon ESCI, Food-101) and model
> weights (CLIP/OpenCLIP/EVA-02/SigLIP, BLIP/BLIP-2, VLM captioners) are **public**
> and fetched on demand; they are **not** bundled here. Absolute paths in scripts
> are replaced by the placeholders `${PROJECT_ROOT}` / `${CONDA_BASE}` / `${HOME}`.

## Layout

```
.
├── main.tex, sections/, references.bib, main.bbl, build.sh, main.pdf   # manuscript (IEEEtran, 10 pp)
├── figures/                 # figure-generation scripts (make_*.py), input data (_data/), rendered figures
├── code/                    # core re-implementation
│   ├── cogeo.py             # CoGEO: product-title anchor + Sobel mask + differentiable JPEG + MI-FGSM PGD
│   ├── envsim.py            # DIM / SIM / TIM environment-simulation transforms
│   ├── diffjpeg_lite.py     # differentiable JPEG layer
│   ├── vlm_anchor.py        # VLM caption -> CLIP-text anchor (captioner-anchor ablation)
│   ├── dataset_*.py         # ESCI / Food-101 / SOP loaders
│   └── requirements.txt
├── n3/code/                 # evaluation + attack + analysis drivers
│   ├── eval_harness.py      # rank-lift scorer + the pre-flight no-op admissibility gate (gap_{E-I})
│   ├── n3_attack.py         # CoGEO attack driver       n3_attack_baselines.py  # PGD-bare / AdvCLIP / Co-Attack
│   ├── n3_attack_ae.py, n3_attack_ae_ens.py            # AE-CoGEO single- / multi-source consensus transfer probe
│   ├── n3_attack_diffprior.py, n3_attack_gfree.py, nes_attack.py, rank_eval_nes.py   # diffusion-prior + NES gradient-free
│   ├── n3_attack_cab.py     # cohort-adaptive-budget attacker
│   ├── eval_harness_blip.py / _blip2.py / _lateint.py  # BLIP / BLIP-2 / MaxSim cross-paradigm transfer
│   ├── purify_dir.py, gated_eps16.py                   # input purification + detector-gated purification
│   ├── ae_cogeo_detect.py, b1_detector_generalization.py   # training-free Laplacian detector + generalization
│   ├── n3_stats.py, n3_stats_v2.py, agg_r5_scaling.py, r4_aggregate.py, harvest_r2_compare.py   # statistics / aggregation
│   └── run_*.sh             # per-experiment launchers
└── repro/
    ├── manifests/           # esci500_manifest.csv (491 imgs / 560 triples), esci1500_manifest.csv (1,430-pair scale-up)
    │                        #   columns: example_id, query_id, query, product_id (ASIN), product_title, esci_label, image_path
    ├── INVENTORY.md         # environment / hardware snapshot (anonymized)
    └── STRUCTURE.md         # original workspace layout (anonymized)
```

## Maps to the artifact items referenced in the paper

| Paper promises | Here |
|---|---|
| anchor⊥query / no-op admissibility verifier | `n3/code/eval_harness.py` (`gap_E_minus_I` pre-flight gate) |
| reference rank-lift scorer | `n3/code/eval_harness.py` |
| reference implementations of both attacks | `code/cogeo.py` + `n3/code/n3_attack.py` (CoGEO); `n3/code/n3_attack_ae*.py` (AE-CoGEO) |
| three baselines | `n3/code/n3_attack_baselines.py` (PGD-bare, AdvCLIP, Co-Attack) |
| ESCI manifest | `repro/manifests/esci500_manifest.csv`, `esci1500_manifest.csv` |
| reference result tables / figure data | `figures/_data/` (cell summaries consumed by the composite figures) |
| reproduction configuration (seeds, hyper-params, hardware) | run scripts in `n3/code/run_*.sh` + `repro/INVENTORY.md` |

## Build the manuscript

```bash
bash build.sh        # pdflatex + bibtex x2 -> main.pdf ; prints page count + warnings
```
Requires a TeX distribution with IEEEtran (TinyTeX or TeX Live).

## Reproduce experiments (high level)

1. Create the conda env and install `code/requirements.txt`.
2. Set `PROJECT_ROOT` and fetch the public datasets/weights (ESCI & Food-101 via HuggingFace; CLIP-family + BLIP weights on demand). Images are addressed as `img/<ASIN>.jpg` relative to each manifest.
3. Attack: `n3/code/run_4way.sh` (white-box four-way), `n3/code/ae_ens_run.sh` (AE-CoGEO consensus transfer), etc.
4. Score: `n3/code/eval_harness.py` produces per-image rank-lift CSVs + the admissibility gate; `n3/code/n3_stats*.py` aggregates to the manuscript tables.

All plotted/tabulated numbers trace to measured experiment outputs; figure scripts hardcode the manuscript values so figures and tables cannot drift.

## Status

Private during peer review (triple-blind); to be opened at camera-ready.
