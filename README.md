# HonestEval — reproduction artifact

Code, manifests, and figure-generation scripts for **"Recalibrating Visual GEO:
Anchor-Decoupled Evaluation and a Transfer Probe"** (ICDM 2026 submission).

> This is the **reproduction artifact only** — the manuscript itself is submitted
> through the conference system and is not included here.
>
> Anonymized for triple-blind review. Datasets (Amazon ESCI, Food-101) and model
> weights (CLIP/OpenCLIP/EVA-02/SigLIP, BLIP/BLIP-2, VLM captioners) are **public**
> and fetched on demand; they are **not** bundled. Absolute paths are replaced by
> the placeholders `${PROJECT_ROOT}` / `${CONDA_BASE}` / `${HOME}`.

## Layout

```
.
├── code/                    # core re-implementation
│   ├── cogeo.py             # CoGEO: product-title anchor + Sobel mask + differentiable JPEG + MI-FGSM PGD
│   ├── envsim.py            # DIM / SIM / TIM environment-simulation transforms
│   ├── diffjpeg_lite.py     # differentiable JPEG layer
│   ├── vlm_anchor.py        # VLM caption -> CLIP-text anchor (captioner-anchor ablation)
│   ├── dataset_*.py         # ESCI / Food-101 / SOP loaders
│   └── requirements.txt
├── n3/code/                 # evaluation + attack + analysis drivers
│   ├── eval_harness.py      # rank-lift scorer + the pre-flight no-op admissibility gate (gap_{E-I})
│   ├── n3_attack.py         # CoGEO driver        n3_attack_baselines.py  # PGD-bare / AdvCLIP / Co-Attack
│   ├── n3_attack_ae.py, n3_attack_ae_ens.py            # AE-CoGEO single-/multi-source consensus transfer probe
│   ├── n3_attack_diffprior.py, n3_attack_gfree.py, nes_attack.py, rank_eval_nes.py   # diffusion-prior + NES gradient-free
│   ├── n3_attack_cab.py     # cohort-adaptive-budget attacker
│   ├── eval_harness_blip.py / _blip2.py / _lateint.py  # BLIP / BLIP-2 / MaxSim cross-paradigm transfer
│   ├── purify_dir.py, gated_eps16.py                   # input purification + detector-gated purification
│   ├── ae_cogeo_detect.py, b1_detector_generalization.py   # training-free Laplacian detector + generalization
│   ├── n3_stats.py, n3_stats_v2.py, agg_r5_scaling.py, r4_aggregate.py, harvest_r2_compare.py   # statistics / aggregation
│   └── run_*.sh             # per-experiment launchers
├── figures/                 # figure-generation scripts (make_*.py) + input data (_data/) + rendered result plots
└── repro/
    ├── manifests/           # esci500_manifest.csv (491 imgs / 560 triples), esci1500_manifest.csv (1,430-pair scale-up)
    │                        #   columns: example_id, query_id, query, product_id (ASIN), product_title, esci_label, image_path
    └── INVENTORY.md         # environment / hardware snapshot (anonymized)
```

## Artifact items referenced in the paper

| Item | File |
|---|---|
| anchor⊥query / no-op admissibility verifier | `n3/code/eval_harness.py` (`gap_E_minus_I` pre-flight gate) |
| reference rank-lift scorer | `n3/code/eval_harness.py` |
| CoGEO attack | `code/cogeo.py` + `n3/code/n3_attack.py` |
| AE-CoGEO cross-encoder transfer probe | `n3/code/n3_attack_ae.py`, `n3_attack_ae_ens.py` |
| three baselines | `n3/code/n3_attack_baselines.py` (PGD-bare, AdvCLIP, Co-Attack) |
| ESCI manifest | `repro/manifests/esci500_manifest.csv`, `esci1500_manifest.csv` |
| reference result tables / figure data | `figures/_data/` |
| reproduction configuration (seeds, hyper-params, hardware) | `n3/code/run_*.sh` + `repro/INVENTORY.md` |

## Reproduce experiments (high level)

1. Create a conda env and install `code/requirements.txt`.
2. Set `PROJECT_ROOT` and fetch the public datasets/weights (ESCI & Food-101 via
   HuggingFace; CLIP-family + BLIP weights on demand). Images are addressed as
   `img/<ASIN>.jpg` relative to each manifest.
3. **Attack:** `n3/code/run_4way.sh` (white-box four-way),
   `n3/code/ae_ens_run.sh` (AE-CoGEO consensus transfer), etc.
4. **Score:** `n3/code/eval_harness.py` produces per-image rank-lift CSVs and the
   admissibility gate; `n3/code/n3_stats*.py` aggregates to the reported tables.

## Reproduce figures

`figures/make_*.py` regenerate every plot from `figures/_data/`. Plotted values are
hardcoded to match the measured experiment outputs so figures and tables cannot drift.

## Status

Private during peer review (triple-blind); to be opened at camera-ready.
