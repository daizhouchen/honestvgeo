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
├── requirements.txt     # Python dependencies
├── ENVIRONMENT.md       # software/hardware the experiments ran under
├── src/                 # all reproduction code (flat, self-contained)
│   ├── cogeo / attacks      n3_attack.py (CoGEO), n3_attack_baselines.py
│   │                        (PGD-bare/AdvCLIP/Co-Attack), n3_attack_ae.py +
│   │                        n3_attack_ae_ens.py (AE-CoGEO transfer probe),
│   │                        n3_attack_diffprior.py, n3_attack_gfree.py +
│   │                        nes_attack.py (gradient-free), n3_attack_cab.py
│   ├── method pieces        envsim.py, diffjpeg_lite.py, n3_vlm_caption.py
│   ├── evaluation           eval_harness.py  (rank-lift scorer + the pre-flight
│   │                        no-op admissibility gate, gap_{E-I}),
│   │                        eval_harness_blip.py / _blip2.py / _lateint.py,
│   │                        rank_eval_nes.py
│   ├── defense              purify_dir.py, gated_eps16.py, ae_cogeo_detect.py,
│   │                        b1_detector_generalization.py
│   ├── data loaders         esci_loader.py, food101_loader.py
│   ├── stats / aggregation  n3_stats.py, n3_stats_v2.py, agg_r5_scaling.py,
│   │                        r4_aggregate.py, harvest_r2_compare.py
│   └── run_*.sh             per-experiment launchers
├── manifests/           # esci500_manifest.csv (491 imgs / 560 triples),
│                        #   esci1500_manifest.csv (1,430-pair scale-up)
│                        #   cols: example_id, query_id, query, product_id (ASIN),
│                        #   product_title, esci_label, image_path (img/<ASIN>.jpg)
├── results/             # reference result tables: one dir per experiment with its
│                        #   summary JSONs; L7_5way/ holds per-pair rank-lift CSVs;
│                        #   esci_full_aggregate.json is the per-cohort aggregate
└── figures/             # make_*.py — regenerate every plot from ../results/
```

## Artifact items referenced in the paper

| Item | Here |
|---|---|
| anchor⊥query / no-op admissibility verifier | `src/eval_harness.py` (`gap_E_minus_I` pre-flight gate) |
| reference rank-lift scorer | `src/eval_harness.py` |
| CoGEO attack | `src/n3_attack.py` |
| AE-CoGEO cross-encoder transfer probe | `src/n3_attack_ae.py`, `src/n3_attack_ae_ens.py` |
| three baselines | `src/n3_attack_baselines.py` |
| ESCI manifest | `manifests/` |
| reference result tables | `results/` |
| reproduction configuration (seeds, hyper-params, hardware) | `src/run_*.sh` + `ENVIRONMENT.md` |

## Reproduce

1. `pip install -r requirements.txt`; set `PROJECT_ROOT`.
2. Fetch the public datasets/weights (ESCI & Food-101 via HuggingFace; CLIP-family
   + BLIP weights on demand). Images are addressed `img/<ASIN>.jpg` per manifest.
3. **Attack:** `src/run_4way.sh` (white-box four-way), `src/ae_ens_run.sh`
   (AE-CoGEO consensus transfer), etc. Outputs land under `runs/`.
4. **Score:** `src/eval_harness.py` emits per-image rank-lift CSVs and the
   admissibility gate; `src/n3_stats*.py` aggregates to the reported tables.
5. **Figures:** `python figures/make_*.py` regenerate every plot from `results/`.

The bundled `results/` are the reference outputs; rerunning the pipeline writes
fresh outputs under `runs/`. Plotted values are hardcoded to match the measured
outputs so figures and tables cannot drift.

## Status

Private during peer review (triple-blind); to be opened at camera-ready.
