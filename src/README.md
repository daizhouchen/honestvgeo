# src/ — reproduction code

All code, grouped by role. Files are organized into subpackages, but each script is
runnable on its own; the only two intra-repo import edges
(`method/envsim.py → method/diffjpeg_lite.py`, `attacks/n3_attack_ae_ens.py →
attacks/n3_attack_ae.py`) stay inside a single subdirectory, so the layout does not
break imports.

Pipeline at a glance: **attack** (`attacks/`, using `method/` building blocks) writes
adversarial images → **score** (`eval/`) computes per-pair rank-lift CSVs and runs the
admissibility gate → **aggregate** (`analysis/`) turns those CSVs into the reported
tables. `defense/` re-scores under detectors/purification; `data/` provides the loaders;
`scripts/` are one-command launchers for each experiment.

## `method/` — CoGEO building blocks
| File | Purpose |
|---|---|
| `envsim.py` | Environment simulation: DIM / SIM / TIM / differentiable-JPEG transforms, each applied stochastically (p=0.5) during optimization |
| `diffjpeg_lite.py` | Minimal differentiable JPEG layer (Q=85 default) used by `envsim` |
| `n3_vlm_caption.py` | Generate per-image VLM captions for the manifest → `{product_id: caption}` JSON (used for the captioner-anchor ablation) |

## `attacks/` — perturbation generators (shared output contract)
| File | Purpose |
|---|---|
| `n3_attack.py` | **CoGEO** — per-ASIN adversarial image `x_p*` (product-title anchor + Sobel mask + differentiable JPEG + MI-FGSM PGD) |
| `n3_attack_baselines.py` | CLIP-targeted **baselines** (PGD-bare / AdvCLIP / Co-Attack), same output contract |
| `n3_attack_ae.py` | **AE-CoGEO** anchor-ensemble variant |
| `n3_attack_ae_ens.py` | **AE-CoGEO** encoder-ensemble transfer probe (optimizes one perturbation against a multi-encoder consensus) |
| `n3_attack_diffprior.py` | Fifth family: diffusion-prior (DiffPGD) query-free attack |
| `n3_attack_gfree.py` / `nes_attack.py` | Gradient-free **NES** black-box attack (white-box-advantage ceiling check) |
| `n3_attack_cab.py` | Cohort-Adaptive-Budget attacker (tests whether per-cohort heterogeneity is exploitable) |

## `verify/` — standalone HonestEval verifiers (no model)
| File | Purpose |
|---|---|
| `anchor_query_verifier.py` | Certifies the **anchor⊥query invariant**: holds each product row fixed, substitutes a foreign eval query, and asserts the anchor is unchanged (and never echoes the query, and varies across products). Run before reporting rank-lift: `python verify/anchor_query_verifier.py --manifest ../manifests/esci500_manifest.csv` |
| `admissibility_check.py` | **No-model** admissibility: L∞-budget compliance (`max\|adv−orig\|/255 ≤ ε`), image-integrity (same dimensions), and four-cohort coverage, from an `img/<ASIN>.jpg` layout + manifest (no retriever/VLM loaded) |

## `eval/` — scoring + the protocol gate
| File | Purpose |
|---|---|
| `eval_harness.py` | The core harness: **rank-lift scorer** (`--mode rank`, per-pair CSVs) **and the pre-flight no-op admissibility gate** (`gap_{E-I}`) |
| `eval_harness_blip.py` / `eval_harness_blip2.py` | Re-score the same adversarial images with a BLIP fused-encoder / BLIP-2 Q-Former retriever (cross-paradigm transfer) |
| `eval_harness_lateint.py` | Re-score with a late-interaction (ColBERT/MaxSim-style) retriever |
| `rank_eval_nes.py` | Standalone rank evaluator with identical protocol to `eval_harness.py --mode rank` |

## `defense/` — detection + mitigation
| File | Purpose |
|---|---|
| `purify_dir.py` | Input-purification defense: apply a benign transform (e.g. JPEG) to a directory of images |
| `ae_cogeo_detect.py` | Detectability of the AE-CoGEO transfer perturbation with the training-free Laplacian flat-region detector |
| `b1_detector_generalization.py` | Detector generalization across attack families and to a held-out domain |

## `data/` — loaders
| File | Purpose |
|---|---|
| `esci_loader.py` | ESCI parquet subset loader for the real-relevance evaluation |
| `food101_loader.py` | Food-101 manifest builder emitting the **same** schema as `esci_loader.py`, so the harness/attacks run unchanged |

## `analysis/` — statistics → reported tables
| File | Purpose |
|---|---|
| `n3_stats.py` | Paired Wilcoxon + bootstrap CI95 + SSIM (primary hypothesis) |
| `n3_stats_v2.py` | 4-way method comparison statistics (CoGEO / PGD-bare / AdvCLIP / Co-Attack) |
| `agg_r5_scaling.py` | AE-CoGEO source-encoder count scaling-law aggregation (n ∈ {1,2,3,4}) |
| `r4_aggregate.py` | Aggregate purification-defense CSVs into a per-cohort/per-encoder defended-vs-undefended table |
| `harvest_r2_compare.py` | Build the CLIP-vs-BLIP cross-paradigm comparison from saved per-pair CSVs |

## `scripts/` — one-command launchers
Each script self-resolves the code root
(`SRC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"`) and calls the modules
above, so it can be run from anywhere. Set `PROJECT_ROOT` (and fetch public
datasets/weights) first; outputs are written under `runs/`.

| Script | Runs |
|---|---|
| `run_4way.sh` | White-box four-way comparison (CoGEO + 3 baselines) → `results/white_box_4way` |
| `run_pair.sh` | ε-sweep helper (CoGEO + PGD-bare for one ε / backbone) |
| `run_d_matrix.sh`, `run_d_vlm.sh` | Anchor × backbone ablation (5 captioners × 6 retrievers) → `results/ablations/D_vlm` |
| `ae_run.sh`, `ae_ens_run.sh`, `ae_transfer_s1.sh`, `run_r5_s2.sh` | AE-CoGEO consensus transfer + source-count scaling |
| `run_c_transfer.sh` | Held-out-encoder transfer |
| `run_r2_blip.sh`, `run_r2_blip2.sh`, `run_r2_ae.sh`, `run_l2_lateint.sh` | Cross-paradigm reranking (BLIP / BLIP-2 / MaxSim) |
| `r4_defense_run.sh`, `p0_adaptive.sh` | Purification / detector-gated / adaptive-attacker defense |
| `l7_eval.sh` | Five-family (incl. diffusion-prior) per-pair evaluation |
| `launch_cell.sh` | Start one experiment as a detached background job |
| `download_food101.sh`, `r2_blip2_download.sh` | Fetch the public Food-101 dataset / BLIP-2 weights |

> Naming note: the `n3_` prefix on several files is an internal experiment-series tag,
> not an identifier; it is wired through imports and result-folder names and is kept to
> preserve a self-consistent, runnable graph.
