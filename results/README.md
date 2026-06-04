# results/ — reference result tables

Measured outputs grouped by the paper's structure. Each experiment folder holds its
`*_summary.json` (aggregate metrics). **Per-pair rank-lift CSVs** (`rank_*_per_pair.csv`,
one row per held-out (query, product) pair, columns
`example_id,query_id,query,product_id,product_title,esci_label,sim_orig,sim_adv,delta_held_out,rank_orig,rank_adv,rank_lift`)
are shipped for every group whose claim rests on a *paired* test, so reviewers can
recompute the Wilcoxon / bootstrap statistics directly:

- `white_box_4way/n3_eval/` — the headline 4-way (n=500)
- `five_family/L7_5way/` — five white-box families
- `scale_up/` — the 1,430-pair scale-up (`T3_scaleup` CoGEO/PGD-bare, `B2_full4method/eps16_1430` AdvCLIP/Co-Attack)
- `backbones/E4_food101_eps16/eval/` — the Food-101 cross-dataset 4-way
- `transfer/{ae_s2_ens2,ae_v2_ens,ae_v3_ens4,ae_transfer_s1}/` — AE-CoGEO consensus, source-count scaling, and the title-vs-consensus baseline
- `ablations/{n3_sanity,n3_sanity_L14}/no_op_per_pair.csv` — the no-op admissibility gate
- `ablations/gaussian_control/gaussian_control_summary.json` — the Gaussian-noise stealth control (manuscript **C3**)

`esci_full_aggregate.json` is the per-cohort aggregate that the composite figure consumes.

> **Recompute notes.** (1) A few `example_id`s repeat inside the per-pair CSVs (2 rows in
> the n=500 sets, 3 in the 1,430 set -- see `manifests/README.md`); pair files positionally
> (both are sorted identically) or `drop_duplicates` before a key-merge, otherwise an
> `example_id` join inflates n to 504/1448. (2) `i_label_top3_promotion_rate_*` uses ALL
> pairs as the denominator (baseline 10.2% = 51/500), not the 125-row Irrelevant cohort.

> The Gaussian-noise directional control (manuscript **C3**, SSIM 0.854 / ≈0 rank-lift)
> is a pixel-domain run outside the retrieval harness: script
> `src/analysis/gaussian_control.py`, output
> `ablations/gaussian_control/gaussian_control_summary.json` (mean SSIM 0.8538 vs CoGEO 0.66 /
> PGD-bare 0.76 / AdvCLIP 0.78 / Co-Attack 0.76). The SSIM is deterministic
> (`sigma=eps/2.58`, seed 20260512) over the 491-image manifest at 224×224; the ≈0 rank-lift
> is the analytical high-dimensional-symmetry bound stated in the manuscript, not a measured run.

| Folder | Experiments | Paper location |
|---|---|---|
| `white_box_4way/` | the headline 4-way white-box comparison (`n3_eval` = pooled per-pair; `n3_{cogeo,pgd_bare,advclip,coattack}_eps16` = per-method runs; `n3_gfree_eps16` = gradient-free) | Table (main 4-way), abstract/intro **C1**, per-cohort **C2** |
| `backbones/` | per-retriever cells: `E1` LAION-2B, `E5` EVA-02, `F_{vitb32,siglip,vith14}` (6-backbone); `E2_*` ε-sweep; `E3` 800px resolution; `E4` Food-101 | `fig:results` (d) backbones, (b) ε-sweep, (f) Food-101; resolution axis in §summary |
| `transfer/` | AE-CoGEO consensus (`ae_v2_ens`, `ae_v3_ens4`, `ae_s2_ens2`), single-encoder arm (`ae_transfer_s1`, `T1_transfer`), source-count scaling, cross-paradigm transfer (`R2_blip_rerank`, `R2_blip2_rerank`, `R2_blip_rerank_ae`, `L2_lateint_rerank`) | `tab:transfer`, `fig:results` (a) scaling, (c) held-out; `fig:bounded` (a) |
| `scale_up/` | 1,430-pair scale-up (`T3_scaleup`) and the four-method 1,430 set (`B2_full4method`) | `fig:bounded` (c), §summary |
| `defense/` | input purification (`R4_purify`), detector-gated purification (`R5_gated`, `R5_gated_eps16`), matched adaptive attacker (`P0_adaptive`), cohort-adaptive-budget (`B4_cab`) | `fig:bounded` (b), §summary |
| `five_family/` | five white-box families incl. diffusion-prior — per-pair rank-lift CSVs (`L7_5way/rank_*_per_pair.csv`) and `L7_diffprior_eps16` | `tab:statsextra`, `fig:distribution` |
| `ablations/` | anchor×backbone robustness (`D_vlm`, 5 captioners × 6 backbones), gradient-free NES ceiling-check (`nes_ceiling`), no-op admissibility gate (`n3_sanity` ViT-B/32 gap 0.0405 fails, `n3_sanity_L14` ViT-L/14 gap 0.0533 passes), Gaussian-noise stealth control (`gaussian_control`, mean SSIM 0.854) | `fig:anchorbb`, limitations (NES), method §no-op gate, §summary **C3** |

All values trace to these files; `figures/make_*.py` read them to regenerate every plot.
