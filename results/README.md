# results/ — reference result tables

Measured outputs grouped by the paper's structure. Each experiment folder holds its
`*_summary.json` (aggregate metrics); per-pair rank-lift CSVs live under
`five_family/L7_5way/`. `esci_full_aggregate.json` is the per-cohort aggregate that
the composite figure consumes.

| Folder | Experiments | Paper location |
|---|---|---|
| `white_box_4way/` | the headline 4-way white-box comparison (`n3_eval` = pooled per-pair; `n3_{cogeo,pgd_bare,advclip,coattack}_eps16` = per-method runs; `n3_gfree_eps16` = gradient-free) | Table (main 4-way), abstract/intro **C1**, per-cohort **C2** |
| `backbones/` | per-retriever cells: `E1` LAION-2B, `E5` EVA-02, `F_{vitb32,siglip,vith14}` (6-backbone); `E2_*` ε-sweep; `E3` 800px resolution; `E4` Food-101 | `fig:results` (d) backbones, (b) ε-sweep, (f) Food-101; resolution axis in §summary |
| `transfer/` | AE-CoGEO consensus (`ae_v2_ens`, `ae_v3_ens4`, `ae_s2_ens2`), single-encoder arm (`ae_transfer_s1`, `T1_transfer`), source-count scaling, cross-paradigm transfer (`R2_blip_rerank`, `R2_blip2_rerank`, `R2_blip_rerank_ae`, `L2_lateint_rerank`) | `tab:transfer`, `fig:results` (a) scaling, (c) held-out; `fig:bounded` (a) |
| `scale_up/` | 1,430-pair scale-up (`T3_scaleup`) and the four-method 1,430 set (`B2_full4method`) | `fig:bounded` (c), §summary |
| `defense/` | input purification (`R4_purify`), detector-gated purification (`R5_gated`, `R5_gated_eps16`), matched adaptive attacker (`P0_adaptive`), cohort-adaptive-budget (`B4_cab`) | `fig:bounded` (b), §summary |
| `five_family/` | five white-box families incl. diffusion-prior — per-pair rank-lift CSVs (`L7_5way/rank_*_per_pair.csv`) and `L7_diffprior_eps16` | `tab:statsextra`, `fig:distribution` |
| `ablations/` | anchor×backbone robustness (`D_vlm`, 5 captioners × 6 backbones), gradient-free NES ceiling-check (`nes_ceiling`), no-op admissibility gate + Gaussian control (`n3_sanity`, `n3_sanity_L14`) | `fig:anchorbb`, limitations (NES), method §no-op gate |

All values trace to these files; `figures/make_*.py` read them to regenerate every plot.
