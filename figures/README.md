# figures/ — plot generation

Each `make_*.py` regenerates a manuscript figure from the measured data in
`../results/`. Run any of them after the experiments (or directly, since the inputs
are bundled): `python figures/make_<name>.py`. Plotted values are hardcoded to the
measured outputs so figures and tables cannot silently drift.

| Script | Produces | Reads | Paper figure |
|---|---|---|---|
| `make_results_composite.py` | `fig_results_main` (full-width 2×3 composite) | `results/backbones/*/cell_summary.json`, `results/esci_full_aggregate.json` | main results figure (panels a–f) |
| `make_anchor_heatmap.py` | `fig_anchor_heatmap` | hardcoded from `results/ablations/D_vlm` | anchor × backbone robustness (5 captioners × 6 retrievers) |
| `make_panels_v2.py` | `fig_d_heatmap` (+ intermediate panels) | `results/esci_full_aggregate.json`, `results/backbones/*/cell_summary.json` | per-cohort / d-matrix panels |
| `make_bounded_defense.py` | `fig_bounded_defense` | hardcoded from `results/transfer`, `results/defense`, `results/scale_up` | bounded-under-stress (transfer / defense / scale-up) |
| `make_capacity_scaling.py` | `fig_capacity_scaling` | hardcoded from the cross-backbone table | retriever-capacity scaling law |
| `make_distribution_figure.py` | `fig_distribution` | `results/five_family/L7_5way/rank_*_per_pair.csv` | per-pair rank-lift distribution (five families) |
| `make_n6_figs.py` | `fig_eps_sweep_curve`, `fig_food101_bars` | `results/backbones/*/cell_summary.json` | ε-sweep + Food-101 panels |

Rendered images are **not** checked in (they are regenerable outputs); run the scripts
to produce the `.pdf`/`.png`. The framework schematic (`fig_framework` in the paper)
has no generator here — it is a hand-drawn diagram that lives only in the manuscript.
