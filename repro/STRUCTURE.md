# CoGEO ICMR 2026 — remote workspace structure

> Working root: `${PROJECT_ROOT}/` on anon@<gpu-host>. Sister of (but separate from) the user's existing `~/anon/{Code,code2,auto,beans,exp1}` — those are unrelated to CoGEO and we leave them alone.

## Layout

```
${PROJECT_ROOT}/
├── code/                   # CoGEO re-implementation (git-init'd, paper-driven; see DECISIONS.md D-001)
│   ├── cogeo.py            # Algorithm 1 — VLM anchor + Sobel mask + MI-FGSM PGD
│   ├── envsim.py           # DIM / SIM / TIM / DiffJPEG
│   ├── vlm_anchor.py       # local Qwen3-VL-7B (fallback InternVL3-8B) caption + CLIP-text encode
│   ├── eval.py             # Δ / SSIM / LPIPS computation + per-image CSV
│   └── requirements.txt
├── data/
│   └── sop50/              # 50-image random subsample of Stanford Online Products (seed=42)
├── runs/                   # one folder per run; run_meta.json + log.txt + metrics.csv + summary.json
├── baselines/
│   └── local/
│       └── json/           # metric_contract.json (canonical) + metric_contract.draft.json (working)
├── paper/                  # later: anonymous PDF + tex + repro repo bundle
├── docs/                   # extra notes, e.g. ablation-mapping, ESCI integration spec
├── STRUCTURE.md            # this file
├── INVENTORY.md            # GPU pool / disk / model-weights inventory
└── README.md               # entry point
```

## Naming conventions

- run ids: `run-YYYYMMDDTHHMMSSZ-XXXX` (created by `run_with_gpu.sh`)
- model weights: pulled on demand into `$HF_HOME` (default `~/.cache/huggingface/`); deleted when no longer needed; never copied into this tree.
- per-run metrics: `runs/<run-id>/metrics.csv` (per-image rows) + `runs/<run-id>/summary.json` (aggregates).

## Hard constraints (DO NOT violate)

- **No commercial APIs** (OpenAI / Anthropic / Qwen-Max / Kimi). All VLM inference local.
- **No killing other users' GPU processes**. Use only cards confirmed idle by `run_with_gpu.sh`.
- **/data2 disk** is at 506 GB free, monitor; do not bulk-cache model weights you no longer need.
- **HF main blocked** — always `export HF_ENDPOINT=https://hf-mirror.com`.
