# CoGEO ICMR 2026 — remote inventory snapshot

> Captured at first remote bootstrap. Re-run `bash ~/anon/scripts/refresh_inventory.sh` to refresh.

## Host

- hostname: <gpu-host>
- kernel: 5.15.0-177-generic
- conda env: `anon` (Python 3.14.2, torch 2.9.1+cu128, CUDA 12.8, transformers 4.57.5)
- HF_ENDPOINT: `https://hf-mirror.com` (HF main is blocked)
- working root: `${PROJECT_ROOT}`
- runner script: `${PROJECT_ROOT}/anon/scripts/run_with_gpu.sh`

## Disk (/data2)

```
/dev/nvme2n1    7.0T  6.1T  506G  93% /data2
```

**Warning**: stay above the 500 GB free floor; purge unused HF cache aggressively.

## GPU pool (live, captured at bootstrap)

| idx | name | total | free | used | util |
|---|---|---|---|---|---|
| 0| NVIDIA GeForce RTX 4090| 49140 MiB| 20084 MiB| 28455 MiB| 100 % |
| 1| NVIDIA GeForce RTX 4090| 49140 MiB| 31887 MiB| 16652 MiB| 76 % |
| 2| NVIDIA GeForce RTX 4090| 49140 MiB| 48535 MiB| 4 MiB| 0 % |
| 3| NVIDIA GeForce RTX 4090| 49140 MiB| 48511 MiB| 28 MiB| 0 % |
| 4| NVIDIA GeForce RTX 4090| 49140 MiB| 25432 MiB| 23107 MiB| 0 % |
| 5| NVIDIA GeForce RTX 4090| 49140 MiB| 33129 MiB| 15410 MiB| 78 % |
| 6| NVIDIA GeForce RTX 4090| 49140 MiB| 33129 MiB| 15410 MiB| 70 % |

### Pool decision (DECISIONS.md D-008)

- **claim**: GPU 2 or GPU 3 (both ≥48 GB free, 0% util at audit).
- **avoid**: GPU 4 (had 23 GB residual memory at audit; util 0% but memory pinned by another user).
- **never touch**: GPU 0/1/5/6 (busy at audit; respect shared discipline — do NOT kill other people's processes).

## Model-weight policy

- Pull on demand into `$HF_HOME` (default `~/.cache/huggingface/`).
- Delete after use if not reused within the same day.
- Required for first SOP-50 reproduction:
  - `openai/clip-vit-base-patch32` (~150 MB)
  - `Qwen/Qwen3-VL-7B-Instruct` or equivalent (~16 GB)  — fallback `OpenGVLab/InternVL3-8B` (~17 GB)
- Approx total first-pull footprint: ~17 GB. Comfortably below 500 GB budget.
