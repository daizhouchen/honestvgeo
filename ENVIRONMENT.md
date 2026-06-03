# Reproduction environment

Software and hardware the experiments were run under.

## Software

- Python 3.14, PyTorch 2.9.1 (CUDA 12.8), `transformers` 4.57.5
- Full Python dependencies: see `requirements.txt`
- HuggingFace weights/datasets are fetched on demand. If the HF main endpoint is
  blocked in your environment, set `HF_ENDPOINT=https://hf-mirror.com`.

## Hardware

- Single NVIDIA RTX 4090 (24 GB) is sufficient for every experiment.
- Approximate first-run weight footprint ~17 GB (CLIP ViT-B/32 ~150 MB; a
  VLM captioner such as Qwen2.5-VL-7B / InternVL3-8B ~16 GB for the
  captioner-anchor ablation). CLIP-family / BLIP / SigLIP / EVA-02 weights are
  pulled per experiment as needed.

## Notes

- All paths in the scripts use the placeholders `${PROJECT_ROOT}` (this repo's
  root), `${CONDA_BASE}`, `${HOME}` — set them for your machine.
- Datasets (Amazon ESCI, Food-101) are public; only their manifests
  (`manifests/`) are bundled. Images are addressed as `img/<ASIN>.jpg`.
