#!/usr/bin/env bash
# Resilient, GPU-free download of the BLIP-2 ITM retriever for R2 second-paradigm
# closure. No CUDA; just populates the HF cache so a later GPU run loads instantly.
# Run under nohup so it survives ssh disconnect.
set -uo pipefail
source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-honestvgeo}
export HF_ENDPOINT=https://hf-mirror.com TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
if python -c "import hf_transfer" 2>/dev/null; then
  export HF_HUB_ENABLE_HF_TRANSFER=1
  echo "HF_TRANSFER=on"
else
  echo "HF_TRANSFER=off"
fi
cd ${PROJECT_ROOT}
echo "DL_BEGIN $(date -u +%FT%TZ)"
python -u -c '
import time
from transformers import AutoProcessor, Blip2ForImageTextRetrieval
m = "Salesforce/blip2-itm-vit-g"
t0 = time.time()
print("DL_START", flush=True)
AutoProcessor.from_pretrained(m)
print("PROC_OK %.1fs" % (time.time() - t0), flush=True)
Blip2ForImageTextRetrieval.from_pretrained(m)
print("DOWNLOAD_DONE %.1fs" % (time.time() - t0), flush=True)
'
echo "DL_EXIT=$? $(date -u +%FT%TZ)"
