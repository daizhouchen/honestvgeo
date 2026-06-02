"""vlm_anchor — Phase 1 of CoGEO: generate a product description with a local VLM, then
encode that description into the CLIP text-embedding space as v_anchor.

Paper used Qwen3.5-Plus API; we use Qwen/Qwen2.5-VL-7B-Instruct locally (D-002 / D-010).
Fallback chain: Qwen2.5-VL-3B-Instruct -> InternVL3-8B -> Qwen3-VL-4B-Instruct.
"""
from __future__ import annotations

import os
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    CLIPModel,
    CLIPProcessor,
)


# Faithful reconstruction of paper §4.1 prompt; not given verbatim in paper.
_DEFAULT_PROMPT = (
    "You are a professional e-commerce product description writer. Write a concise "
    "70-120 word English description of the product in this image, focused on:\n"
    "1. Product category and main appearance (color, material, form).\n"
    "2. Key selling points and quality indicators (e.g., high-quality, professional-grade, well-made).\n"
    "3. Suitable scenarios or target users.\n"
    "Do not invent details not visible in the image. Output a single paragraph, no bullet points."
)


class VLMAnchor:
    def __init__(self,
                 vlm_model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 clip_model_id: str = "openai/clip-vit-base-patch32",
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float16,
                 max_new_tokens: int = 180):
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.prompt = _DEFAULT_PROMPT

        # ---- VLM ----
        self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            vlm_model_id, torch_dtype=dtype, device_map=device,
        ).eval()
        self.vlm_processor = AutoProcessor.from_pretrained(vlm_model_id)

        # ---- CLIP text encoder for v_anchor ----
        self.clip = CLIPModel.from_pretrained(clip_model_id).eval().to(device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_id)

    @torch.no_grad()
    def caption(self, pil_image) -> str:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": self.prompt},
            ],
        }]
        chat = self.vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.vlm_processor(text=[chat], images=[pil_image], return_tensors="pt").to(self.device)
        out = self.vlm.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1]:]
        text = self.vlm_processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
        return text

    @torch.no_grad()
    def encode_anchor_text(self, text: str) -> torch.Tensor:
        # CLIP text encoder is limited to 77 tokens; truncate.
        toks = self.clip_processor.tokenizer(text, return_tensors="pt", truncation=True,
                                             max_length=77, padding=True)
        toks = {k: v.to(self.device) for k, v in toks.items()}
        z = self.clip.get_text_features(**toks)
        return F.normalize(z, dim=-1)
