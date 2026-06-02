"""envsim — Environment simulation for CoGEO Phase 3.

Implements DIM, SIM, TIM, DiffJPEG with stochastic apply-probability=0.5 each,
matching paper §4.3.
"""
from __future__ import annotations

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffjpeg_lite import DiffJPEG


def _dim_resize(x: torch.Tensor, lo: float = 0.9, hi: float = 1.1) -> torch.Tensor:
    """DIM: random resize within [lo*s, hi*s] (paper: 0.9..1.1, p=0.5)."""
    B, C, H, W = x.shape
    s = random.uniform(lo, hi)
    new_h = max(8, int(round(H * s) // 8 * 8))
    new_w = max(8, int(round(W * s) // 8 * 8))
    if (new_h, new_w) == (H, W):
        return x
    y = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=True)
    # bring back to (H,W) so downstream CLIP-resize stays consistent
    return F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False, antialias=True)


def _gaussian_kernel(k: int, sigma: float, device, dtype) -> torch.Tensor:
    half = (k - 1) / 2.0
    xs = torch.arange(k, device=device, dtype=dtype) - half
    g = torch.exp(-(xs ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = torch.outer(g, g)  # (k,k)
    return kernel


def _tim_smooth(x: torch.Tensor, k: int = 7, sigma: float = 1.5) -> torch.Tensor:
    """TIM: smooth gradients spatially with a 2D Gaussian (here applied to the IMAGE during forward;
    paper applies to gradients but conv-on-image then back-prop to delta is equivalent under
    linearity assumptions of the loss-of-cos-sim; common simplification)."""
    B, C, H, W = x.shape
    g = _gaussian_kernel(k, sigma, x.device, x.dtype)
    weight = g.expand(C, 1, k, k).contiguous()
    pad = k // 2
    return F.conv2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), weight, groups=C)


class EnvSim(nn.Module):
    """Stochastic environment simulation: DIM + Resize + DiffJPEG. SIM is handled in cogeo.py
    as multi-scale gradient averaging (it's a gradient-side trick, not a forward perturbation).

    Each transform applies independently with probability `p`.
    """

    def __init__(self, p: float = 0.5, jpeg_quality: int = 85, tim_k: int = 7,
                 dim_lo: float = 0.9, dim_hi: float = 1.1):
        super().__init__()
        self.p = float(p)
        self.dim_lo = dim_lo
        self.dim_hi = dim_hi
        self.tim_k = tim_k
        self.jpeg = DiffJPEG(quality=jpeg_quality)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            x = _dim_resize(x, self.dim_lo, self.dim_hi)
        if random.random() < self.p:
            x = _tim_smooth(x, k=self.tim_k)
        if random.random() < self.p:
            x = self.jpeg(x)
        return x.clamp(0.0, 1.0)
