"""diffjpeg_lite — minimal differentiable JPEG (Q=85 by default) vendored for CoGEO.

Reference: standard JPEG luminance/chrominance Q-tables; this is a compact reimplementation
of mlomnitz/DiffJPEG's core path, simplified for our env-sim use case.

Pipeline: RGB(float [0,1], B,3,H,W) -> YCbCr -> 8x8 DCT -> quantize -> dequantize -> IDCT -> RGB
Quantization is differentiable via straight-through estimator.

Outputs are float images, not bytes — we feed them into CLIP, not write JPEG files.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


_Y_TABLE = torch.tensor(
    [[16, 11, 10, 16, 24, 40, 51, 61],
     [12, 12, 14, 19, 26, 58, 60, 55],
     [14, 13, 16, 24, 40, 57, 69, 56],
     [14, 17, 22, 29, 51, 87, 80, 62],
     [18, 22, 37, 56, 68, 109, 103, 77],
     [24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101],
     [72, 92, 95, 98, 112, 100, 103, 99]], dtype=torch.float32)

_C_TABLE = torch.tensor(
    [[17, 18, 24, 47, 99, 99, 99, 99],
     [18, 21, 26, 66, 99, 99, 99, 99],
     [24, 26, 56, 99, 99, 99, 99, 99],
     [47, 66, 99, 99, 99, 99, 99, 99],
     [99, 99, 99, 99, 99, 99, 99, 99],
     [99, 99, 99, 99, 99, 99, 99, 99],
     [99, 99, 99, 99, 99, 99, 99, 99],
     [99, 99, 99, 99, 99, 99, 99, 99]], dtype=torch.float32)


def _quality_to_factor(q: float) -> float:
    if q < 50:
        return 5000.0 / max(q, 1.0)
    return 200.0 - 2.0 * q


def _build_dct_matrix(N: int = 8) -> torch.Tensor:
    M = torch.empty(N, N, dtype=torch.float32)
    for k in range(N):
        for n in range(N):
            ck = math.sqrt(1.0 / N) if k == 0 else math.sqrt(2.0 / N)
            M[k, n] = ck * math.cos(math.pi * (2 * n + 1) * k / (2 * N))
    return M


_DCT8 = _build_dct_matrix(8)


def _block_dct(x: torch.Tensor) -> torch.Tensor:
    """x: (B, C, H, W), H,W % 8 == 0; returns same shape, blockwise DCT."""
    B, C, H, W = x.shape
    M = _DCT8.to(x.device, x.dtype)
    x = x.unfold(2, 8, 8).unfold(3, 8, 8)  # (B,C,H/8,W/8,8,8)
    out = M @ x @ M.t()
    return out.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, H, W)


def _block_idct(x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    M = _DCT8.to(x.device, x.dtype)
    x = x.unfold(2, 8, 8).unfold(3, 8, 8)
    out = M.t() @ x @ M
    return out.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, H, W)


def _round_ste(x: torch.Tensor) -> torch.Tensor:
    """differentiable round via straight-through estimator."""
    return x + (torch.round(x) - x).detach()


def _rgb_to_ycbcr(rgb: torch.Tensor) -> torch.Tensor:
    """rgb: (B,3,H,W) in [0,1] -> ycbcr in [0,1] (Y), [-0.5,0.5] (Cb,Cr)."""
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b
    return torch.cat([y, cb, cr], dim=1)


def _ycbcr_to_rgb(ycbcr: torch.Tensor) -> torch.Tensor:
    y, cb, cr = ycbcr[:, 0:1], ycbcr[:, 1:2], ycbcr[:, 2:3]
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return torch.cat([r, g, b], dim=1)


class DiffJPEG(nn.Module):
    """Differentiable JPEG-Q forward pass.

    Implementation choices:
    - keeps the same H,W (no chroma subsampling); paper's contract only requires
      "JPEG compression at Q=85" robustness, not byte-exact JPEG bitstream.
    - quantization uses straight-through estimator so gradients survive.
    - 0..1 RGB clamp at output; pre-clamp at input.
    """

    def __init__(self, quality: int = 85):
        super().__init__()
        factor = _quality_to_factor(float(quality))
        y_tab = torch.clamp(torch.floor((_Y_TABLE * factor + 50.0) / 100.0), min=1.0)
        c_tab = torch.clamp(torch.floor((_C_TABLE * factor + 50.0) / 100.0), min=1.0)
        self.register_buffer("y_q", y_tab.repeat(1, 1, 1, 1))   # (1,1,8,8)
        self.register_buffer("c_q", c_tab.repeat(1, 1, 1, 1))

    def _quantize_channel(self, x: torch.Tensor, qtab: torch.Tensor) -> torch.Tensor:
        # x: (B,1,H,W), centered around 0 (subtract 128 in Y, already 0-centered for Cb,Cr)
        coeffs = _block_dct(x)
        B, C, H, W = coeffs.shape
        qmap = qtab.repeat(1, 1, H // 8, W // 8)
        q = _round_ste(coeffs / qmap) * qmap
        return _block_idct(q)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        rgb = rgb.clamp(0.0, 1.0)
        ycc = _rgb_to_ycbcr(rgb)
        # Y in [0,1], shift to [-128,127]/256 to match JPEG's Y-128
        y = (ycc[:, 0:1] * 255.0) - 128.0
        cb = ycc[:, 1:2] * 255.0
        cr = ycc[:, 2:3] * 255.0

        # block-pad to multiple of 8 (training images are 800x800 -> already multiple of 8 if 800%8=0; yes)
        H, W = y.shape[-2:]
        ph = (8 - H % 8) % 8
        pw = (8 - W % 8) % 8
        if ph or pw:
            y = F.pad(y, (0, pw, 0, ph), mode="replicate")
            cb = F.pad(cb, (0, pw, 0, ph), mode="replicate")
            cr = F.pad(cr, (0, pw, 0, ph), mode="replicate")

        y2 = self._quantize_channel(y, self.y_q)
        cb2 = self._quantize_channel(cb, self.c_q)
        cr2 = self._quantize_channel(cr, self.c_q)

        if ph or pw:
            y2 = y2[..., :H, :W]
            cb2 = cb2[..., :H, :W]
            cr2 = cr2[..., :H, :W]

        ycc2 = torch.cat([(y2 + 128.0) / 255.0, cb2 / 255.0, cr2 / 255.0], dim=1)
        rgb2 = _ycbcr_to_rgb(ycc2)
        return rgb2.clamp(0.0, 1.0)
