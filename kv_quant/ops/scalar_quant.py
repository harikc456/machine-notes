from __future__ import annotations

import torch


def quantize(h: torch.Tensor, n_bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token n-bit scalar quantization.

    h: (..., d) float
    Returns:
      h_int: (..., d) int8  — quantized values in [0, 2^n_bits - 1]
      scale: (..., 1) float16 — per-token abs-max scale
    n_bits must be in [1, 7].
    """
    n_levels = 2 ** n_bits  # number of quantization levels
    scale = h.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    h_norm = (h / scale).clamp(-1.0, 1.0)
    # Map [-1, 1] -> [0, n_levels - 1]
    h_int = ((h_norm + 1.0) / 2.0 * (n_levels - 1)).round().clamp(0, n_levels - 1)
    return h_int.to(torch.int8), scale.to(torch.float16)


def dequantize(h_int: torch.Tensor, scale: torch.Tensor, n_bits: int) -> torch.Tensor:
    """Inverse of quantize.

    h_int: (..., d) int8
    scale: (..., 1) float16
    Returns: (..., d) float32
    """
    n_levels = 2 ** n_bits
    h_norm = h_int.float() / (n_levels - 1) * 2.0 - 1.0  # back to [-1, 1]
    return h_norm * scale.float()
