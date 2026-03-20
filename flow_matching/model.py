"""
DiT (Diffusion Transformer) vector field network for Rectified Flow.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn

from flow_matching.config import FlowConfig


# ── Embedding helpers ─────────────────────────────────────────────────────────

def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal time embedding.

    t:   (B,) float in [0, 1]
    dim: embedding dimension (must be even)
    Returns: (B, dim)
    """
    assert dim % 2 == 0, f"dim must be even, got {dim}"
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / max(half - 1, 1)
    )
    args = t[:, None].float() * freqs[None]   # (B, half)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, dim)


def make_2d_sincos_pos_embed(d_model: int, grid_size: int = 8) -> torch.Tensor:
    """Fixed 2D sinusoidal positional embeddings.

    d_model must be divisible by 4 (half per axis, each axis uses sin+cos).
    Returns: (1, grid_size^2, d_model) — intended to register as a buffer.
    """
    assert d_model % 4 == 0, f"d_model must be divisible by 4, got {d_model}"
    half = d_model // 2  # half the dims for each spatial axis

    omega = 1.0 / (
        10000 ** (torch.arange(half // 2).float() / max(half // 2 - 1, 1))
    )

    grid = torch.arange(grid_size).float()
    emb = torch.outer(grid, omega)                              # (grid_size, half//2)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (grid_size, half)

    # Expand for (h, w) pairs: each token gets [h_emb || w_emb]
    emb_h = emb.unsqueeze(1).expand(-1, grid_size, -1)  # (grid_size, grid_size, half)
    emb_w = emb.unsqueeze(0).expand(grid_size, -1, -1)  # (grid_size, grid_size, half)
    pos = torch.cat([emb_h, emb_w], dim=-1)             # (grid_size, grid_size, d_model)
    pos = pos.reshape(grid_size * grid_size, d_model)    # (N, d_model)
    return pos.unsqueeze(0)                              # (1, N, d_model)
