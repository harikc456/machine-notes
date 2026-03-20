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


# ── PatchEmbed ────────────────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    """Split image into patches and project to d_model.

    Input:  (B, 3, H, W)
    Output: (B, H//p * W//p, d_model)
    """

    def __init__(self, patch_size: int, d_model: int):
        super().__init__()
        self.proj = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)       # (B, d_model, H//p, W//p)
        x = x.flatten(2)       # (B, d_model, N)
        x = x.transpose(1, 2)  # (B, N, d_model)
        return x


# ── DiTBlock ──────────────────────────────────────────────────────────────────

class DiTBlock(nn.Module):
    """Pre-norm transformer block with adaLN-Zero conditioning.

    Forward:
        x: (B, N, d_model) — patch tokens
        c: (B, d_model)    — conditioning signal (time + class)
    Returns: (B, N, d_model)
    """

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn  = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True, bias=True
        )
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        ffn_hidden = int(mlp_ratio * d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, d_model),
        )
        # adaLN MLP: SiLU activation + single linear projecting c → 6 * d_model
        self.adaln_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model),
        )
        # Zero-init the final projection (the "Zero" in adaLN-Zero)
        nn.init.zeros_(self.adaln_mlp[-1].weight)
        nn.init.zeros_(self.adaln_mlp[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # Compute 6 adaLN parameters; unsqueeze to broadcast over sequence dim
        s = self.adaln_mlp(c)  # (B, 6*d_model)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = s.chunk(6, dim=-1)
        shift_msa = shift_msa.unsqueeze(1)  # (B, 1, d_model)
        scale_msa = scale_msa.unsqueeze(1)
        gate_msa  = gate_msa.unsqueeze(1)
        shift_mlp = shift_mlp.unsqueeze(1)
        scale_mlp = scale_mlp.unsqueeze(1)
        gate_mlp  = gate_mlp.unsqueeze(1)

        # Attention sublayer
        normed = (1 + scale_msa) * self.norm1(x) + shift_msa
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + gate_msa * attn_out

        # FFN sublayer
        normed = (1 + scale_mlp) * self.norm2(x) + shift_mlp
        x = x + gate_mlp * self.ffn(normed)

        return x
