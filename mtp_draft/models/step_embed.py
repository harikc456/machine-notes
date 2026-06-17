from __future__ import annotations
import math
import torch
import torch.nn as nn


def _sinusoidal(steps: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    steps: (B, S) integer indices
    Returns (B, S, d_model) sinusoidal embeddings (DDPM-style).
    """
    half = d_model // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=steps.device, dtype=torch.float32) / half
    )  # (half,)
    args = steps.float().unsqueeze(-1) * freqs  # (B, S, half)
    return torch.cat([args.sin(), args.cos()], dim=-1)  # (B, S, d_model)


class StepEmbedding(nn.Module):
    """
    Maps integer draft-step indices to d_model vectors via:
        sinusoidal(i) → Linear(d_model, 4*d_model) → SiLU → Linear(4*d_model, d_model)

    Input:  (B, S) integer step indices (1-indexed)
    Output: (B, S, d_model)
    """

    def __init__(self, d_model: int, max_steps: int = 64) -> None:
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.SiLU(),
            nn.Linear(4 * d_model, d_model, bias=False),
        )

    def forward(self, steps: torch.Tensor) -> torch.Tensor:
        """steps: (B, S) → (B, S, d_model)"""
        emb = _sinusoidal(steps, self.d_model)  # (B, S, d_model)
        return self.mlp(emb)
