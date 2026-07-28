from __future__ import annotations
import math
import torch
import torch.nn as nn


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard transformer/DDPM sinusoidal timestep embedding.
    timesteps: (B,) int/long tensor. Returns (B, dim) float tensor."""
    half = dim // 2
    device = timesteps.device
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=device) / half
    )
    args = timesteps.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class RefineBlock(nn.Module):
    """One diffusion-style refinement step: timestep conditioning, self-attn
    over the K latent slots, cross-attn to question hidden states, feed-forward.
    Shared across all T steps (only the timestep embedding differs per step),
    matching standard diffusion-network practice."""

    def __init__(self, d_z: int, d_model: int, n_heads: int):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(d_z, d_z), nn.SiLU(), nn.Linear(d_z, d_z)
        )
        self.ln_self = nn.LayerNorm(d_z)
        self.self_attn = nn.MultiheadAttention(d_z, n_heads, batch_first=True)
        self.ln_cross = nn.LayerNorm(d_z)
        self.cross_attn = nn.MultiheadAttention(
            d_z, n_heads, kdim=d_model, vdim=d_model, batch_first=True
        )
        self.ln_ff = nn.LayerNorm(d_z)
        self.ff = nn.Sequential(
            nn.Linear(d_z, 4 * d_z), nn.GELU(), nn.Linear(4 * d_z, d_z)
        )

    def forward(
        self, z: torch.Tensor, t_emb: torch.Tensor,
        question_hidden: torch.Tensor, question_kpm: torch.Tensor,
    ) -> torch.Tensor:
        z = z + self.time_mlp(t_emb).unsqueeze(1)  # broadcast over K slots

        h = self.ln_self(z)
        attn_out, _ = self.self_attn(h, h, h, need_weights=False)
        z = z + attn_out

        h = self.ln_cross(z)
        cross_out, _ = self.cross_attn(
            h, question_hidden, question_hidden,
            key_padding_mask=question_kpm, need_weights=False,
        )
        z = z + cross_out

        h = self.ln_ff(z)
        z = z + self.ff(h)
        return z


class DiffusionReasoningEncoder(nn.Module):
    """Produces a reasoning latent z (K x d_z) from the question ALONE, via
    T fully-unrolled refinement steps starting from Gaussian noise. No
    ground-truth z, no denoising-score-matching loss: the only supervision
    is whatever loss the caller backprops through `forward`'s output. Runs
    in float32, matching ReasoningEncoder's precision convention."""

    def __init__(self, d_model: int, n_slots: int, d_z: int, n_heads: int, n_steps: int):
        super().__init__()
        self.n_slots = n_slots
        self.d_z = d_z
        self.n_steps = n_steps
        self.block = RefineBlock(d_z, d_model, n_heads)

    def forward(self, question_hidden: torch.Tensor, question_kpm: torch.Tensor) -> torch.Tensor:
        question_hidden = question_hidden.float()
        B = question_hidden.size(0)
        z = torch.randn(B, self.n_slots, self.d_z, device=question_hidden.device)
        for t in reversed(range(self.n_steps)):
            t_batch = torch.full((B,), t, device=question_hidden.device, dtype=torch.long)
            t_emb = sinusoidal_embedding(t_batch, self.d_z)
            z = self.block(z, t_emb, question_hidden, question_kpm)
        return z
