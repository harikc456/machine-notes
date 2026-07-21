from __future__ import annotations
import torch
import torch.nn as nn


class ReasoningEncoder(nn.Module):
    """Compress a variable-length sequence of hidden states into K x d_z slots
    via K learnable queries that cross-attend the sequence. Runs in float32."""

    def __init__(self, d_model: int, n_slots: int, d_z: int, n_heads: int):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_slots, d_model) * 0.02)
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        self.down = nn.Linear(d_model, d_z)

    def forward(self, hidden: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        hidden = hidden.float()
        B = hidden.size(0)
        q = self.ln_q(self.queries).unsqueeze(0).expand(B, -1, -1)
        kv = self.ln_kv(hidden)
        attn_out, _ = self.cross_attn(
            q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False
        )
        return self.down(attn_out)
