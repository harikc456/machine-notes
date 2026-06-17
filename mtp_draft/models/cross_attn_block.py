from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from mtp_draft.config import MTPConfig


class CrossAttnBlock(nn.Module):
    """
    Pre-norm cross-attention block with KV-sharing and SwiGLU FFN.

    K and V both come from a single kv_proj applied to the context
    (same projection, no RoPE — context positions are not relative to
    draft positions). No causal mask between draft query positions.

    Input:
        query:   (B, S, d_draft)  — S independent draft positions
        context: (B, N, d_draft)  — N context tokens

    Output: (B, S, d_draft)
    """

    def __init__(self, cfg: MTPConfig) -> None:
        super().__init__()
        D = cfg.d_draft
        H = cfg.n_heads
        assert D % H == 0, f"d_draft ({D}) must be divisible by n_heads ({H})"
        self.n_heads = H
        self.head_dim = D // H
        self._dropout = cfg.dropout

        # Cross-attention projections
        self.q_proj = nn.Linear(D, D, bias=False)
        self.kv_proj = nn.Linear(D, D, bias=False)   # K = V = kv_proj(context)
        self.o_proj = nn.Linear(D, D, bias=False)

        # FFN (SwiGLU)
        H_ffn = cfg.ffn_hidden
        self.gate_proj = nn.Linear(D, H_ffn, bias=False)
        self.up_proj = nn.Linear(D, H_ffn, bias=False)
        self.down_proj = nn.Linear(H_ffn, D, bias=False)

        # Pre-norm
        self.norm_q = nn.RMSNorm(D)
        self.norm_ctx = nn.RMSNorm(D)
        self.norm_ffn = nn.RMSNorm(D)

    def _attn(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, S, D = query.shape
        N = context.shape[1]
        H, hd = self.n_heads, self.head_dim

        q = self.q_proj(query).view(B, S, H, hd).transpose(1, 2)    # (B, H, S, hd)
        kv = self.kv_proj(context).view(B, N, H, hd).transpose(1, 2)  # (B, H, N, hd)
        # K and V are the same projected tensor (KV-sharing)
        k = kv
        v = kv

        dp = self._dropout if self.training else 0.0
        # is_causal=False: draft positions attend freely to all context positions
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dp, is_causal=False)
        return self.o_proj(out.transpose(1, 2).contiguous().view(B, S, D))

    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        query:   (B, S, d_draft)
        context: (B, N, d_draft)
        returns: (B, S, d_draft)
        """
        query = query + self._attn(self.norm_q(query), self.norm_ctx(context))
        query = query + self._ffn(self.norm_ffn(query))
        return query
