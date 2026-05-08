"""
Euclidean Causal Language Model (EucLM).

Same architectural skeleton as HypLM — same depth, width, and LM head —
but with standard dot-product attention and Euclidean operations throughout.
Used to isolate the effect of hyperbolic geometry from other design choices.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class EuclideanCausalMHSA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = math.sqrt(self.d_head)
        self.qkv     = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj    = nn.Linear(d_model, d_model)
        self.drop    = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        H, Dh   = self.n_heads, self.d_head
        qkv = self.qkv(x).reshape(B, N, 3, H, Dh).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv.unbind(0)
        scores = (Q @ K.transpose(-2, -1)) / self.scale        # (B, H, N, N)
        mask   = torch.triu(
            torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(mask, float("-inf"))
        attn   = F.softmax(scores, dim=-1)
        attn   = self.drop(attn)
        out    = (attn @ V).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class EuclideanBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = EuclideanCausalMHSA(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        d_ff = d_model * mlp_ratio
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class EucLM(nn.Module):
    """Euclidean causal language model for WikiText experiments."""

    def __init__(self, cfg):
        super().__init__()
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed   = nn.Parameter(torch.zeros(1, cfg.seq_len, cfg.d_model))
        self.embed_drop  = nn.Dropout(cfg.embed_dropout)
        self.blocks = nn.ModuleList([
            EuclideanBlock(cfg.d_model, cfg.n_heads, cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.n_blocks)
        ])
        self.norm    = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        nn.init.trunc_normal_(self.token_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.lm_head.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T) — token indices
        Returns: (B, T, vocab_size) logits
        """
        T   = x.size(1)
        emb = self.token_embed(x) + self.pos_embed[:, :T]
        emb = self.embed_drop(emb)
        for block in self.blocks:
            emb = block(emb)
        return self.lm_head(self.norm(emb))
