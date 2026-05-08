"""
Hyperbolic Causal Language Model (HyViT-LM).

Architecture:
  token_ids → HyperbolicTokenEmbed → N × LorentzTransformerBlock (causal)
            → LorentzLayerNorm → log_map_origin (spatial) → linear LM head

The token and positional embeddings live in Euclidean space and are projected
to the hyperboloid via project_to_hyperboloid before the transformer blocks.
The LM head maps back to Euclidean space via log_map_origin before the linear
projection to vocabulary logits.
"""

import torch
import torch.nn as nn
from geometry.lorentz import project_to_hyperboloid, log_map_origin
from models.lorentz_block import LorentzTransformerBlock
from models.lorentz_layers import LorentzLayerNorm


class HyperbolicTokenEmbed(nn.Module):
    """
    Token + positional embedding projected to the hyperboloid.

    Stores Euclidean token and position vectors, sums them,
    then lifts to H^{d_model} via project_to_hyperboloid.
    """

    def __init__(self, vocab_size: int, d_model: int, seq_len: int, dropout: float = 0.0):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed   = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.dropout     = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.token_embed.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T) — token indices
        Returns: (B, T, d_model+1) on H^{d_model}
        """
        T   = x.size(1)
        emb = self.token_embed(x) + self.pos_embed[:, :T]   # (B, T, d_model)
        emb = self.dropout(emb)
        return project_to_hyperboloid(emb)                   # (B, T, d_model+1)


class HyViTCausalLM(nn.Module):
    """Hyperbolic causal language model for WikiText experiments."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.embed = HyperbolicTokenEmbed(
            vocab_size = cfg.vocab_size,
            d_model    = cfg.d_model,
            seq_len    = cfg.seq_len,
            dropout    = cfg.embed_dropout,
        )

        self.blocks = nn.ModuleList([
            LorentzTransformerBlock(
                d_model   = cfg.d_model,
                n_heads   = cfg.n_heads,
                mlp_ratio = cfg.mlp_ratio,
                dropout   = cfg.dropout,
                causal    = True,
            )
            for _ in range(cfg.n_blocks)
        ])

        self.norm = LorentzLayerNorm(cfg.d_model)

        # LM head: d_model Euclidean features → vocab logits
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        nn.init.trunc_normal_(self.lm_head.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T) — token indices
        Returns: (B, T, vocab_size) logits
        """
        x = self.embed(x)        # (B, T, d_model+1)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)         # (B, T, d_model+1)

        # Map from hyperboloid to Euclidean tangent space, drop time component
        x_euclid = log_map_origin(x)[..., 1:]   # (B, T, d_model)

        return self.lm_head(x_euclid)            # (B, T, vocab_size)
