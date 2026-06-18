from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from medusa.config import MedusaConfig


class MedusaHead(nn.Module):
    """Single Medusa decoding head.

    Implements: logits = W2 · SiLU(W1 · h + h)
    W1 initialized to zero — at init, output is W2(SiLU(h)); heads learn a residual correction
    W2 init: clone of frozen teacher LM head weight
    """

    def __init__(self, d_model: int, lm_head_weight: torch.Tensor) -> None:
        super().__init__()
        vocab_size = lm_head_weight.shape[0]
        self.W1 = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.W1.weight)
        self.W2 = nn.Linear(d_model, vocab_size, bias=False)
        self.W2.weight = nn.Parameter(lm_head_weight.clone().float())

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.W2(F.silu(self.W1(h) + h))


class MedusaModel(nn.Module):
    """K Medusa-1 decoding heads sharing a common input hidden state."""

    def __init__(self, cfg: MedusaConfig, lm_head_weight: torch.Tensor) -> None:
        super().__init__()
        self.heads = nn.ModuleList([
            MedusaHead(cfg.d_model, lm_head_weight)
            for _ in range(cfg.n_heads)
        ])

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.stack([head(h) for head in self.heads], dim=1)
