from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from medusa.config import MedusaConfig


class MedusaHead(nn.Module):
    """Single Medusa-1 residual MLP: h_k = SiLU(W1 · h + h).

    Outputs a transformed hidden state (d_model), not logits.
    The shared frozen LM head is applied in MedusaModel.
    W1 initialized to zero so each head starts as identity through SiLU.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.W1 = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.W1.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return F.silu(self.W1(h) + h)


class MedusaModel(nn.Module):
    """K Medusa-1 decoding heads sharing the frozen teacher LM head.

    Each head has its own trainable W1 (d_model×d_model).
    A single frozen lm_head_weight (vocab×d_model) is shared across all heads.
    """

    def __init__(self, cfg: MedusaConfig, lm_head_weight: torch.Tensor) -> None:
        super().__init__()
        self.heads = nn.ModuleList([
            MedusaHead(cfg.d_model)
            for _ in range(cfg.n_heads)
        ])
        self.register_buffer("lm_head_weight", lm_head_weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Returns hidden states (B, K, d_model) — lm_head projection is in the loss."""
        return torch.stack([head(h) for head in self.heads], dim=1)

    def get_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Returns logits (B, K, vocab) — for inference only."""
        return F.linear(self.forward(h), self.lm_head_weight)
