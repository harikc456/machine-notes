"""SwiGLU FFN (Llama-style)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from kromhc_transformer.config import KromHCConfig


class SwiGLUFFN(nn.Module):
    """SwiGLU: d_model → ffn_hidden (gate) × ffn_hidden (value) → d_model."""

    def __init__(self, cfg: KromHCConfig):
        super().__init__()
        self.w = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.out = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(F.silu(self.w(x)) * self.v(x))
