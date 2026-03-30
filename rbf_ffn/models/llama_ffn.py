# rbf_ffn/models/llama_ffn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from rbf_ffn.config import ModelConfig
from rbf_ffn.models.kronecker_linear import KroneckerLinear


class SwiGLUFFN(nn.Module):
    """
    Llama-style SwiGLU feed-forward network.

        gate = SiLU(gate_proj(x))
        up   = up_proj(x)
        out  = down_proj(gate * up)

    No bias on any projection (Llama convention).
    Input/output: (B, N, d_model)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        D, H = cfg.d_model, cfg.ffn_hidden
        linear_cls = KroneckerLinear if cfg.kronecker_mlp else nn.Linear
        self.gate_proj = linear_cls(D, H, bias=False)
        self.up_proj   = linear_cls(D, H, bias=False)
        self.down_proj = linear_cls(H, D, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d_model)"""
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
