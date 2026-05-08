# rbf_ffn/models/llama_ffn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from rbf_ffn.config import ModelConfig
from rbf_ffn.models.kronecker_linear import KroneckerLinear, KroneckerDeltaLinear


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
        if cfg.kronecker_delta_mlp:
            self.gate_proj = nn.Linear(D, H, bias=False)
            self.up_proj   = KroneckerDeltaLinear(D, H, bias=False, delta_rank=cfg.kronecker_delta_rank)
            self.down_proj = KroneckerDeltaLinear(H, D, bias=False, delta_rank=cfg.kronecker_delta_rank)
        elif cfg.kronecker_mlp:
            self.gate_proj = KroneckerLinear(D, H, bias=False)
            self.up_proj   = KroneckerLinear(D, H, bias=False)
            self.down_proj = KroneckerLinear(H, D, bias=False)
        else:
            self.gate_proj = nn.Linear(D, H, bias=False)
            self.up_proj   = nn.Linear(D, H, bias=False)
            self.down_proj = nn.Linear(H, D, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d_model)"""
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LeakyReLUSquaredFFN(nn.Module):
    """
    Gated FFN using Leaky ReLU Squared as the gate activation.

        gate = LeakyReLU(gate_proj(x)) ** 2
        up   = up_proj(x)
        out  = down_proj(gate * up)

    LeakyReLU² is defined as leaky_relu(x)², always non-negative.
    No bias on any projection (Llama convention).
    Input/output: (B, N, d_model)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        D, H = cfg.d_model, cfg.ffn_hidden
        self.gate_proj = nn.Linear(D, H, bias=False)
        self.up_proj   = nn.Linear(D, H, bias=False)
        self.down_proj = nn.Linear(H, D, bias=False)

    @staticmethod
    def _leaky_relu_squared(x: torch.Tensor) -> torch.Tensor:
        a = F.leaky_relu(x, negative_slope=0.01)
        return a * a

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d_model)"""
        return self.down_proj(self._leaky_relu_squared(self.gate_proj(x)) * self.up_proj(x))
