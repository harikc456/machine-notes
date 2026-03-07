"""
Lorentz Transformer Block.

Structure (Pre-norm):
    x_attn = LorentzLayerNorm(x)
    x_attn = LorentzMHSA(x_attn)
    x       = lorentz_normalize(x + x_attn)          ← hyperbolic residual

    x_ffn  = LorentzLayerNorm(x)
    x_ffn  = LorentzFFN(x_ffn)
    x       = lorentz_normalize(x + x_ffn)            ← hyperbolic residual

Residual connection: lorentz_normalize(x + delta)
The ambient sum x + delta moves off the hyperboloid; lorentz_normalize
reprojects it. Faster and more stable than geodesic midpoints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from geometry.lorentz import project_to_hyperboloid, lorentz_normalize
from models.lorentz_layers import LorentzLinear, LorentzLayerNorm
from models.lorentz_attention import LorentzMultiheadAttention


class LorentzFFN(nn.Module):
    """
    Fully hyperbolic feed-forward network.

    Pattern: LorentzLinear → GELU (on spatial components) → LorentzLinear
    """
    def __init__(self, d_model: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        d_hidden  = d_model * mlp_ratio
        self.fc1  = LorentzLinear(d_model, d_hidden)
        self.fc2  = LorentzLinear(d_hidden, d_model)
        self.drop = nn.Dropout(dropout)
        self.norm = LorentzLayerNorm(d_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., d_model+1) on H^{d_model}"""
        x = self.fc1(x)                               # (..., d_hidden+1)
        # Apply GELU to spatial components only, then reproject to hyperboloid
        x_space = F.gelu(x[..., 1:])                 # (..., d_hidden)
        x_space = self.drop(x_space)
        x = project_to_hyperboloid(x_space)
        x = self.norm(x)
        x = self.fc2(x)                               # (..., d_model+1)
        return x


class LorentzTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = LorentzLayerNorm(d_model)
        self.attn  = LorentzMultiheadAttention(d_model, n_heads, dropout)
        self.norm2 = LorentzLayerNorm(d_model)
        self.ffn   = LorentzFFN(d_model, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d_model+1) on H^{d_model}"""
        # Self-attention sub-layer with hyperbolic residual
        x = lorentz_normalize(x + self.attn(self.norm1(x)))
        # FFN sub-layer with hyperbolic residual
        x = lorentz_normalize(x + self.ffn(self.norm2(x)))
        return x
