"""
Lorentz Multi-Head Self-Attention.

Score function: score(qᵢ, kⱼ) = -⟨qᵢ, kⱼ⟩_L  (≥ 1 for manifold points)
Higher score = points are closer on the hyperboloid = should attend more.

Aggregation: Lorentz centroid (weighted projection back to hyperboloid).

Each head operates in H^{d_head} (d_head = d_model // n_heads).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from geometry.lorentz import lorentz_inner_pairwise, lorentz_normalize
from models.lorentz_layers import LorentzLinear, LorentzLayerNorm


class LorentzMultiheadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        return_attn_weights: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        self.return_attn_weights = return_attn_weights

        self.W_q = LorentzLinear(d_model, d_model)
        self.W_k = LorentzLinear(d_model, d_model)
        self.W_v = LorentzLinear(d_model, d_model)
        self.W_o = LorentzLinear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale   = math.sqrt(self.d_head)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, d_model+1)
        Returns: (B, n_heads, N, d_head+1)

        Takes spatial components, reshapes into heads, recomputes time-like dim
        so each head's slice is itself a valid hyperboloid point.
        """
        B, N, _ = x.shape
        x_space = x[..., 1:]                                           # (B, N, d_model)
        x_space = x_space.view(B, N, self.n_heads, self.d_head)        # (B, N, H, d_head)
        x_space = x_space.permute(0, 2, 1, 3)                         # (B, H, N, d_head)
        # Recompute time component for each head slice
        x_time  = torch.sqrt(1.0 + (x_space ** 2).sum(-1, keepdim=True))
        return torch.cat([x_time, x_space], dim=-1)                    # (B, H, N, d_head+1)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, n_heads, N, d_head+1)
        Returns: (B, N, d_model+1)
        """
        B, H, N, _ = x.shape
        x_space = x[..., 1:]                                           # (B, H, N, d_head)
        x_space = x_space.permute(0, 2, 1, 3).contiguous()            # (B, N, H, d_head)
        x_space = x_space.view(B, N, self.d_model)                     # (B, N, d_model)
        x_time  = torch.sqrt(1.0 + (x_space ** 2).sum(-1, keepdim=True))
        return torch.cat([x_time, x_space], dim=-1)                    # (B, N, d_model+1)

    def forward(self, x: torch.Tensor):
        """
        x: (B, N, d_model+1) — sequence of Lorentz points
        Returns: (B, N, d_model+1)
        """
        Q = self._split_heads(self.W_q(x))   # (B, H, N, d_head+1)
        K = self._split_heads(self.W_k(x))
        V = self._split_heads(self.W_v(x))

        # Lorentz similarity: score = -⟨q, k⟩_L   (B, H, N, N)
        scores = -lorentz_inner_pairwise(Q, K) / self.scale
        attn_w = F.softmax(scores, dim=-1)                             # (B, H, N, N)
        attn_w = self.dropout(attn_w)

        # Aggregate: z[b,h,i] = Σⱼ attn_w[b,h,i,j] * V[b,h,j]  (ambient sum)
        z   = torch.einsum("bhin,bhnd->bhid", attn_w, V)              # (B, H, N, d_head+1)
        out = lorentz_normalize(z)                                     # project to H^{d_head}

        out = self._merge_heads(out)                                   # (B, N, d_model+1)
        out = self.W_o(out)

        if self.return_attn_weights:
            return out, attn_w
        return out
