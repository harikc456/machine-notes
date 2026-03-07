"""
Lorentz-model neural network building blocks.

LorentzLinear   — fully hyperbolic linear layer (maps H^n → H^m)
LorentzLayerNorm — normalize spatial components, reproject to manifold
LorentzCentroid  — weighted Fréchet mean (for attention aggregation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from geometry.lorentz import project_to_hyperboloid, lorentz_normalize

EPS = 1e-8


class LorentzLinear(nn.Module):
    """
    Fully hyperbolic linear layer: H^in → H^out

    For x = (x₀, xₛ) ∈ H^in:
        yₛ = W xₛ + b     (Euclidean linear on spatial components)
        y  = (sqrt(1 + ‖yₛ‖²), yₛ) ∈ H^out

    Input dim  to this layer: in_features + 1 (Lorentz)
    Output dim:               out_features + 1 (Lorentz)
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., in_features + 1)  — points on H^in
        Returns: (..., out_features + 1) — points on H^out
        """
        x_space = x[..., 1:]                                          # (..., in_features)
        y_space = F.linear(x_space, self.weight, self.bias)           # (..., out_features)
        return project_to_hyperboloid(y_space)


class LorentzLayerNorm(nn.Module):
    """
    Layer normalization for hyperbolic features.

    Normalizes spatial components of the Lorentz representation,
    applies learnable scale/shift, then reprojects onto the hyperboloid.

    Input/output: (..., d_model+1)
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps     = eps
        self.gamma   = nn.Parameter(torch.ones(d_model))
        self.beta    = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., d_model + 1)"""
        x_space = x[..., 1:]                                          # (..., d_model)
        mean    = x_space.mean(dim=-1, keepdim=True)
        var     = x_space.var(dim=-1, keepdim=True, unbiased=False)
        x_norm  = (x_space - mean) / torch.sqrt(var + self.eps)
        x_scaled = self.gamma * x_norm + self.beta
        return project_to_hyperboloid(x_scaled)


class LorentzCentroid(nn.Module):
    """
    Weighted Lorentz centroid (Fréchet mean approximation):
        z = Σᵢ wᵢ xᵢ  (ambient weighted sum)
        output = lorentz_normalize(z)  (project back to hyperboloid)

    Used as the aggregation step in Lorentz attention.
    """

    @staticmethod
    def apply_weights(
        x: torch.Tensor,       # (..., N, d+1) — value points on H^n
        weights: torch.Tensor  # (..., N)       — attention weights summing to 1
    ) -> torch.Tensor:
        """Returns (..., d+1) on hyperboloid."""
        z = torch.einsum("...n,...nd->...d", weights, x)  # (..., d+1)
        return lorentz_normalize(z)
