# rbf_ffn/models/polar_ffn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from rbf_ffn.config import ModelConfig


class AdaptivePolarMLP(nn.Module):
    """
    Purely directional feed-forward network operating in polar coordinate space.

    Input token vectors are L2-normalised onto the unit sphere; cosine similarity
    is computed against `ffn_hidden` learned key directions; a per-neuron sigmoid
    gate (fixed sharpness=10, learnable threshold init=0.7) masks low-alignment
    activations; a bias-free down_proj maps back to d_model.

    Magnitude of the input is discarded entirely.
    Key directions are L2-normalised during the forward pass; raw key norms are
    unconstrained and will be handled by the Muon optimiser.
    Input/output: (B, N, d_model).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        D, H = cfg.d_model, cfg.ffn_hidden
        self.keys       = nn.Parameter(torch.randn(H, D))
        self.thresholds = nn.Parameter(torch.full((H,), 0.7))
        self.down_proj  = nn.Linear(H, D, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d_model)"""
        x_dir   = F.normalize(x, dim=-1, eps=1e-6)            # (B, N, D)
        key_dir = F.normalize(self.keys, dim=-1, eps=1e-6)     # (H, D)
        cos_sim = torch.matmul(x_dir, key_dir.t())             # (B, N, H)
        gate    = torch.sigmoid(10.0 * (cos_sim - self.thresholds))
        return self.down_proj(cos_sim * gate)                  # (B, N, D)
