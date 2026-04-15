# rbf_ffn/models/orthogonal_ffn.py
import torch
import torch.nn as nn


class OrthogonalMLPWrapper(nn.Module):
    """
    Wraps any FFN module so its output is orthogonal to its input.

    Given the standard FFN output y = mlp(x), computes:

        y_parallel    = (y · x / (x · x + eps)) * x
        y_orthogonal  = y - y_parallel

    This projects out the component of the FFN output that lies along the
    residual stream direction, so the additive update is guaranteed to be
    perpendicular to x at every position.

    Input/output: (B, N, d_model)
    """

    def __init__(self, mlp_module: nn.Module, eps: float = 1e-8):
        super().__init__()
        self.mlp = mlp_module
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.mlp(x)
        dot_yx = torch.sum(y * x, dim=-1, keepdim=True)
        dot_xx = torch.sum(x * x, dim=-1, keepdim=True)
        y_parallel = (dot_yx / (dot_xx + self.eps)) * x
        return y - y_parallel
