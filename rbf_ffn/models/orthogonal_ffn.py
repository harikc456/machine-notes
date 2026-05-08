# rbf_ffn/models/orthogonal_ffn.py
import torch
import torch.nn as nn


class GatedOrthogonalMLPWrapper(nn.Module):
    """
    Wraps any FFN so the update combines strictly orthogonal novel features
    with a gated amplification/erasure of the input direction.

    Given y = mlp(x) and scalar projection c = (y · x) / (||x||² + eps):

        y_orthogonal        = y - c * x          (novel features, ⊥ to x)
        amplification_gate  = gate_scale * act(c) (bounded gate on input direction)
        y_parallel_gated    = amplification_gate * x

    Output = y_orthogonal + y_parallel_gated

    With gate_activation='tanh' the gate is bounded in (−1, 1) times gate_scale,
    so the parallel contribution can erase (negative) or amplify (positive) the
    input, but never diverge.

    gate_scale is a learnable scalar initialised to 0.1 to keep amplification
    small at the start of training.

    Input/output: (B, N, d_model)
    """

    def __init__(
        self,
        mlp_module: nn.Module,
        eps: float = 1e-8,
        gate_activation: str = "tanh",
    ):
        super().__init__()
        self.mlp = mlp_module
        self.eps = eps
        self.gate_scale = nn.Parameter(torch.tensor(0.1))

        if gate_activation == "tanh":
            self.activation = torch.tanh
        elif gate_activation == "softsign":
            self.activation = nn.Softsign()
        else:
            self.activation = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.mlp(x)
        dot_yx = torch.sum(y * x, dim=-1, keepdim=True)
        dot_xx = torch.sum(x * x, dim=-1, keepdim=True)
        c = dot_yx / (dot_xx + self.eps)
        y_orthogonal = y - c * x
        amplification_gate = self.gate_scale * self.activation(c)
        return y_orthogonal + amplification_gate * x


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
