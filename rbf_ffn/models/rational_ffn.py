# rbf_ffn/models/rational_ffn.py
import torch
import torch.nn as nn

from rbf_ffn.config import RBFFFNConfig


class RationalActivation(nn.Module):
    """
    Learnable rational activation f(x) = P(x) / Q(x).

    P(x) = a0 + a1·x + a2·x² + a3·x³  (Horner's method)
    Q(x) = 1 + |x·(b0 + x·b1)|

    Applied element-wise; a and b are shared across all positions and channels.
    Q(x) >= 1 always — division is numerically safe.
    """

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor([0.4401, 0.5, 0.507, 0.05]))
        self.b = nn.Parameter(torch.tensor([0.0, 0.01]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p_x = self.a[0] + x * (self.a[1] + x * (self.a[2] + x * self.a[3]))
        q_x = 1.0 + torch.abs(x * (self.b[0] + x * self.b[1]))
        return p_x / q_x


class RationalFFN(nn.Module):
    """
    Feed-forward network with learnable rational activation.

        up_proj → RationalActivation → down_proj

    No bias on projections (Llama convention).
    Input/output: (B, N, d_model).
    """

    def __init__(self, cfg: RBFFFNConfig):
        super().__init__()
        self.up_proj   = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.act       = RationalActivation()
        self.down_proj = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.up_proj(x)))
