# rbf_ffn/models/rational_ffn.py
import torch
import torch.nn as nn

from rbf_ffn.config import ModelConfig
from rbf_ffn.models.kronecker_linear import KroneckerLinear


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

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        linear_cls = KroneckerLinear if cfg.kronecker_mlp else nn.Linear
        self.up_proj   = linear_cls(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.act       = RationalActivation()
        self.down_proj = linear_cls(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.up_proj(x)))


class RationalGatedFFN(nn.Module):
    """
    Gated FFN with learnable rational activation replacing SiLU.

        gate = RationalActivation(gate_proj(x))
        out  = down_proj(gate * up_proj(x))

    Matches SwiGLU parameter count at ffn_hidden=688.
    No bias (Llama convention). Input/output: (B, N, d_model).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        linear_cls = KroneckerLinear if cfg.kronecker_mlp else nn.Linear
        self.gate_proj = linear_cls(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up_proj   = linear_cls(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.act       = RationalActivation()
        self.down_proj = linear_cls(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class PFDRationalActivation(nn.Module):
    """
    Partial Fraction Decomposition Rational Activation.

    f(x) = sum_{i=1}^{n} (a_i * x + b_i) / (x^2 + c_i^2) + gamma * x

    Denominator x^2 + c_i^2 >= c_i^2 — numerically safe when c_i != 0.
    Parameters a, b, c are vectors of length n; gamma is a scalar.
    Applied element-wise; all parameters are shared across positions and channels.
    """

    def __init__(self, n: int = 4):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(n))
        self.b = nn.Parameter(torch.ones(n) * 0.1)
        self.c = nn.Parameter(torch.arange(1, n + 1, dtype=torch.float))
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_exp = x.unsqueeze(-1)                         # (..., 1)
        denom = x_exp.pow(2) + self.c.pow(2)            # (..., n)
        numer = self.a * x_exp + self.b                 # (..., n)
        return (numer / denom).sum(dim=-1) + self.gamma * x


class PFDRationalFFN(nn.Module):
    """
    Feed-forward network with Partial Fraction Decomposition rational activation.

        up_proj → PFDRationalActivation → down_proj

    No bias on projections (Llama convention).
    Input/output: (B, N, d_model).
    """

    def __init__(self, cfg: ModelConfig, n: int = 4):
        super().__init__()
        linear_cls = KroneckerLinear if cfg.kronecker_mlp else nn.Linear
        self.up_proj   = linear_cls(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.act       = PFDRationalActivation(n)
        self.down_proj = linear_cls(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.up_proj(x)))


class PFDRationalGatedFFN(nn.Module):
    """
    Gated FFN with Partial Fraction Decomposition rational activation replacing SiLU.

        gate = PFDRationalActivation(gate_proj(x))
        out  = down_proj(gate * up_proj(x))

    No bias (Llama convention). Input/output: (B, N, d_model).
    """

    def __init__(self, cfg: ModelConfig, n: int = 4):
        super().__init__()
        linear_cls = KroneckerLinear if cfg.kronecker_mlp else nn.Linear
        self.gate_proj = linear_cls(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up_proj   = linear_cls(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.act       = PFDRationalActivation(n)
        self.down_proj = linear_cls(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class FirstOrderPFDRationalFFN(nn.Module):
    """
    First-order gated FFN with PFD rational activation and shared projection.

        u    = up_proj(x)
        gate = PFDRationalActivation(sin(u + phi))
        out  = down_proj(gate * u)

    phi is a learnable vector of shape (ffn_hidden,) — phase shift that decouples
    the gate signal from the value despite sharing the same projection u.

    2 large matrices instead of 3 (no gate_proj) — ~33% fewer FFN params vs SwiGLU.
    No bias (Llama convention). Input/output: (B, N, d_model).
    """

    def __init__(self, cfg: ModelConfig, n: int = 4):
        super().__init__()
        linear_cls = KroneckerLinear if cfg.kronecker_mlp else nn.Linear
        self.up_proj   = linear_cls(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down_proj = linear_cls(cfg.ffn_hidden, cfg.d_model, bias=False)
        self.phi       = nn.Parameter(torch.randn(cfg.ffn_hidden) * 0.02)
        self.act       = PFDRationalActivation(n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.up_proj(x)
        gate = self.act(torch.sin(u + self.phi))
        return self.down_proj(gate * u)
