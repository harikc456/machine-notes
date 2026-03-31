# rbf_ffn/models/kronecker_linear.py
import torch
import torch.nn as nn
import math


def _get_factors(n: int) -> tuple[int, int]:
    """Returns (a, b) with a * b == n and |a - b| minimised."""
    root = int(math.isqrt(n))
    for i in range(root, 0, -1):
        if n % i == 0:
            return i, n // i
    return 1, n  # Fallback for primes: creates an unbalanced (1, n) factorisation


class KroneckerLinear(nn.Module):
    """
    A drop-in replacement for torch.nn.Linear that uses Kronecker-factored weights
    to drastically reduce parameters while maintaining matrix rank.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Automatically factorize the input and output dimensions
        self.in1, self.in2 = _get_factors(in_features)
        self.out1, self.out2 = _get_factors(out_features)

        # Initialize A and B as 2D parameters for Muon compatibility
        self.A = nn.Parameter(torch.empty((self.out1, self.in1), **factory_kwargs))
        self.B = nn.Parameter(torch.empty((self.out2, self.in2), **factory_kwargs))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming uniform initialization adapted for factored matrices
        bound = 1 / math.sqrt(self.in_features)
        factor_bound = math.sqrt(bound)

        nn.init.uniform_(self.A, -factor_bound, factor_bound)
        nn.init.uniform_(self.B, -factor_bound, factor_bound)

        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [..., in_features]
        original_shape = x.shape
        batch_dims = original_shape[:-1]

        # 1. Reshape input: [..., in1, in2]
        x_reshaped = x.view(*batch_dims, self.in1, self.in2)

        # 2. Apply A and B efficiently without materializing the Kronecker product
        out = torch.einsum('...ij, mi, nj -> ...mn', x_reshaped, self.A, self.B)

        # 3. Flatten the output dimensions back
        out = out.reshape(*batch_dims, self.out_features)

        if self.bias is not None:
            out += self.bias

        return out

    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, '
                f'factors=(in:{self.in1}x{self.in2}, out:{self.out1}x{self.out2})')


class KroneckerDeltaLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear using a Kronecker-factored core
    plus a low-rank delta pathway for parameter-efficient pretraining.

    Optimizer routing:
      - A, B (2-D)            → Muon  (via ndim == 2 rule)
      - delta_C, delta_D (2-D)→ AdamW (via "delta_" name rule)

    Initialisation (LoRA convention): delta_C is zero-initialised so the delta
    pathway contributes nothing on the first forward pass. delta_D.grad is
    structurally zero on the first backward pass; delta_D will not move until
    delta_C becomes nonzero after the first AdamW step on delta_C.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 delta_rank: int = 16, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.delta_rank = delta_rank

        self.in1, self.in2 = _get_factors(in_features)
        self.out1, self.out2 = _get_factors(out_features)

        # Kronecker core — trained by Muon
        self.A = nn.Parameter(torch.empty((self.out1, self.in1), **factory_kwargs))
        self.B = nn.Parameter(torch.empty((self.out2, self.in2), **factory_kwargs))

        # Low-rank delta — trained by AdamW (name prefix "delta_" triggers routing)
        self.delta_C = nn.Parameter(torch.empty((out_features, delta_rank), **factory_kwargs))
        self.delta_D = nn.Parameter(torch.empty((delta_rank, in_features), **factory_kwargs))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        kron_bound = 1 / math.sqrt(self.in_features)
        factor_bound = math.sqrt(kron_bound)
        nn.init.uniform_(self.A, -factor_bound, factor_bound)
        nn.init.uniform_(self.B, -factor_bound, factor_bound)
        nn.init.kaiming_uniform_(self.delta_D, a=math.sqrt(5))
        nn.init.zeros_(self.delta_C)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -kron_bound, kron_bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        batch_dims = original_shape[:-1]

        x_reshaped = x.view(*batch_dims, self.in1, self.in2)
        kron_out = torch.einsum('...ij,mi,nj->...mn', x_reshaped, self.A, self.B)
        kron_out = kron_out.reshape(*batch_dims, self.out_features)

        delta_out = (x @ self.delta_D.T) @ self.delta_C.T

        out = kron_out + delta_out
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'bias={self.bias is not None}, '
                f'factors=(in:{self.in1}x{self.in2}, out:{self.out1}x{self.out2}), '
                f'delta_rank={self.delta_rank}')
