# rbf_ffn/models/kronecker_linear.py
import torch
import torch.nn as nn
import math


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
        self.in1, self.in2 = self._get_factors(in_features)
        self.out1, self.out2 = self._get_factors(out_features)

        # Initialize A and B as 2D parameters for Muon compatibility
        self.A = nn.Parameter(torch.empty((self.out1, self.in1), **factory_kwargs))
        self.B = nn.Parameter(torch.empty((self.out2, self.in2), **factory_kwargs))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def _get_factors(self, n: int):
        """Finds two integers a and b such that a * b = n and a is as close to b as possible."""
        for i in range(int(math.isqrt(n)), 0, -1):
            if n % i == 0:
                return i, n // i
        return 1, n  # Fallback (effectively creates a dense matrix if n is prime)

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
