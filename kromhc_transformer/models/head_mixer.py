"""KromHC Head Mixer: Kronecker-factored doubly-stochastic head mixing."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import permutations


class KromHCHeadMixer(nn.Module):
    """
    Mixes attention head outputs using Kronecker-factored permutation matrices.

    For n_heads=8: factors=[2,2,2], builds 3 tiny 2×2 permutation matrices,
    Kronecker-chains them into an 8×8 doubly-stochastic matrix per token.

    Input:  (bs, n_heads, head_dim)
    Output: mixed (bs, n_heads, head_dim), H (bs, n_heads, n_heads)
    """

    def __init__(self, n_heads: int = 8, head_dim: int = 64, d_context: int = None):
        super().__init__()
        self.n = n_heads
        self.head_dim = head_dim
        if d_context is None:
            d_context = head_dim

        k = int(math.log2(n_heads))
        assert 2 ** k == n_heads, f"n_heads ({n_heads}) must be a power of 2"
        self.K = k

        # One permutation basis + weight generator per factor of 2
        self.perm_bases = nn.ParameterList()
        self.weight_gens = nn.ModuleList()

        for _ in range(k):
            # All 2! = 2 permutation matrices of size 2×2
            basis = torch.zeros(2, 2, 2)
            for idx, p in enumerate(permutations(range(2))):
                for r, c in enumerate(p):
                    basis[idx, r, c] = 1.0
            self.perm_bases.append(nn.Parameter(basis, requires_grad=False))

            # Small MLP: context → 2 weights (for convex combination of the 2 permutations)
            self.weight_gens.append(nn.Sequential(
                nn.Linear(d_context, 32),
                nn.ReLU(),
                nn.Linear(32, 2),
            ))

    def _batched_kronecker(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Batched Kronecker product: (bs, m, m) ⊗ (bs, p, p) → (bs, m*p, m*p)"""
        bs, m, _ = A.shape
        p = B.shape[1]
        return torch.einsum('b i j, b k l -> b i k j l', A, B).reshape(bs, m * p, m * p)

    def forward(self, x: torch.Tensor):
        """
        x: (bs, n_heads, head_dim)
        Returns: (mixed: same shape, H: (bs, n_heads, n_heads))
        """
        bs, n, d = x.shape
        assert n == self.n, f"Expected {self.n} heads, got {n}"

        # Context: mean across heads for each token
        context = x.mean(dim=1)  # (bs, head_dim)

        # Build one 2×2 doubly-stochastic matrix per factor
        small_us = []
        for gen, basis in zip(self.weight_gens, self.perm_bases):
            logits = gen(context)                       # (bs, 2)
            a = F.softmax(logits, dim=-1)              # convex weights summing to 1
            U = (a @ basis.view(2, -1)).view(bs, 2, 2) # (bs, 2, 2)
            small_us.append(U)

        # Kronecker chain: [2×2, 2×2, 2×2] → 8×8 for n_heads=8
        H = small_us[0]
        for U in small_us[1:]:
            H = self._batched_kronecker(H, U)  # (bs, n_heads, n_heads)

        # Apply mixing matrix
        out = torch.matmul(H, x)  # (bs, n_heads, head_dim)
        return out, H
