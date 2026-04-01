"""KromHC Head Mixer: Kronecker-factored doubly-stochastic head mixing."""
from __future__ import annotations

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

    def __init__(self, n_heads: int = 8, head_dim: int = 64, d_context: int | None = None, mixer_hidden: int = 32):
        super().__init__()
        self.n = n_heads
        self.head_dim = head_dim
        if d_context is None:
            d_context = head_dim
        self.d_context = d_context

        k = int(math.log2(n_heads))
        assert 2 ** k == n_heads, f"n_heads ({n_heads}) must be a power of 2"
        self.K = k

        # Project from head_dim to d_context if needed
        if head_dim != d_context:
            self.context_proj = nn.Linear(head_dim, d_context, bias=False)
        else:
            self.context_proj = None

        self.weight_gens = nn.ModuleList()
        bases = []

        for i in range(k):
            basis = torch.zeros(2, 2, 2)
            for idx, p in enumerate(permutations(range(2))):
                for r, c in enumerate(p):
                    basis[idx, r, c] = 1.0
            bases.append(basis)

            self.weight_gens.append(nn.Sequential(
                nn.Linear(d_context, mixer_hidden, bias=False),
                nn.ReLU(),
                nn.Linear(mixer_hidden, 2, bias=False),
            ))

        # Stack all bases into a single buffer: (K, 2, 2, 2)
        self.register_buffer('perm_bases', torch.stack(bases))

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

        context = x.mean(dim=1)  # (bs, head_dim)
        if self.context_proj is not None:
            context = self.context_proj(context)  # (bs, d_context)

        small_us = []
        for i, gen in enumerate(self.weight_gens):
            basis = self.perm_bases[i]                   # (2, 2, 2)
            logits = gen(context)                        # (bs, 2)
            a = F.softmax(logits, dim=-1)               # convex weights
            U = (a @ basis.view(2, -1)).view(bs, 2, 2)  # (bs, 2, 2)
            small_us.append(U)

        H = small_us[0]
        for U in small_us[1:]:
            H = self._batched_kronecker(H, U)  # (bs, n_heads, n_heads)

        out = torch.matmul(H, x)  # (bs, n_heads, head_dim)
        return out, H
