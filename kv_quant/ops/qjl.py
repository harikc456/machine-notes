from __future__ import annotations

import torch


def make_sign_matrix(m: int, d: int, device=None) -> torch.Tensor:
    """Random ±1/√m sign matrix of shape (m, d)."""
    S = torch.randint(0, 2, (m, d), device=device).float() * 2.0 - 1.0
    return S / (m ** 0.5)


def encode(h: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """1-bit QJL encoding for 4-D tensors (TurboQuant path).

    h: (batch, heads, seq, d)
    S: (heads, m, d)
    Returns: (batch, heads, seq, m) bool  — True encodes +1
    """
    # proj[b,h,s,m] = sum_d h[b,h,s,d] * S[h,m,d]  (= h @ S.T per head)
    proj = torch.einsum('bhsd,hmd->bhsm', h, S)
    return proj >= 0.0


def encode_2d(h: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """1-bit QJL encoding for 2-D tensors (SpectralQuant path).

    h: (N, d)
    S: (m, d)
    Returns: (N, m) bool
    """
    return (h @ S.T) >= 0.0
