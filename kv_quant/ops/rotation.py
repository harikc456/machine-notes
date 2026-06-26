from __future__ import annotations

import torch


def make_rotation(d: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Random orthogonal matrix (d, d) via QR decomposition."""
    G = torch.randn(d, d, device=device, dtype=dtype)
    Q, _ = torch.linalg.qr(G)
    return Q


def rotate(h: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """h @ R.T per head.
    h: (batch, heads, seq, d)
    R: (heads, d, d)
    Returns: (batch, heads, seq, d)
    """
    # result[b,h,s,e] = sum_d h[b,h,s,d] * R[h,e,d]  (= h @ R.T per head)
    return torch.einsum('bhsd,hed->bhse', h, R)


def unrotate(h: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """h @ R per head — inverse of rotate.
    h: (batch, heads, seq, d)
    R: (heads, d, d)
    Returns: (batch, heads, seq, d)
    """
    # result[b,h,s,d] = sum_e h[b,h,s,e] * R[h,e,d]  (= h @ R per head)
    return torch.einsum('bhse,hed->bhsd', h, R)
