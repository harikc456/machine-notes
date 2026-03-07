"""
Core Lorentz (hyperboloid) manifold operations.

Points in H^n are represented as vectors in R^{n+1} satisfying:
    ⟨x, x⟩_L = -x₀² + x₁² + ... + xₙ² = -1,   x₀ > 0

All operations assume the last dimension is the Lorentz dimension (d+1),
with index 0 being the time-like component.
"""

import torch
import torch.nn.functional as F

EPS = 1e-8
MAX_NORM = 50.0   # clip spatial norm for stability (avoids cosh overflow)


def lorentz_inner(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Lorentz (Minkowski) inner product: ⟨x, y⟩_L = -x₀y₀ + Σᵢ xᵢyᵢ

    Args:
        x, y: (..., d+1)
    Returns:
        (...,)  — one scalar per pair of points
    """
    x_time, x_space = x[..., :1], x[..., 1:]
    y_time, y_space = y[..., :1], y[..., 1:]
    return -(x_time * y_time).sum(-1) + (x_space * y_space).sum(-1)


def lorentz_inner_pairwise(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Pairwise Lorentz inner product for attention.

    Args:
        x: (..., N, d+1)
        y: (..., M, d+1)
    Returns:
        (..., N, M)
    """
    x_time, x_space = x[..., :1], x[..., 1:]   # (..., N, 1), (..., N, d)
    y_time, y_space = y[..., :1], y[..., 1:]   # (..., M, 1), (..., M, d)

    time_term  = torch.matmul(x_time, y_time.transpose(-2, -1))   # (..., N, M)
    space_term = torch.matmul(x_space, y_space.transpose(-2, -1)) # (..., N, M)

    return -time_term + space_term


def lorentz_normalize(z: torch.Tensor) -> torch.Tensor:
    """
    Project an arbitrary vector in Minkowski space onto the hyperboloid.
    Computes: z / sqrt(max(-⟨z,z⟩_L, ε))  (with x₀ forced positive)

    Use this for residual connections: lorentz_normalize(x + delta).
    """
    inner = lorentz_inner(z, z)                             # (...,)
    denom = torch.sqrt(torch.clamp(-inner, min=EPS))        # (...,)
    z_norm = z / denom.unsqueeze(-1)
    # Ensure time-like component is positive
    sign = torch.sign(z_norm[..., :1])
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return z_norm * sign


def project_to_hyperboloid(x_space: torch.Tensor) -> torch.Tensor:
    """
    Given spatial components xₛ ∈ Rⁿ, compute the hyperboloid point:
        x = (sqrt(1 + ‖xₛ‖²), xₛ) ∈ H^n

    Args:
        x_space: (..., n)
    Returns:
        x: (..., n+1)
    """
    # Clip to prevent cosh overflow in downstream ops
    space_norm = x_space.norm(dim=-1, keepdim=True)
    scale = (MAX_NORM / (space_norm + EPS)).clamp(max=1.0)
    x_space = x_space * scale

    x_time = torch.sqrt(1.0 + (x_space * x_space).sum(dim=-1, keepdim=True))
    return torch.cat([x_time, x_space], dim=-1)


def exp_map_origin(v: torch.Tensor) -> torch.Tensor:
    """
    Exponential map at the origin o = (1, 0,...,0).

    Maps a tangent vector v ∈ T_o H^n (where v = (0, vₛ))
    to a point on H^n:
        exp_o(v) = (cosh(‖vₛ‖), sinh(‖vₛ‖) · v̂ₛ)

    Args:
        v: (..., n+1) — tangent vector, time component ignored / assumed 0
    Returns:
        x: (..., n+1) on hyperboloid
    """
    v_space = v[..., 1:]                                               # (..., n)
    norm    = v_space.norm(dim=-1, keepdim=True).clamp(max=MAX_NORM)  # (..., 1)
    x_time  = torch.cosh(norm)                                        # (..., 1)
    x_space = torch.sinh(norm) * F.normalize(v_space, dim=-1, eps=EPS)
    return torch.cat([x_time, x_space], dim=-1)


def log_map_origin(x: torch.Tensor) -> torch.Tensor:
    """
    Logarithmic map at the origin o = (1, 0,...,0).

    Maps a point x ∈ H^n to a tangent vector in T_o H^n:
        log_o(x) = (0, arcosh(x₀) · xₛ / ‖xₛ‖)

    Args:
        x: (..., n+1) on hyperboloid
    Returns:
        v: (..., n+1) tangent vector with v₀ = 0
    """
    x_time  = x[..., :1]                                              # (..., 1)
    x_space = x[..., 1:]                                              # (..., n)
    angle   = torch.acosh(x_time.clamp(min=1.0 + EPS))               # (..., 1)
    v_space = angle * F.normalize(x_space, dim=-1, eps=EPS)
    v_time  = torch.zeros_like(x_time)
    return torch.cat([v_time, v_space], dim=-1)


def lorentz_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Geodesic distance in H^n:  d(x, y) = arcosh(-⟨x, y⟩_L)

    Args:
        x, y: (..., n+1)
    Returns:
        (...,)
    """
    inner = lorentz_inner(x, y)
    return torch.acosh((-inner).clamp(min=1.0 + EPS))
