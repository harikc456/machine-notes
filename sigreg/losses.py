# sigreg/losses.py
"""
SIGReg auxiliary losses.

  sigreg_strong_loss  — matches the empirical characteristic function (ECF) to
                        a standard Gaussian CF, forcing all moments toward
                        Gaussian (Maximum Entropy Cloud).

  sigreg_weak_loss    — matches the sample covariance to the identity matrix,
                        forcing second-moment sphericity only.

Both operate on a 2-D tensor x of shape (N, C) where N is the effective
batch dimension (B*T tokens) and C is the channel/feature dimension.
"""
from __future__ import annotations
import torch


def sigreg_strong_loss(x: torch.Tensor, sketch_dim: int = 64) -> torch.Tensor:
    """
    Forces ECF(x) ~ ECF(Gaussian).
    Matches ALL moments (Maximum Entropy Cloud).

    Algorithm:
      1. Project channels → sketch_dim via a random unit-column matrix.
      2. Evaluate the empirical CF at 17 uniformly-spaced integration points.
      3. Compare against the theoretical standard-Gaussian CF exp(-t²/2).
      4. Integrate the Gaussian-weighted squared error with torch.trapz.

    Args:
        x:          (N, C) float tensor — N tokens, C features.
        sketch_dim: number of random projections; trades variance for compute.

    Returns:
        Scalar loss.
    """
    N, C = x.size()

    # 1. Random projection: columns are unit-normalised
    A = torch.randn(C, sketch_dim, device=x.device, dtype=x.dtype)
    A = A / (A.norm(p=2, dim=0, keepdim=True) + 1e-6)

    # 2. Integration points on [-5, 5]
    t = torch.linspace(-5, 5, 17, device=x.device, dtype=x.dtype)

    # 3. Theoretical Gaussian CF: φ_gauss(t) = exp(-t²/2)
    exp_f = torch.exp(-0.5 * t ** 2)                   # (T,)

    # 4. Empirical CF
    proj = x @ A                                        # (N, sketch_dim)
    args = proj.unsqueeze(2) * t.view(1, 1, -1)        # (N, sketch_dim, T)
    ecf = torch.exp(1j * args).mean(dim=0)             # (sketch_dim, T)

    # 5. Gaussian-weighted squared L2 distance
    diff_sq = (ecf - exp_f.unsqueeze(0)).abs().square() # (sketch_dim, T)
    err = diff_sq * exp_f.unsqueeze(0)                  # weight by Gaussian envelope

    # 6. Trapezoidal integration over t, then mean over sketch directions
    # ECF is already a sample mean (bounded), so the integral is O(1) — no N scaling.
    loss = torch.trapz(err, t, dim=1)                   # (sketch_dim,)
    return loss.mean()


def sigreg_weak_loss(x: torch.Tensor, sketch_dim: int = 64) -> torch.Tensor:
    """
    Forces Cov(x) ~ Identity.
    Matches the 2nd moment only (Spherical Cloud).

    Algorithm:
      1. Optionally sketch C → sketch_dim when C > sketch_dim.
      2. Centre and estimate the sample covariance.
      3. Return the Frobenius distance to the identity matrix.

    Args:
        x:          (N, C) float tensor — N tokens, C features.
        sketch_dim: random sketch dim used when C > sketch_dim.

    Returns:
        Scalar loss.
    """
    N, C = x.size()

    # 1. Sketch large feature spaces
    if C > sketch_dim:
        S = torch.randn(sketch_dim, C, device=x.device, dtype=x.dtype) / (C ** 0.5)
        x = x @ S.T                                     # (N, sketch_dim)
        sketch_dim = sketch_dim
    else:
        sketch_dim = C

    # 2. Centre and compute sample covariance
    x = x - x.mean(dim=0, keepdim=True)
    cov = (x.T @ x) / (N - 1 + 1e-6)                  # (sketch_dim, sketch_dim)

    # 3. Frobenius distance to identity
    target = torch.eye(sketch_dim, device=x.device, dtype=x.dtype)
    return torch.norm(cov - target, p="fro")


def sigreg_loss(
    x: torch.Tensor,
    loss_type: str = "strong",
    sketch_dim: int = 64,
) -> torch.Tensor:
    """
    Dispatcher for strong / weak / both.

    Args:
        x:         (N, C) tensor.
        loss_type: "strong" | "weak" | "both"
        sketch_dim: sketch dimension forwarded to underlying functions.

    Returns:
        Scalar loss (mean of strong and weak when loss_type="both").
    """
    if loss_type == "strong":
        return sigreg_strong_loss(x, sketch_dim)
    elif loss_type == "weak":
        return sigreg_weak_loss(x, sketch_dim)
    elif loss_type == "both":
        return 0.5 * (sigreg_strong_loss(x, sketch_dim) + sigreg_weak_loss(x, sketch_dim))
    else:
        raise ValueError(f"Unknown loss_type: {loss_type!r}")
