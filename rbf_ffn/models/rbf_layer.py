import math
import torch
import torch.nn as nn
import torch.nn.functional as F

_SIGMA_VARIANTS = {"global", "per_center", "per_dim"}


class RBFLayer(nn.Module):
    """
    Element-wise Radial Basis Function expansion.

        φ_k(x_i) = exp( -(x_i - c_k)² / (2σ²) )

    Supports three σ strategies (sigma_variant):
        "global"     — 1 shared scalar (σ-A)
        "per_center" — K scalars, one per center (σ-B)
        "per_dim"    — d_model × K scalars, one per feature per center (σ-C)

    Input:  (B, N, d_model)
    Output: (B, N, d_model * K)
    """

    def __init__(
        self,
        d_model: int,
        centers: list[float],
        sigma_init: float = 0.5,
        sigma_variant: str = "global",
    ):
        super().__init__()
        if sigma_variant not in _SIGMA_VARIANTS:
            raise ValueError(
                f"Unknown sigma_variant '{sigma_variant}'. "
                f"Choose from {sorted(_SIGMA_VARIANTS)}."
            )
        if sigma_init <= 0:
            raise ValueError(f"sigma_init must be positive, got {sigma_init}")
        centers_t = torch.tensor(centers, dtype=torch.float32)
        self.register_buffer("centers", centers_t)   # (K,), frozen
        self.K = len(centers)
        self.d_model = d_model
        self.sigma_variant = sigma_variant

        # σ_raw shape depends on variant; σ = softplus(σ_raw) > 0
        # Note: math.exp overflows for sigma_init >= ~710; practical range is 0.01–5.0
        raw_init = math.log(math.exp(sigma_init) - 1.0)
        if sigma_variant == "global":
            self.sigma_raw = nn.Parameter(torch.tensor(raw_init))
        elif sigma_variant == "per_center":
            self.sigma_raw = nn.Parameter(torch.full((self.K,), raw_init))
        else:  # per_dim
            self.sigma_raw = nn.Parameter(
                torch.full((d_model, self.K), raw_init)
            )

    @property
    def sigma(self) -> torch.Tensor:
        return F.softplus(self.sigma_raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d_model) → (B, N, d_model * K)"""
        sigma = self.sigma

        x_exp = x.unsqueeze(-1)          # (B, N, d_model, 1)
        diff = x_exp - self.centers      # (B, N, d_model, K)

        if self.sigma_variant == "global":
            denom = 2.0 * sigma.pow(2)                        # scalar → broadcasts everywhere
        elif self.sigma_variant == "per_center":
            denom = 2.0 * sigma.pow(2)                        # (K,) → aligns to last dim of (B,N,d_model,K)
        else:  # per_dim: sigma is (d_model, K)
            denom = 2.0 * sigma.pow(2).unsqueeze(0).unsqueeze(0)  # (1,1,d_model,K) → broadcasts over B,N

        rbf = torch.exp(-diff.pow(2) / denom)                 # (B, N, d_model, K)
        B, N, D, K = rbf.shape
        return rbf.reshape(B, N, D * K)
