import math
import torch
import torch.nn as nn
from rbf_ffn.config import RBFFFNConfig
from rbf_ffn.models.rbf_layer import RBFLayer
from rbf_ffn.models.gate_layer import G0Gate, G1AGate, G1BGate, G2SinkhornGate


class RBFFFN(nn.Module):
    """
    Drop-in FFN replacement:
        LayerNorm → RBF → Gate → Down Projection

    Gate variant selected via cfg.gate_variant:
        G0  : element-wise self-gate (w, b learnable)
        G1A : cross-kernel mixing gate from RBF output
        G1B : input-driven gate from pre-RBF x
        G2  : Sinkhorn aggregation — collapses K, no learnable params

    For G0/G1A/G1B: down_proj is d_model*K → d_model
    For G2:         down_proj is d_model   → d_model

    Double LayerNorm is intentional: the internal norm (self.norm) has its own
    learnable (γ, β) to decouple RBF input from attention sublayer output.
    """

    _VARIANTS = {"G0", "G1A", "G1B", "G2"}

    def __init__(self, cfg: RBFFFNConfig):
        super().__init__()
        if cfg.gate_variant not in self._VARIANTS:
            raise ValueError(
                f"Unknown gate_variant '{cfg.gate_variant}'. "
                f"Choose from {sorted(self._VARIANTS)}."
            )

        self.gate_variant = cfg.gate_variant
        D, K = cfg.d_model, cfg.K

        self.norm = nn.RMSNorm(D)
        self.rbf = RBFLayer(D, cfg.centers, cfg.sigma_init, cfg.sigma_variant)

        if cfg.gate_variant == "G0":
            self.gate = G0Gate(D, K)
        elif cfg.gate_variant == "G1A":
            self.gate = G1AGate(D, K)
        elif cfg.gate_variant == "G1B":
            self.gate = G1BGate(D, K)
        else:  # G2
            self.gate = G2SinkhornGate(D, K, cfg.sinkhorn_iters)

        down_in = D * K if cfg.gate_variant != "G2" else D
        self.down_proj = nn.Linear(down_in, D, bias=False)
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, d_model) — may be raw or pre-normalized; this module applies
        its own internal LayerNorm first. When called from RBFTransformerBlock,
        x is already norm2(x_post_attn), resulting in double normalization
        (intentional — see class docstring).
        """
        x_norm = self.norm(x)                      # (B, N, D)
        rbf_out = self.rbf(x_norm)                 # (B, N, D*K)

        if self.gate_variant == "G1B":
            gated = self.gate(rbf_out, x_norm)     # G1B needs pre-RBF x
        else:
            gated = self.gate(rbf_out)             # (B, N, D*K) or (B, N, D) for G2

        return self.down_proj(gated)               # (B, N, D)
