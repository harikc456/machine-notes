import torch
import torch.nn as nn
from rbf_ffn.config import RBFFFNConfig
from rbf_ffn.models.rbf_ffn import RBFFFN


class RBFTransformerBlock(nn.Module):
    """
    Standard pre-norm transformer block with RBF-FFN replacing the MLP.

        x = x + attn(norm1(x))
        x = x + ffn(norm2(x))        ← ffn is RBFFFN

    Double LayerNorm is intentional (see spec): norm2 is the standard pre-block
    norm; RBFFFN applies a second internal LayerNorm with its own learnable (γ, β)
    to decouple the RBF input distribution from the attention sublayer output.
    Do NOT remove either norm.
    """

    def __init__(self, cfg: RBFFFNConfig):
        super().__init__()
        D, H = cfg.d_model, cfg.n_heads
        self.norm1 = nn.LayerNorm(D)
        self.attn = nn.MultiheadAttention(
            embed_dim=D,
            num_heads=H,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(D)
        self.ffn = RBFFFN(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d_model)"""
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x
