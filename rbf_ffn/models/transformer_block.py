# rbf_ffn/models/transformer_block.py
import torch
import torch.nn as nn
from rbf_ffn.config import ModelConfig
from rbf_ffn.models.attention import ATTN_REGISTRY
from rbf_ffn.models.llama_ffn import SwiGLUFFN
from rbf_ffn.models.rational_ffn import RationalFFN, RationalGatedFFN, PFDRationalFFN, PFDRationalGatedFFN, FirstOrderPFDRationalFFN
from rbf_ffn.models.polar_ffn import AdaptivePolarMLP
from rbf_ffn.models.head_mixer import KromHCHeadMixer

FFN_REGISTRY: dict[str, type] = {
    "swiglu":                  SwiGLUFFN,
    "rational":                RationalFFN,
    "rationalglu":             RationalGatedFFN,
    "pfd_rational":            PFDRationalFFN,
    "pfd_rationalglu":         PFDRationalGatedFFN,
    "first_order_pfd_rational": FirstOrderPFDRationalFFN,
    "polar":                   AdaptivePolarMLP,
}


class TransformerBlock(nn.Module):
    """
    Composable causal transformer block.

    Builds attention and FFN from registries keyed by cfg.attn_type and
    cfg.ffn_type, so any attention variant can be paired with any FFN variant
    without additional block subclasses.

        x = x + attn(norm1(x))
        x = x + ffn(norm2(x))

    Pre-norm with RMSNorm. No bias anywhere.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.attn_type not in ATTN_REGISTRY:
            raise ValueError(
                f"Unknown attn_type '{cfg.attn_type}'. Valid: {sorted(ATTN_REGISTRY)}"
            )
        if cfg.ffn_type not in FFN_REGISTRY:
            raise ValueError(
                f"Unknown ffn_type '{cfg.ffn_type}'. Valid: {sorted(FFN_REGISTRY)}"
            )
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.attn  = ATTN_REGISTRY[cfg.attn_type](cfg)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.ffn   = FFN_REGISTRY[cfg.ffn_type](cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class KromHCWrapper(nn.Module):
    """
    Wraps any transformer block with KromHC head mixing.

    Applies head mixing as an additive residual after the inner block:

        x_block = inner_block(x)
        heads   = x_block reshaped to (B*N, n_heads, head_dim)
        mixed   = KromHCHeadMixer(heads)
        out     = x_block + mixer_proj(mixed reshaped back)

    Returns (out, H) where H: (B, N, n_heads, n_heads).
    """

    def __init__(self, inner_block: nn.Module, cfg: ModelConfig):
        super().__init__()
        self.inner_block = inner_block
        self.n_heads   = cfg.n_heads
        assert cfg.d_model % cfg.n_heads == 0, (
            f"d_model ({cfg.d_model}) must be divisible by n_heads ({cfg.n_heads})"
        )
        self.head_dim  = cfg.d_model // cfg.n_heads
        self.head_mixer = KromHCHeadMixer(
            n_heads=cfg.n_heads,
            head_dim=self.head_dim,
            d_context=self.head_dim,
            mixer_hidden=cfg.kromhc_mixer_hidden,
        )
        self.mixer_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, N, D)
        Returns: (out: (B, N, D), H: (B, N, n_heads, n_heads))
        """
        x_block = self.inner_block(x)                           # (B, N, D)
        B, N, D = x_block.shape
        heads = x_block.view(B * N, self.n_heads, self.head_dim)
        mixed, H = self.head_mixer(heads)                       # mixed: (B*N, n_heads, head_dim)
        correction = self.mixer_proj(mixed.view(B, N, D))
        H_4d = H.view(B, N, self.n_heads, self.n_heads)
        return x_block + correction, H_4d
