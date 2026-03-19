"""Transformer blocks: LlamaBlock (standard) and KromHCBlock (with KromHC head mixing)."""
import torch
import torch.nn as nn
from kromhc_transformer.config import KromHCConfig
from kromhc_transformer.models.attention import CausalSelfAttention
from kromhc_transformer.models.head_mixer import KromHCHeadMixer
from kromhc_transformer.models.llama_ffn import SwiGLUFFN


class LlamaBlock(nn.Module):
    """Standard pre-norm transformer block: norm → attn → residual → norm → ffn → residual."""

    def __init__(self, cfg: KromHCConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.ffn = SwiGLUFFN(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class KromHCBlock(nn.Module):
    """Pre-norm transformer block with KromHC head mixing after attention."""

    def __init__(self, cfg: KromHCConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads

        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.head_mixer = KromHCHeadMixer(n_heads=cfg.n_heads, head_dim=self.head_dim)
        self.mixer_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.ffn = SwiGLUFFN(cfg)

    def forward(self, x: torch.Tensor):
        """
        x: (B, N, d_model)
        Returns: (out: (B, N, d_model), H: (B*N, n_heads, n_heads))
        """
        B, N, D = x.shape

        # Standard attention
        attn_out = self.attn(self.norm1(x))  # (B, N, d_model)

        # Reshape to per-token heads, mix, reshape back
        heads = attn_out.reshape(B * N, self.n_heads, self.head_dim)
        mixed_heads, H = self.head_mixer(heads)
        attn_out = self.mixer_proj(mixed_heads.reshape(B, N, D))

        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, H
