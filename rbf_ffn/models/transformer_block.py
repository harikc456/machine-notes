# rbf_ffn/models/transformer_block.py
import torch
import torch.nn as nn
from rbf_ffn.config import RBFFFNConfig
from rbf_ffn.models.attention import CausalSelfAttention
from rbf_ffn.models.llama_ffn import SwiGLUFFN
from rbf_ffn.models.rbf_ffn import RBFFFN


class LlamaBlock(nn.Module):
    """
    Llama-style transformer block with SwiGLU FFN.

        x = x + attn(norm1(x))
        x = x + ffn(norm2(x))

    Pre-norm with RMSNorm. No bias anywhere.
    """

    def __init__(self, cfg: RBFFFNConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.attn  = CausalSelfAttention(cfg)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.ffn   = SwiGLUFFN(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class RBFBlock(nn.Module):
    """
    Transformer block with RBF-FFN replacing the MLP.

        x = x + attn(norm1(x))
        x = x + ffn(norm2(x))    ← ffn is RBFFFN

    Double normalisation: norm2 (outer, this block) + RBFFFN.norm (inner).
    Both are intentional — see spec. Do NOT remove either.
    """

    def __init__(self, cfg: RBFFFNConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.attn  = CausalSelfAttention(cfg)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.ffn   = RBFFFN(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
