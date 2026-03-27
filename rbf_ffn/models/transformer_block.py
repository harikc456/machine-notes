# rbf_ffn/models/transformer_block.py
import torch
import torch.nn as nn
from rbf_ffn.config import ModelConfig
from rbf_ffn.models.attention import CausalSelfAttention
from rbf_ffn.models.llama_ffn import SwiGLUFFN
from rbf_ffn.models.rational_ffn import RationalFFN, RationalGatedFFN, PFDRationalFFN, PFDRationalGatedFFN, FirstOrderPFDRationalFFN
from rbf_ffn.models.polar_ffn import AdaptivePolarMLP


class LlamaBlock(nn.Module):
    """
    Llama-style transformer block with SwiGLU FFN.

        x = x + attn(norm1(x))
        x = x + ffn(norm2(x))

    Pre-norm with RMSNorm. No bias anywhere.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.attn  = CausalSelfAttention(cfg)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.ffn   = SwiGLUFFN(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class RationalBlock(nn.Module):
    """
    Transformer block with RationalFFN replacing the MLP.

        x = x + attn(norm1(x))
        x = x + ffn(norm2(x))    ← ffn is RationalFFN

    Pre-norm with RMSNorm. No bias anywhere.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.attn  = CausalSelfAttention(cfg)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.ffn   = RationalFFN(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class RationalGLUBlock(nn.Module):
    """
    Transformer block with RationalGatedFFN replacing the MLP.

        x = x + attn(norm1(x))
        x = x + ffn(norm2(x))    ← ffn is RationalGatedFFN

    Pre-norm with RMSNorm. No bias anywhere.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.attn  = CausalSelfAttention(cfg)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.ffn   = RationalGatedFFN(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class PFDRationalBlock(nn.Module):
    """
    Transformer block with PFDRationalFFN replacing the MLP.

        x = x + attn(norm1(x))
        x = x + ffn(norm2(x))    ← ffn is PFDRationalFFN

    Pre-norm with RMSNorm. No bias anywhere.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.attn  = CausalSelfAttention(cfg)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.ffn   = PFDRationalFFN(cfg, n=cfg.pfd_n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class PFDRationalGLUBlock(nn.Module):
    """
    Transformer block with PFDRationalGatedFFN replacing the MLP.

        x = x + attn(norm1(x))
        x = x + ffn(norm2(x))    ← ffn is PFDRationalGatedFFN

    Pre-norm with RMSNorm. No bias anywhere.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.attn  = CausalSelfAttention(cfg)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.ffn   = PFDRationalGatedFFN(cfg, n=cfg.pfd_n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class FirstOrderPFDRationalBlock(nn.Module):
    """
    Transformer block with FirstOrderPFDRationalFFN replacing the MLP.

        x = x + attn(norm1(x))
        x = x + ffn(norm2(x))    ← ffn is FirstOrderPFDRationalFFN

    Pre-norm with RMSNorm. No bias anywhere.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.attn  = CausalSelfAttention(cfg)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.ffn   = FirstOrderPFDRationalFFN(cfg, n=cfg.pfd_n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class PolarMLPBlock(nn.Module):
    """
    Transformer block with AdaptivePolarMLP replacing the FFN.

        x = x + attn(norm1(x))
        x = x + ffn(norm2(x))    ← ffn is AdaptivePolarMLP

    Pre-norm with RMSNorm. No bias anywhere.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.attn  = CausalSelfAttention(cfg)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.ffn   = AdaptivePolarMLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
