# rbf_ffn/models/transformer_block.py
import torch
import torch.nn as nn
from rbf_ffn.config import ModelConfig
from rbf_ffn.models.attention import CausalSelfAttention, ExclusiveSelfAttention, PolarAttention
from rbf_ffn.models.llama_ffn import SwiGLUFFN
from rbf_ffn.models.rational_ffn import RationalFFN, RationalGatedFFN, PFDRationalFFN, PFDRationalGatedFFN, FirstOrderPFDRationalFFN
from rbf_ffn.models.polar_ffn import AdaptivePolarMLP
from rbf_ffn.models.head_mixer import KromHCHeadMixer


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


class ExclusiveAttnBlock(nn.Module):
    """
    Ablation block: ExclusiveSelfAttention + SwiGLU FFN.

    Replaces CausalSelfAttention with ExclusiveSelfAttention, keeping the FFN
    unchanged so that the effect of the XSA mechanism can be isolated.

        x = x + attn(norm1(x))   ← ExclusiveSelfAttention
        x = x + ffn(norm2(x))    ← SwiGLUFFN

    Pre-norm with RMSNorm. No bias anywhere.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.attn  = ExclusiveSelfAttention(cfg)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.ffn   = SwiGLUFFN(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class PolarAttnBlock(nn.Module):
    """
    Ablation block: PolarAttention + SwiGLU FFN.

    Replaces the standard CausalSelfAttention with PolarAttention while
    keeping the SwiGLU FFN unchanged, isolating the effect of the polar
    attention mechanism.

        x = x + attn(norm1(x))   ← attn is PolarAttention
        x = x + ffn(norm2(x))    ← ffn is SwiGLUFFN

    Pre-norm with RMSNorm. No bias anywhere.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.attn  = PolarAttention(cfg)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.ffn   = SwiGLUFFN(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class PolarFullBlock(nn.Module):
    """
    Fully polar transformer block: PolarAttention + AdaptivePolarMLP.

        x = x + attn(norm1(x))   ← PolarAttention
        x = x + ffn(norm2(x))    ← AdaptivePolarMLP

    Pre-norm with RMSNorm. No bias anywhere.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.attn  = PolarAttention(cfg)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.ffn   = AdaptivePolarMLP(cfg)

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
