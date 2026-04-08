# sigreg/models/block.py
"""
Transformer block with optional residual connections and normalisation layers.

Architecture is controlled by cfg.use_residual and cfg.norm_type:
    use_residual=False, norm_type="none"  →  x = ffn(attn(x))
    use_residual=True,  norm_type="none"  →  x = x + attn(x); x = x + ffn(x)
    use_residual=False, norm_type=*       →  x = ffn(attn(norm(x)))
    use_residual=True,  norm_type=*       →  x = x + attn(norm(x)); x = x + ffn(norm(x))

See TransformerBlock for full details.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from sigreg.config import SIGRegConfig


# ── Attention ─────────────────────────────────────────────────────────────────

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: int = 10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cos: torch.Tensor | None = None
        self._sin: torch.Tensor | None = None
        self._cached_len: int = 0

    def _build_cache(self, seq_len: int, device: torch.device) -> None:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self._cos = emb.cos()
        self._sin = emb.sin()
        self._cached_len = seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_heads, N, head_dim)"""
        seq_len = x.shape[2]
        if self._cos is None or seq_len > self._cached_len:
            self._build_cache(seq_len, x.device)
        cos = self._cos[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self._sin[:seq_len].unsqueeze(0).unsqueeze(0)
        return (x * cos) + (_rotate_half(x) * sin)


_SDPA_BACKENDS = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with RoPE positional encoding.

    No LayerNorm or residual — those live outside in the block.
    Optional QK normalisation via F.normalize for stability.
    """

    def __init__(self, cfg: SIGRegConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qk_norm = cfg.qk_norm
        self.dropout = cfg.dropout

        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj  = nn.Linear(cfg.d_model, cfg.d_model,     bias=False)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d_model)"""
        B, N, D = x.shape
        qkv = self.qkv_proj(x)                                       # (B, N, 3D)
        q, k, v = qkv.split(D, dim=-1)

        # (B, n_heads, N, head_dim)
        def reshape(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)
        q = self.rope(q)
        k = self.rope(k)

        if self.qk_norm:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)

        p_drop = self.dropout if self.training else 0.0
        with sdpa_kernel(_SDPA_BACKENDS):
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=p_drop, is_causal=True,
            )

        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)


# ── FFN ───────────────────────────────────────────────────────────────────────

class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network (no bias, no norm)."""

    def __init__(self, cfg: SIGRegConfig):
        super().__init__()
        D, H = cfg.d_model, cfg.ffn_hidden
        self.gate_proj = nn.Linear(D, H, bias=False)
        self.up_proj   = nn.Linear(D, H, bias=False)
        self.down_proj = nn.Linear(H, D, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ── Block ─────────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    Transformer block with optional residual connections and pre-norm.

    Controlled by cfg.use_residual and cfg.norm_type:

        use_residual=False, norm_type="none"  → x = ffn(attn(x))
        use_residual=True,  norm_type="none"  → x = x + attn(x); x = x + ffn(x)
        use_residual=False, norm_type=*       → x = ffn(attn(norm_attn(x)))
        use_residual=True,  norm_type=*       → x = x + attn(norm_attn(x)); x = x + ffn(norm_ffn(x))

    When both use_residual and a norm are active, pre-norm ordering is used.
    """

    def __init__(self, cfg: SIGRegConfig):
        super().__init__()
        self.attn = CausalSelfAttention(cfg)
        self.ffn  = SwiGLUFFN(cfg)
        self.use_residual = cfg.use_residual

        if cfg.norm_type == "rmsnorm":
            self.norm_attn: nn.Module | None = nn.RMSNorm(cfg.d_model)
            self.norm_ffn:  nn.Module | None = nn.RMSNorm(cfg.d_model)
        elif cfg.norm_type == "layernorm":
            self.norm_attn = nn.LayerNorm(cfg.d_model)
            self.norm_ffn  = nn.LayerNorm(cfg.d_model)
        else:
            self.norm_attn = None
            self.norm_ffn  = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d_model) → (B, N, d_model)"""
        attn_in = self.norm_attn(x) if self.norm_attn is not None else x
        attn_out = self.attn(attn_in)
        x = x + attn_out if self.use_residual else attn_out

        ffn_in = self.norm_ffn(x) if self.norm_ffn is not None else x
        ffn_out = self.ffn(ffn_in)
        x = x + ffn_out if self.use_residual else ffn_out

        return x
