# rbf_ffn/models/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from rbf_ffn.config import ModelConfig

# Backend preference order: FlashAttention → MemEfficient → Math fallback.
# PyTorch tries each in order and picks the first that is supported for the
# given dtype/device/sequence-length at runtime.
_FLASH_BACKENDS = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


def _flash_available() -> bool:
    """Return True if the FlashAttention SDPA backend is globally enabled on CUDA."""
    return torch.cuda.is_available() and torch.backends.cuda.flash_sdp_enabled()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension: [x1, x2] → [-x2, x1]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Applies position-dependent rotations to Q and K tensors.
    No learnable parameters; sin/cos cache is built lazily on first call.

    Input/output: (B, n_heads, N, head_dim)
    """

    def __init__(self, head_dim: int, base: int = 10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cos: torch.Tensor | None = None
        self._sin: torch.Tensor | None = None
        self._cached_len: int = 0

    def _build_cache(self, seq_len: int, device: torch.device) -> None:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)          # (N, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)        # (N, head_dim)
        self._cos = emb.cos()                          # (N, head_dim)
        self._sin = emb.sin()
        self._cached_len = seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_heads, N, head_dim)"""
        seq_len = x.shape[2]
        if self._cos is None or seq_len > self._cached_len:
            self._build_cache(seq_len, x.device)
        cos = self._cos[:seq_len].unsqueeze(0).unsqueeze(0)   # (1, 1, N, head_dim)
        sin = self._sin[:seq_len].unsqueeze(0).unsqueeze(0)
        return (x * cos) + (_rotate_half(x) * sin)


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with RoPE.

    No bias on any projection. Causal mask via F.scaled_dot_product_attention
    with is_causal=True (no explicit mask tensor stored).

    On CUDA when FlashAttention is available, uses sdp_kernel to explicitly
    prefer the FlashAttention backend with graceful fallback to MemEfficient
    and Math backends. On CPU, delegates backend selection to PyTorch.

    Input/output: (B, N, d_model)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        D, H = cfg.d_model, cfg.n_heads
        assert D % H == 0, f"d_model ({D}) must be divisible by n_heads ({H})"
        self.n_heads = H
        self.head_dim = D // H
        self.q_proj = nn.Linear(D, D, bias=False)
        self.k_proj = nn.Linear(D, D, bias=False)
        self.v_proj = nn.Linear(D, D, bias=False)
        self.o_proj = nn.Linear(D, D, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)
        self._dropout = cfg.dropout
        self._use_flash = _flash_available()
        self._qk_norm = cfg.qk_norm
        if self._qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d_model)"""
        B, N, D = x.shape
        def split_heads(t):
            return t.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        q = self.rope(split_heads(self.q_proj(x)))   # (B, H, N, head_dim)
        k = self.rope(split_heads(self.k_proj(x)))
        v = split_heads(self.v_proj(x))

        # Apply QK normalization if enabled
        if self._qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        dp = self._dropout if self.training else 0.0
        if self._use_flash:
            with sdpa_kernel(_FLASH_BACKENDS):
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=dp, is_causal=True)
        else:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dp, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.o_proj(out)
