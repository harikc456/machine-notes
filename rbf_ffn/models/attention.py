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


class PolarAttention(nn.Module):
    """
    Polar-coordinates causal self-attention.

    Decomposes Q and K into direction (unit vector) and magnitude, computes
    cosine similarity as the base geometric score, then re-weights by the
    outer product of magnitudes scaled by per-head learnable confidence
    parameters q_scale and k_scale.  Causal masking is applied via an
    additive -inf mask before softmax.

    q_scale / k_scale (shape: n_heads) are 1-D and go to AdamW.

    Input/output: (B, N, d_model)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        D, H = cfg.d_model, cfg.n_heads
        assert D % H == 0, f"d_model ({D}) must be divisible by n_heads ({H})"
        self.n_heads  = H
        self.head_dim = D // H
        self.q_proj = nn.Linear(D, D, bias=False)
        self.k_proj = nn.Linear(D, D, bias=False)
        self.v_proj = nn.Linear(D, D, bias=False)
        self.o_proj = nn.Linear(D, D, bias=False)
        # Per-head learnable confidence scalars for the magnitude contribution
        self.q_scale = nn.Parameter(torch.ones(H))
        self.k_scale = nn.Parameter(torch.ones(H))
        self._qkv_silu = cfg.qkv_silu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d_model)"""
        B, N, D = x.shape

        q_raw = self.q_proj(x)
        k_raw = self.k_proj(x)
        v_raw = self.v_proj(x)
        if self._qkv_silu:
            q_raw = F.silu(q_raw)
            k_raw = F.silu(k_raw)
            v_raw = F.silu(v_raw)
        q = q_raw.view(B, N, self.n_heads, self.head_dim)
        k = k_raw.view(B, N, self.n_heads, self.head_dim)
        v = v_raw.view(B, N, self.n_heads, self.head_dim)

        # --- Polar decomposition ---
        r_q = torch.norm(q, p=2, dim=-1, keepdim=True)   # (B, N, H, 1)
        r_k = torch.norm(k, p=2, dim=-1, keepdim=True)
        q_dir = q / (r_q + 1e-6)                          # unit vectors
        k_dir = k / (r_k + 1e-6)

        # Reshape to (B, H, N, .) for batch matmul
        q_dir = q_dir.transpose(1, 2)          # (B, H, N, head_dim)
        k_dir = k_dir.transpose(1, 2)
        v     = v.transpose(1, 2)
        r_q   = r_q.transpose(1, 2)            # (B, H, N, 1)
        r_k   = r_k.transpose(1, 2)

        # Cosine similarity: (B, H, N, N)
        attn_weights = torch.matmul(q_dir, k_dir.transpose(-2, -1))

        # Re-weight by magnitude product with per-head confidence scalars
        scale_q = self.q_scale.view(1, -1, 1, 1)          # (1, H, 1, 1)
        scale_k = self.k_scale.view(1, -1, 1, 1)
        attn_weights = attn_weights * (r_q * scale_q) * (r_k.transpose(-2, -1) * scale_k)

        # Causal mask
        mask = torch.ones(N, N, device=x.device, dtype=torch.bool).tril()
        attn_weights = attn_weights.masked_fill(~mask, float("-inf"))

        attn_probs = F.softmax(attn_weights, dim=-1)
        out = torch.matmul(attn_probs, v)                  # (B, H, N, head_dim)

        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.o_proj(out)


class ExclusiveSelfAttention(nn.Module):
    """
    Exclusive Self-Attention (XSA).

    Runs standard causal multi-head attention to produce Y, then projects each
    output vector onto the subspace orthogonal to the corresponding normalised
    value vector:

        Vn = V / ||V||          (per head, per position)
        Z  = Y - (Y · Vn) Vn   (Gram-Schmidt step)

    The subtraction removes the component of the attention output that lies in
    the direction of the value, so each head's output is "exclusive" of its own
    value direction.  The output projection is then applied to Z.

    Supports RoPE, QK-norm, and optional SiLU on projections — same flags as
    CausalSelfAttention.

    Input/output: (B, N, d_model)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        D, H = cfg.d_model, cfg.n_heads
        assert D % H == 0, f"d_model ({D}) must be divisible by n_heads ({H})"
        self.n_heads  = H
        self.head_dim = D // H
        self.q_proj = nn.Linear(D, D, bias=False)
        self.k_proj = nn.Linear(D, D, bias=False)
        self.v_proj = nn.Linear(D, D, bias=False)
        self.o_proj = nn.Linear(D, D, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)
        self._dropout  = cfg.dropout
        self._use_flash = _flash_available()
        self._qk_norm  = cfg.qk_norm
        self._qkv_silu = cfg.qkv_silu
        if self._qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d_model)"""
        B, N, D = x.shape

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        q_raw = self.q_proj(x)
        k_raw = self.k_proj(x)
        v_raw = self.v_proj(x)
        if self._qkv_silu:
            q_raw = F.silu(q_raw)
            k_raw = F.silu(k_raw)
            v_raw = F.silu(v_raw)

        q = self.rope(split_heads(q_raw))   # (B, H, N, head_dim)
        k = self.rope(split_heads(k_raw))
        v = split_heads(v_raw)

        if self._qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        dp = self._dropout if self.training else 0.0
        if self._use_flash:
            with sdpa_kernel(_FLASH_BACKENDS):
                Y = F.scaled_dot_product_attention(q, k, v, dropout_p=dp, is_causal=True)
        else:
            Y = F.scaled_dot_product_attention(q, k, v, dropout_p=dp, is_causal=True)

        # XSA: subtract the component of Y along the normalised value direction
        Vn = F.normalize(v, dim=-1)                                  # (B, H, N, head_dim)
        Z  = Y - (Y * Vn).sum(dim=-1, keepdim=True) * Vn            # (B, H, N, head_dim)

        out = Z.transpose(1, 2).contiguous().view(B, N, D)
        return self.o_proj(out)


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
        self._qkv_silu = cfg.qkv_silu
        if self._qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d_model)"""
        B, N, D = x.shape
        def split_heads(t):
            return t.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        q = self.rope(split_heads(F.silu(self.q_proj(x)) if self._qkv_silu else self.q_proj(x)))   # (B, H, N, head_dim)
        k = self.rope(split_heads(F.silu(self.k_proj(x)) if self._qkv_silu else self.k_proj(x)))
        v = split_heads(F.silu(self.v_proj(x)) if self._qkv_silu else self.v_proj(x))

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


ATTN_REGISTRY: dict[str, type] = {
    "standard": CausalSelfAttention,
    "polar":    PolarAttention,
    "xsa":      ExclusiveSelfAttention,
}
