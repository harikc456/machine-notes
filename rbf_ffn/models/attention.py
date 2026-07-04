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
    Polar-coordinates causal self-attention with GQA support.

    Decomposes Q and K into direction (unit vector) and magnitude, computes
    cosine similarity as the base geometric score, then re-weights by the
    outer product of magnitudes scaled by per-head learnable confidence
    parameters q_scale (shape: n_heads) and k_scale (shape: n_kv_heads).

    With GQA (n_kv_heads < n_heads), K and V projections output
    n_kv_heads * head_dim. Polar decomposition runs at n_kv_heads size;
    k_scale is applied before expansion; k_dir, r_k, and v are then
    expanded to n_heads via repeat_interleave before the attention matmul.

    Input/output: (B, N, d_model)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        D, H = cfg.d_model, cfg.n_heads
        assert D % H == 0, f"d_model ({D}) must be divisible by n_heads ({H})"
        self.n_heads = H
        self.head_dim = D // H
        self.n_kv_heads = cfg.n_kv_heads
        self.n_groups = H // self.n_kv_heads
        KV = self.n_kv_heads * self.head_dim
        self.q_proj = nn.Linear(D, D, bias=False)
        self.k_proj = nn.Linear(D, KV, bias=False)
        self.v_proj = nn.Linear(D, KV, bias=False)
        self.o_proj = nn.Linear(D, D, bias=False)
        self.q_scale = nn.Parameter(torch.ones(H))
        self.k_scale = nn.Parameter(torch.ones(self.n_kv_heads))
        self._qkv_silu = cfg.qkv_silu
        _gain_targets = set(cfg.qkv_gain_targets) if cfg.qkv_gain else set()
        if "q" in _gain_targets:
            self.q_gain = nn.Parameter(torch.zeros(H))
        if "k" in _gain_targets:
            self.k_gain = nn.Parameter(torch.zeros(self.n_kv_heads))
        if "v" in _gain_targets:
            self.v_gain = nn.Parameter(torch.zeros(self.n_kv_heads))

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
        k = k_raw.view(B, N, self.n_kv_heads, self.head_dim)
        v = v_raw.view(B, N, self.n_kv_heads, self.head_dim)

        r_q = torch.norm(q, p=2, dim=-1, keepdim=True)   # (B, N, H, 1)
        r_k = torch.norm(k, p=2, dim=-1, keepdim=True)   # (B, N, n_kv_heads, 1)
        q_dir = q / (r_q + 1e-6)
        k_dir = k / (r_k + 1e-6)

        q_dir = q_dir.transpose(1, 2)   # (B, H, N, head_dim)
        k_dir = k_dir.transpose(1, 2)   # (B, n_kv_heads, N, head_dim)
        v     = v.transpose(1, 2)        # (B, n_kv_heads, N, head_dim)
        r_q   = r_q.transpose(1, 2)     # (B, H, N, 1)
        r_k   = r_k.transpose(1, 2)     # (B, n_kv_heads, N, 1)

        if hasattr(self, "q_gain"):
            q_dir = q_dir * (1 + self.q_gain.view(1, -1, 1, 1))
        if hasattr(self, "k_gain"):
            k_dir = k_dir * (1 + self.k_gain.view(1, -1, 1, 1))
        if hasattr(self, "v_gain"):
            v = v * (1 + self.v_gain.view(1, -1, 1, 1))

        # Apply k_scale before expansion (one scalar per KV head)
        r_k = r_k * self.k_scale.view(1, -1, 1, 1)

        # Expand KV tensors to full head count
        k_dir = k_dir.repeat_interleave(self.n_groups, dim=1)   # (B, H, N, head_dim)
        v     = v.repeat_interleave(self.n_groups, dim=1)        # (B, H, N, head_dim)
        r_k   = r_k.repeat_interleave(self.n_groups, dim=1)     # (B, H, N, 1)

        # Cosine similarity: (B, H, N, N)
        attn_weights = torch.matmul(q_dir, k_dir.transpose(-2, -1))

        # Re-weight by magnitude product with per-head confidence scalars
        scale_q = self.q_scale.view(1, -1, 1, 1)                # (1, H, 1, 1)
        attn_weights = attn_weights * (r_q * scale_q) * r_k.transpose(-2, -1)

        # Causal mask
        mask = torch.ones(N, N, device=x.device, dtype=torch.bool).tril()
        attn_weights = attn_weights.masked_fill(~mask, float("-inf"))

        attn_probs = F.softmax(attn_weights, dim=-1)
        out = torch.matmul(attn_probs, v)                        # (B, H, N, head_dim)

        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.o_proj(out)


class ExclusiveSelfAttention(nn.Module):
    """
    Exclusive Self-Attention (XSA) with GQA support.

    Runs causal MHA to produce Y, then projects each output vector onto the
    subspace orthogonal to the normalised value vector:

        Vn = V / ||V||          (per head, per position, post-expansion)
        Z  = Y - (Y · Vn) Vn

    Supports GQA via cfg.n_kv_heads — K and V are projected to
    n_kv_heads * head_dim then expanded before SDPA. The Gram-Schmidt step
    uses the expanded V so each Q head is orthogonalised against its KV group.

    Input/output: (B, N, d_model)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        D, H = cfg.d_model, cfg.n_heads
        assert D % H == 0, f"d_model ({D}) must be divisible by n_heads ({H})"
        self.n_heads = H
        self.head_dim = D // H
        self.n_kv_heads = cfg.n_kv_heads
        self.n_groups = H // self.n_kv_heads
        KV = self.n_kv_heads * self.head_dim
        self.q_proj = nn.Linear(D, D, bias=False)
        self.k_proj = nn.Linear(D, KV, bias=False)
        self.v_proj = nn.Linear(D, KV, bias=False)
        self.o_proj = nn.Linear(D, D, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)
        self._dropout = cfg.dropout
        self._use_flash = _flash_available()
        self._qk_norm = cfg.qk_norm
        self._qkv_silu = cfg.qkv_silu
        if self._qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)
        _gain_targets = set(cfg.qkv_gain_targets) if cfg.qkv_gain else set()
        if "q" in _gain_targets:
            self.q_gain = nn.Parameter(torch.zeros(H))
        if "k" in _gain_targets:
            self.k_gain = nn.Parameter(torch.zeros(self.n_kv_heads))
        if "v" in _gain_targets:
            self.v_gain = nn.Parameter(torch.zeros(self.n_kv_heads))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d_model)"""
        B, N, D = x.shape

        def split_q(t):
            return t.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        def split_kv(t):
            return t.view(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q_raw = self.q_proj(x)
        k_raw = self.k_proj(x)
        v_raw = self.v_proj(x)
        if self._qkv_silu:
            q_raw = F.silu(q_raw)
            k_raw = F.silu(k_raw)
            v_raw = F.silu(v_raw)

        q = self.rope(split_q(q_raw))
        k = self.rope(split_kv(k_raw))
        v = split_kv(v_raw)

        if hasattr(self, "q_gain"):
            q = q * (1 + self.q_gain.view(1, -1, 1, 1))
        if hasattr(self, "k_gain"):
            k = k * (1 + self.k_gain.view(1, -1, 1, 1))
        if hasattr(self, "v_gain"):
            v = v * (1 + self.v_gain.view(1, -1, 1, 1))

        if self._qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        k = k.repeat_interleave(self.n_groups, dim=1)
        v = v.repeat_interleave(self.n_groups, dim=1)

        dp = self._dropout if self.training else 0.0
        if self._use_flash:
            with sdpa_kernel(_FLASH_BACKENDS):
                Y = F.scaled_dot_product_attention(q, k, v, dropout_p=dp, is_causal=True)
        else:
            Y = F.scaled_dot_product_attention(q, k, v, dropout_p=dp, is_causal=True)

        Vn = F.normalize(v, dim=-1)
        Z  = Y - (Y * Vn).sum(dim=-1, keepdim=True) * Vn

        out = Z.transpose(1, 2).contiguous().view(B, N, D)
        return self.o_proj(out)


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with RoPE.

    Supports Grouped Query Attention (GQA) via cfg.n_kv_heads. When
    n_kv_heads < n_heads, K and V projections output n_kv_heads * head_dim
    and are expanded to n_heads via repeat_interleave before SDPA.
    n_kv_heads == n_heads is standard MHA (no overhead).

    Input/output: (B, N, d_model)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        D, H = cfg.d_model, cfg.n_heads
        assert D % H == 0, f"d_model ({D}) must be divisible by n_heads ({H})"
        self.n_heads = H
        self.head_dim = D // H
        self.n_kv_heads = cfg.n_kv_heads
        self.n_groups = H // self.n_kv_heads
        KV = self.n_kv_heads * self.head_dim
        self.q_proj = nn.Linear(D, D, bias=False)
        self.k_proj = nn.Linear(D, KV, bias=False)
        self.v_proj = nn.Linear(D, KV, bias=False)
        self.o_proj = nn.Linear(D, D, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)
        self._dropout = cfg.dropout
        self._use_flash = _flash_available()
        self._qk_norm = cfg.qk_norm
        self._qkv_silu = cfg.qkv_silu
        if self._qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)
        _gain_targets = set(cfg.qkv_gain_targets) if cfg.qkv_gain else set()
        if "q" in _gain_targets:
            self.q_gain = nn.Parameter(torch.zeros(H))
        if "k" in _gain_targets:
            self.k_gain = nn.Parameter(torch.zeros(self.n_kv_heads))
        if "v" in _gain_targets:
            self.v_gain = nn.Parameter(torch.zeros(self.n_kv_heads))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d_model)"""
        B, N, D = x.shape

        def split_q(t):
            return t.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        def split_kv(t):
            return t.view(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = self.rope(split_q(F.silu(self.q_proj(x)) if self._qkv_silu else self.q_proj(x)))
        k = self.rope(split_kv(F.silu(self.k_proj(x)) if self._qkv_silu else self.k_proj(x)))
        v = split_kv(F.silu(self.v_proj(x)) if self._qkv_silu else self.v_proj(x))

        if hasattr(self, "q_gain"):
            q = q * (1 + self.q_gain.view(1, -1, 1, 1))
        if hasattr(self, "k_gain"):
            k = k * (1 + self.k_gain.view(1, -1, 1, 1))
        if hasattr(self, "v_gain"):
            v = v * (1 + self.v_gain.view(1, -1, 1, 1))

        if self._qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        k = k.repeat_interleave(self.n_groups, dim=1)
        v = v.repeat_interleave(self.n_groups, dim=1)

        dp = self._dropout if self.training else 0.0
        if self._use_flash:
            with sdpa_kernel(_FLASH_BACKENDS):
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=dp, is_causal=True)
        else:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dp, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.o_proj(out)


class KVSharedExclusiveSelfAttention(nn.Module):
    """
    Exclusive Self-Attention (XSA) with K = V (shared projection) and GQA support.

    Combines KVSharedAttention's projection scheme (kv_proj outputs
    n_kv_heads * head_dim) with XSA's Gram-Schmidt orthogonalisation step.
    K and V are expanded from n_kv_heads to n_heads before SDPA; the XSA
    step uses the expanded V.

    Input/output: (B, N, d_model)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        D, H = cfg.d_model, cfg.n_heads
        assert D % H == 0, f"d_model ({D}) must be divisible by n_heads ({H})"
        self.n_heads = H
        self.head_dim = D // H
        self.n_kv_heads = cfg.n_kv_heads
        self.n_groups = H // self.n_kv_heads
        KV = self.n_kv_heads * self.head_dim
        self.q_proj = nn.Linear(D, D, bias=False)
        self.kv_proj = nn.Linear(D, KV, bias=False)
        self.o_proj = nn.Linear(D, D, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)
        self._dropout = cfg.dropout
        self._use_flash = _flash_available()
        self._qk_norm = cfg.qk_norm
        self._qkv_silu = cfg.qkv_silu
        if self._qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)
        _gain_targets = set(cfg.qkv_gain_targets) if cfg.qkv_gain else set()
        if "q" in _gain_targets:
            self.q_gain = nn.Parameter(torch.zeros(H))
        if "k" in _gain_targets:
            self.k_gain = nn.Parameter(torch.zeros(self.n_kv_heads))
        if "v" in _gain_targets:
            self.v_gain = nn.Parameter(torch.zeros(self.n_kv_heads))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d_model)"""
        B, N, D = x.shape

        def split_q(t):
            return t.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        def split_kv(t):
            return t.view(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q_raw = self.q_proj(x)
        kv_raw = self.kv_proj(x)
        if self._qkv_silu:
            q_raw = F.silu(q_raw)
            kv_raw = F.silu(kv_raw)

        q = self.rope(split_q(q_raw))
        v = split_kv(kv_raw)
        k = self.rope(v)

        if hasattr(self, "q_gain"):
            q = q * (1 + self.q_gain.view(1, -1, 1, 1))
        if hasattr(self, "k_gain"):
            k = k * (1 + self.k_gain.view(1, -1, 1, 1))
        if hasattr(self, "v_gain"):
            v = v * (1 + self.v_gain.view(1, -1, 1, 1))

        if self._qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        k = k.repeat_interleave(self.n_groups, dim=1)
        v = v.repeat_interleave(self.n_groups, dim=1)

        dp = self._dropout if self.training else 0.0
        if self._use_flash:
            with sdpa_kernel(_FLASH_BACKENDS):
                Y = F.scaled_dot_product_attention(q, k, v, dropout_p=dp, is_causal=True)
        else:
            Y = F.scaled_dot_product_attention(q, k, v, dropout_p=dp, is_causal=True)

        Vn = F.normalize(v, dim=-1)
        Z  = Y - (Y * Vn).sum(dim=-1, keepdim=True) * Vn

        out = Z.transpose(1, 2).contiguous().view(B, N, D)
        return self.o_proj(out)


class KVSharedAttention(nn.Module):
    """
    Multi-head causal self-attention with K = V (shared projection) and GQA support.

    Q is produced by its own projection; K and V both come from a single shared
    `kv_proj` (output size n_kv_heads * head_dim). RoPE is applied to Q and K.
    K and V are expanded from n_kv_heads to n_heads via repeat_interleave before SDPA.

    Input/output: (B, N, d_model)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        D, H = cfg.d_model, cfg.n_heads
        assert D % H == 0, f"d_model ({D}) must be divisible by n_heads ({H})"
        self.n_heads = H
        self.head_dim = D // H
        self.n_kv_heads = cfg.n_kv_heads
        self.n_groups = H // self.n_kv_heads
        KV = self.n_kv_heads * self.head_dim
        self.q_proj = nn.Linear(D, D, bias=False)
        self.kv_proj = nn.Linear(D, KV, bias=False)
        self.o_proj = nn.Linear(D, D, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)
        self._dropout = cfg.dropout
        self._use_flash = _flash_available()
        self._qk_norm = cfg.qk_norm
        self._qkv_silu = cfg.qkv_silu
        if self._qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)
        _gain_targets = set(cfg.qkv_gain_targets) if cfg.qkv_gain else set()
        if "q" in _gain_targets:
            self.q_gain = nn.Parameter(torch.zeros(H))
        if "k" in _gain_targets:
            self.k_gain = nn.Parameter(torch.zeros(self.n_kv_heads))
        if "v" in _gain_targets:
            self.v_gain = nn.Parameter(torch.zeros(self.n_kv_heads))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d_model)"""
        B, N, D = x.shape

        def split_q(t):
            return t.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        def split_kv(t):
            return t.view(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q_raw = self.q_proj(x)
        kv_raw = self.kv_proj(x)
        if self._qkv_silu:
            q_raw = F.silu(q_raw)
            kv_raw = F.silu(kv_raw)

        q = self.rope(split_q(q_raw))
        v = split_kv(kv_raw)
        k = self.rope(v)

        if hasattr(self, "q_gain"):
            q = q * (1 + self.q_gain.view(1, -1, 1, 1))
        if hasattr(self, "k_gain"):
            k = k * (1 + self.k_gain.view(1, -1, 1, 1))
        if hasattr(self, "v_gain"):
            v = v * (1 + self.v_gain.view(1, -1, 1, 1))

        if self._qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        k = k.repeat_interleave(self.n_groups, dim=1)
        v = v.repeat_interleave(self.n_groups, dim=1)

        dp = self._dropout if self.training else 0.0
        if self._use_flash:
            with sdpa_kernel(_FLASH_BACKENDS):
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=dp, is_causal=True)
        else:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dp, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.o_proj(out)


class DeepSeekSparseAttention(nn.Module):
    """
    DeepSeek Sparse Attention (DSA), as introduced in DeepSeek-V3.2.

    A small "lightning indexer" scores every (query, key) pair independently
    of the main attention heads: per-position index vectors q_I, k_I of dim
    cfg.sparse_index_dim are projected with cfg.sparse_index_heads heads,
    dot-producted, passed through ReLU, and summed across indexer heads into
    a single score shared by all attention heads (DeepSeek uses one shared
    top-k set per query rather than per-head sets, so the sparsity pattern
    is consistent across heads). For each query position, only the top
    cfg.sparse_topk keys (causally restricted to positions <= query) are
    kept; the rest are masked out of the main attention.

    The indexer receives NO gradient from the main task loss: `.topk(...).indices`
    is a hard, non-differentiable selection, so index_q_proj/index_k_proj/
    index_weight stay at their random init unless trained separately (DeepSeek
    trains the indexer with an auxiliary KL-distillation loss against the
    dense attention distribution — not implemented here). Without that,
    this module is a fixed random sparsification pattern, useful for
    measuring the cost of sparsity itself but not representative of a
    trained DSA model. Falls back to dense causal attention when
    N <= cfg.sparse_topk (no keys to prune).

    Supports GQA via cfg.n_kv_heads, same as CausalSelfAttention.

    Input/output: (B, N, d_model)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        D, H = cfg.d_model, cfg.n_heads
        assert D % H == 0, f"d_model ({D}) must be divisible by n_heads ({H})"
        self.n_heads = H
        self.head_dim = D // H
        self.n_kv_heads = cfg.n_kv_heads
        self.n_groups = H // self.n_kv_heads
        KV = self.n_kv_heads * self.head_dim
        self.q_proj = nn.Linear(D, D, bias=False)
        self.k_proj = nn.Linear(D, KV, bias=False)
        self.v_proj = nn.Linear(D, KV, bias=False)
        self.o_proj = nn.Linear(D, D, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)
        self._dropout = cfg.dropout
        self._qk_norm = cfg.qk_norm
        self._qkv_silu = cfg.qkv_silu
        if self._qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)
        _gain_targets = set(cfg.qkv_gain_targets) if cfg.qkv_gain else set()
        if "q" in _gain_targets:
            self.q_gain = nn.Parameter(torch.zeros(H))
        if "k" in _gain_targets:
            self.k_gain = nn.Parameter(torch.zeros(self.n_kv_heads))
        if "v" in _gain_targets:
            self.v_gain = nn.Parameter(torch.zeros(self.n_kv_heads))

        # Lightning indexer: shared across attention heads, small dim.
        self.topk = cfg.sparse_topk
        self.index_heads = cfg.sparse_index_heads
        self.index_dim = cfg.sparse_index_dim
        self.index_q_proj = nn.Linear(D, self.index_heads * self.index_dim, bias=False)
        self.index_k_proj = nn.Linear(D, self.index_heads * self.index_dim, bias=False)
        self.index_rope = RotaryEmbedding(self.index_dim)
        self.index_weight = nn.Parameter(torch.ones(self.index_heads))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d_model)"""
        B, N, D = x.shape

        def split_q(t):
            return t.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        def split_kv(t):
            return t.view(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = self.rope(split_q(F.silu(self.q_proj(x)) if self._qkv_silu else self.q_proj(x)))
        k = self.rope(split_kv(F.silu(self.k_proj(x)) if self._qkv_silu else self.k_proj(x)))
        v = split_kv(F.silu(self.v_proj(x)) if self._qkv_silu else self.v_proj(x))

        if hasattr(self, "q_gain"):
            q = q * (1 + self.q_gain.view(1, -1, 1, 1))
        if hasattr(self, "k_gain"):
            k = k * (1 + self.k_gain.view(1, -1, 1, 1))
        if hasattr(self, "v_gain"):
            v = v * (1 + self.v_gain.view(1, -1, 1, 1))

        if self._qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        k = k.repeat_interleave(self.n_groups, dim=1)
        v = v.repeat_interleave(self.n_groups, dim=1)

        causal_mask = torch.ones(N, N, device=x.device, dtype=torch.bool).tril()

        dp = self._dropout if self.training else 0.0
        if N <= self.topk:
            # Nothing to prune; skip the indexer and run dense causal attention.
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dp, is_causal=True)
        else:
            iq = self.index_rope(
                self.index_q_proj(x).view(B, N, self.index_heads, self.index_dim).transpose(1, 2)
            )
            ik = self.index_rope(
                self.index_k_proj(x).view(B, N, self.index_heads, self.index_dim).transpose(1, 2)
            )
            # (B, index_heads, N, N) -> shared (B, N, N) score per key
            idx_scores = F.relu(torch.matmul(iq, ik.transpose(-2, -1)))
            idx_scores = (idx_scores * self.index_weight.view(1, -1, 1, 1)).sum(dim=1)
            idx_scores = idx_scores.masked_fill(~causal_mask, float("-inf"))

            topk_idx = idx_scores.topk(self.topk, dim=-1).indices   # (B, N, topk)
            sparse_mask = torch.zeros(B, N, N, device=x.device, dtype=torch.bool)
            sparse_mask.scatter_(-1, topk_idx, True)
            sparse_mask = sparse_mask & causal_mask.unsqueeze(0)   # re-clip ties selected from -inf rows

            attn_mask = sparse_mask.unsqueeze(1)   # (B, 1, N, N), broadcasts across heads
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dp)

        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.o_proj(out)


ATTN_REGISTRY: dict[str, type] = {
    "standard":         CausalSelfAttention,
    "polar":            PolarAttention,
    "xsa":              ExclusiveSelfAttention,
    "kv_shared":        KVSharedAttention,
    "xsa_kv_shared":    KVSharedExclusiveSelfAttention,
    "deepseek_sparse":  DeepSeekSparseAttention,
}
