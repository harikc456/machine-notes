# rbf_ffn/tests/test_attention.py
import math
import torch
import torch.nn as nn
import pytest
from rbf_ffn.config import ModelConfig
from rbf_ffn.models.attention import RotaryEmbedding, CausalSelfAttention

B, N, D, H = 2, 16, 64, 4   # small dims for fast tests
HEAD_DIM = D // H            # 16


@pytest.fixture
def cfg():
    return ModelConfig(d_model=D, n_heads=H, dropout=0.0)


# ── RotaryEmbedding ───────────────────────────────────────────────────────────

def test_rope_output_shape():
    rope = RotaryEmbedding(head_dim=HEAD_DIM)
    x = torch.randn(B, H, N, HEAD_DIM)
    out = rope(x)
    assert out.shape == (B, H, N, HEAD_DIM)


def test_rope_preserves_norm():
    """RoPE is a rotation; it must preserve the L2 norm of each head vector."""
    rope = RotaryEmbedding(head_dim=HEAD_DIM)
    x = torch.randn(B, H, N, HEAD_DIM)
    out = rope(x)
    norms_in  = x.norm(dim=-1)
    norms_out = out.norm(dim=-1)
    assert torch.allclose(norms_in, norms_out, atol=1e-5)


def test_rope_position_dependent():
    """Two tokens at different positions must get different rotations."""
    rope = RotaryEmbedding(head_dim=HEAD_DIM)
    x = torch.ones(1, 1, 2, HEAD_DIM)   # same vector at positions 0 and 1
    out = rope(x)
    assert not torch.allclose(out[:, :, 0, :], out[:, :, 1, :])


# ── CausalSelfAttention ───────────────────────────────────────────────────────

def test_attn_output_shape(cfg):
    attn = CausalSelfAttention(cfg)
    x = torch.randn(B, N, D)
    assert attn(x).shape == (B, N, D)


def test_attn_no_bias(cfg):
    """All projection weights must have bias=None."""
    attn = CausalSelfAttention(cfg)
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert getattr(attn, name).bias is None, f"{name} has unexpected bias"


def test_attn_causal_mask(cfg):
    """Output at position i must not depend on positions j > i."""
    attn = CausalSelfAttention(cfg)
    attn.eval()
    x = torch.randn(1, N, D)
    out_full = attn(x)

    # Corrupt all tokens after position 0 — position 0 output must be unchanged
    x_corrupt = x.clone()
    x_corrupt[:, 1:, :] = torch.randn_like(x_corrupt[:, 1:, :])
    out_corrupt = attn(x_corrupt)

    assert torch.allclose(out_full[:, 0, :], out_corrupt[:, 0, :], atol=1e-5)


def test_attn_gradient_flows(cfg):
    attn = CausalSelfAttention(cfg)
    x = torch.randn(B, N, D, requires_grad=True)
    attn(x).sum().backward()
    assert x.grad is not None
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert getattr(attn, name).weight.grad is not None


def test_attn_flash_flag_matches_hardware(cfg):
    """_use_flash must be True iff CUDA is present and flash_sdp is enabled."""
    attn = CausalSelfAttention(cfg)
    expected = torch.cuda.is_available() and torch.backends.cuda.flash_sdp_enabled()
    assert attn._use_flash == expected


# ── ATTN_REGISTRY ─────────────────────────────────────────────────────────────

def test_attn_registry_keys():
    from rbf_ffn.models.attention import ATTN_REGISTRY
    assert set(ATTN_REGISTRY.keys()) == {"standard", "polar", "xsa"}


def test_attn_registry_standard_is_causal_self_attention():
    from rbf_ffn.models.attention import ATTN_REGISTRY
    assert ATTN_REGISTRY["standard"] is CausalSelfAttention


def test_attn_registry_instantiates(cfg):
    from rbf_ffn.models.attention import ATTN_REGISTRY
    for key, cls in ATTN_REGISTRY.items():
        attn = cls(cfg)
        x = torch.randn(B, N, D)
        assert attn(x).shape == (B, N, D), f"Registry key '{key}' produced wrong shape"
