import pytest
import torch
from kromhc_transformer.models.attention import CausalSelfAttention, RotaryEmbedding
from kromhc_transformer.config import KromHCConfig

def test_rotary_embedding_shape():
    rope = RotaryEmbedding(head_dim=64)
    x = torch.randn(2, 8, 512, 64)  # (B, H, N, head_dim)
    out = rope(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype

def test_rotary_embedding_is_rotation():
    """RoPE should preserve tensor norm."""
    rope = RotaryEmbedding(head_dim=64)
    x = torch.randn(1, 1, 16, 64)
    out = rope(x)
    # Norms should be approximately equal (rotation preserves norms)
    assert torch.allclose(x.norm(dim=-1), out.norm(dim=-1), atol=1e-5)

def test_causal_self_attention_shape():
    cfg = KromHCConfig(d_model=256, n_heads=8, dropout=0.0)
    attn = CausalSelfAttention(cfg)
    x = torch.randn(2, 512, 256)
    out = attn(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype

def test_causal_self_attention_qk_norm_enabled():
    cfg = KromHCConfig(d_model=256, n_heads=8, qk_norm=True, dropout=0.0)
    attn = CausalSelfAttention(cfg)
    assert hasattr(attn, 'q_norm')
    assert hasattr(attn, 'k_norm')
    x = torch.randn(2, 128, 256)
    out = attn(x)
    assert out.shape == x.shape

def test_causal_self_attention_qk_norm_disabled():
    cfg = KromHCConfig(d_model=256, n_heads=8, qk_norm=False, dropout=0.0)
    attn = CausalSelfAttention(cfg)
    assert not hasattr(attn, 'q_norm')
    x = torch.randn(2, 128, 256)
    out = attn(x)
    assert out.shape == x.shape

def test_causal_self_attention_no_bias():
    cfg = KromHCConfig(d_model=256, n_heads=8, dropout=0.0)
    attn = CausalSelfAttention(cfg)
    for name, param in attn.named_parameters():
        if 'bias' in name:
            assert False, f"Found bias parameter: {name}"

def test_causal_self_attention_gradient_flow():
    cfg = KromHCConfig(d_model=64, n_heads=4, dropout=0.0)
    attn = CausalSelfAttention(cfg)
    x = torch.randn(2, 32, 64, requires_grad=True)
    out = attn(x)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().max() > 0
