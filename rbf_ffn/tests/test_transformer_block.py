# rbf_ffn/tests/test_transformer_block.py
import torch
import pytest
from rbf_ffn.config import ModelConfig
from rbf_ffn.models.transformer_block import FFN_REGISTRY, TransformerBlock

D, H, B, N = 32, 4, 2, 16


def make_cfg(attn_type: str = "standard", ffn_type: str = "swiglu") -> ModelConfig:
    return ModelConfig(
        d_model=D, n_heads=H, dropout=0.0,
        attn_type=attn_type, ffn_type=ffn_type,
        ffn_hidden=86, pfd_n=4,
    )


# ── FFN_REGISTRY ──────────────────────────────────────────────────────────────

def test_ffn_registry_keys():
    expected = {
        "swiglu", "rational", "rationalglu",
        "pfd_rational", "pfd_rationalglu", "first_order_pfd_rational", "polar",
    }
    assert set(FFN_REGISTRY.keys()) == expected


def test_ffn_registry_swiglu_is_swiglu_ffn():
    from rbf_ffn.models.llama_ffn import SwiGLUFFN
    assert FFN_REGISTRY["swiglu"] is SwiGLUFFN


# ── TransformerBlock shape and basic behaviour ────────────────────────────────

@pytest.mark.parametrize("attn_type", ["standard", "polar", "xsa"])
@pytest.mark.parametrize("ffn_type", ["swiglu", "rational", "rationalglu", "pfd_rational", "pfd_rationalglu", "first_order_pfd_rational", "polar"])
def test_transformer_block_output_shape(attn_type, ffn_type):
    block = TransformerBlock(make_cfg(attn_type=attn_type, ffn_type=ffn_type))
    x = torch.randn(B, N, D)
    assert block(x).shape == (B, N, D)


def test_transformer_block_gradient_flows():
    block = TransformerBlock(make_cfg())
    x = torch.randn(B, N, D, requires_grad=True)
    block(x).sum().backward()
    assert x.grad is not None


def test_transformer_block_residual_connection():
    """Zero out o_proj and down_proj → output equals input."""
    block = TransformerBlock(make_cfg())
    with torch.no_grad():
        block.ffn.down_proj.weight.zero_()
        block.attn.o_proj.weight.zero_()
    x = torch.randn(B, N, D)
    assert torch.allclose(block(x), x, atol=1e-5)


def test_transformer_block_has_attn_and_ffn_attrs():
    block = TransformerBlock(make_cfg())
    assert hasattr(block, "attn")
    assert hasattr(block, "ffn")
    assert hasattr(block, "norm1")
    assert hasattr(block, "norm2")


def test_transformer_block_pfd_rational_gradient_flow():
    block = TransformerBlock(make_cfg(ffn_type="first_order_pfd_rational"))
    x = torch.randn(B, N, D)
    block(x).sum().backward()
    assert block.ffn.phi.grad is not None


def test_transformer_block_rational_residual():
    block = TransformerBlock(make_cfg(ffn_type="rational"))
    with torch.no_grad():
        block.ffn.down_proj.weight.zero_()
        block.attn.o_proj.weight.zero_()
    x = torch.randn(B, N, D)
    assert torch.allclose(block(x), x, atol=1e-5)
