# rbf_ffn/tests/test_transformer_block.py
import torch
import pytest
from rbf_ffn.config import ModelConfig
from rbf_ffn.models.transformer_block import FFN_REGISTRY, TransformerBlock
from rbf_ffn.models.attention import ATTN_REGISTRY

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
        "swiglu", "relu_sq", "leaky_relu_sq",
        "rational", "rationalglu",
        "pfd_rational", "pfd_rationalglu", "first_order_pfd_rational",
        "polar", "moe",
    }
    assert set(FFN_REGISTRY.keys()) == expected


def test_ffn_registry_swiglu_is_swiglu_ffn():
    from rbf_ffn.models.llama_ffn import SwiGLUFFN
    assert FFN_REGISTRY["swiglu"] is SwiGLUFFN


# ── TransformerBlock shape and basic behaviour ────────────────────────────────

@pytest.mark.parametrize("attn_type", ["standard", "polar", "xsa", "kv_shared", "xsa_kv_shared"])
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


# ── n_kv_heads config ─────────────────────────────────────────────────────────

def test_n_kv_heads_default_resolves_to_n_heads():
    cfg = ModelConfig(d_model=32, n_heads=4)
    assert cfg.n_kv_heads == 4


def test_n_kv_heads_zero_resolves_to_n_heads():
    cfg = ModelConfig(d_model=32, n_heads=4, n_kv_heads=0)
    assert cfg.n_kv_heads == 4


def test_n_kv_heads_explicit():
    cfg = ModelConfig(d_model=32, n_heads=4, n_kv_heads=2)
    assert cfg.n_kv_heads == 2


def test_n_kv_heads_mqa():
    cfg = ModelConfig(d_model=32, n_heads=4, n_kv_heads=1)
    assert cfg.n_kv_heads == 1


def test_n_kv_heads_indivisible_raises():
    with pytest.raises(ValueError, match="n_kv_heads"):
        ModelConfig(d_model=32, n_heads=4, n_kv_heads=3)


# ── GQA shape tests ───────────────────────────────────────────────────────────

def make_gqa_cfg(attn_type: str) -> ModelConfig:
    return ModelConfig(
        d_model=D, n_heads=H, n_kv_heads=2, dropout=0.0,
        attn_type=attn_type, ffn_type="swiglu",
        ffn_hidden=86, pfd_n=4,
    )


@pytest.mark.parametrize("attn_type", ["standard", "xsa"])
def test_gqa_sdpa_output_shape(attn_type):
    block = TransformerBlock(make_gqa_cfg(attn_type))
    x = torch.randn(B, N, D)
    assert block(x).shape == (B, N, D)


def test_gqa_standard_gradient_flows():
    block = TransformerBlock(make_gqa_cfg("standard"))
    x = torch.randn(B, N, D, requires_grad=True)
    block(x).sum().backward()
    assert x.grad is not None


@pytest.mark.parametrize("attn_type", ["kv_shared", "xsa_kv_shared"])
def test_gqa_kv_shared_output_shape(attn_type):
    block = TransformerBlock(make_gqa_cfg(attn_type))
    x = torch.randn(B, N, D)
    assert block(x).shape == (B, N, D)


@pytest.mark.parametrize("attn_type", ["kv_shared", "xsa_kv_shared"])
def test_gqa_kv_shared_kv_proj_size(attn_type):
    """Verify kv_proj outputs n_kv_heads * head_dim, not d_model."""
    cfg = make_gqa_cfg(attn_type)
    if attn_type == "kv_shared":
        from rbf_ffn.models.attention import KVSharedAttention
        attn = KVSharedAttention(cfg)
    else:
        from rbf_ffn.models.attention import KVSharedExclusiveSelfAttention
        attn = KVSharedExclusiveSelfAttention(cfg)

    expected_kv_output = cfg.n_kv_heads * (cfg.d_model // cfg.n_heads)
    assert attn.kv_proj.out_features == expected_kv_output


def test_gqa_polar_output_shape():
    block = TransformerBlock(make_gqa_cfg("polar"))
    x = torch.randn(B, N, D)
    assert block(x).shape == (B, N, D)


def test_gqa_polar_gradient_flows():
    block = TransformerBlock(make_gqa_cfg("polar"))
    x = torch.randn(B, N, D, requires_grad=True)
    block(x).sum().backward()
    assert x.grad is not None


@pytest.mark.parametrize("attn_type", list(ATTN_REGISTRY.keys()))
def test_gqa_all_registry_variants_shape(attn_type):
    """Every ATTN_REGISTRY entry must forward correctly with n_kv_heads=2."""
    block = TransformerBlock(make_gqa_cfg(attn_type))
    x = torch.randn(B, N, D)
    assert block(x).shape == (B, N, D)


@pytest.mark.parametrize("attn_type", list(ATTN_REGISTRY.keys()))
def test_mqa_all_registry_variants_shape(attn_type):
    """n_kv_heads=1 (MQA) must forward correctly for every variant."""
    cfg = ModelConfig(
        d_model=D, n_heads=H, n_kv_heads=1, dropout=0.0,
        attn_type=attn_type, ffn_type="swiglu",
        ffn_hidden=86, pfd_n=4,
    )
    block = TransformerBlock(cfg)
    x = torch.randn(B, N, D)
    assert block(x).shape == (B, N, D)
