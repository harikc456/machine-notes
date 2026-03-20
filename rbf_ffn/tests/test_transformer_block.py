# rbf_ffn/tests/test_transformer_block.py
import torch
import pytest
from rbf_ffn.config import RBFFFNConfig
from rbf_ffn.models.transformer_block import LlamaBlock, RBFBlock, RationalBlock, RationalGLUBlock, FirstOrderPFDRationalBlock

D, H, B, N = 32, 4, 2, 16


def make_llama(variant: str = "G0") -> LlamaBlock:
    return LlamaBlock(RBFFFNConfig(d_model=D, n_heads=H, gate_variant=variant, dropout=0.0))


def make_rbf(variant: str = "G0") -> RBFBlock:
    return RBFBlock(RBFFFNConfig(d_model=D, n_heads=H, gate_variant=variant, dropout=0.0))


def make_rational() -> RationalBlock:
    return RationalBlock(RBFFFNConfig(d_model=D, n_heads=H, dropout=0.0, model_type="rational"))


# ── LlamaBlock ────────────────────────────────────────────────────────────────

def test_llama_output_shape():
    assert make_llama()(torch.randn(B, N, D)).shape == (B, N, D)


def test_llama_gradient_flows():
    x = torch.randn(B, N, D, requires_grad=True)
    make_llama()(x).sum().backward()
    assert x.grad is not None


def test_llama_residual_connection():
    """Zero FFN and attn output projections → output equals input."""
    block = make_llama()
    with torch.no_grad():
        block.ffn.down_proj.weight.zero_()
        block.attn.o_proj.weight.zero_()
    x = torch.randn(B, N, D)
    assert torch.allclose(block(x), x, atol=1e-5)


# ── RBFBlock ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("variant", ["G0", "G1A", "G1B", "G2"])
def test_rbf_output_shape(variant):
    assert make_rbf(variant)(torch.randn(B, N, D)).shape == (B, N, D)


@pytest.mark.parametrize("variant", ["G0", "G1A", "G1B", "G2"])
def test_rbf_gradient_flows(variant):
    x = torch.randn(B, N, D, requires_grad=True)
    make_rbf(variant)(x).sum().backward()
    assert x.grad is not None


def test_rbf_residual_connection():
    """Zero FFN and attn output projections → output equals input."""
    block = make_rbf()
    with torch.no_grad():
        block.ffn.down_proj.weight.zero_()
        block.attn.o_proj.weight.zero_()
    x = torch.randn(B, N, D)
    assert torch.allclose(block(x), x, atol=1e-5)


def test_rbf_norm1_norm2_are_rmsnorm():
    block = make_rbf()
    assert isinstance(block.norm1, torch.nn.RMSNorm)
    assert isinstance(block.norm2, torch.nn.RMSNorm)


# ── RationalBlock ─────────────────────────────────────────────────────────────

def test_rational_block_shape():
    block = make_rational()
    x = torch.randn(B, N, D)
    assert block(x).shape == (B, N, D)


def test_rational_block_gradient_flow():
    block = make_rational()
    x = torch.randn(B, N, D, requires_grad=True)
    block(x).sum().backward()
    assert x.grad is not None


def test_rational_block_residual_connection():
    block = make_rational()
    with torch.no_grad():
        block.ffn.down_proj.weight.zero_()
        block.attn.o_proj.weight.zero_()
    x = torch.randn(B, N, D)
    assert torch.allclose(block(x), x, atol=1e-5)


# ── RationalGLUBlock ──────────────────────────────────────────────────────────

def make_rationalglu() -> RationalGLUBlock:
    return RationalGLUBlock(RBFFFNConfig(d_model=D, n_heads=H, dropout=0.0, model_type="rationalglu"))


def test_rationalglu_block_shape():
    block = make_rationalglu()
    x = torch.randn(B, N, D)
    assert block(x).shape == (B, N, D)


def test_rationalglu_block_gradient_flow():
    block = make_rationalglu()
    x = torch.randn(B, N, D, requires_grad=True)
    block(x).sum().backward()
    assert x.grad is not None


def test_rationalglu_block_residual_connection():
    block = make_rationalglu()
    with torch.no_grad():
        block.ffn.down_proj.weight.zero_()
        block.attn.o_proj.weight.zero_()
    x = torch.randn(B, N, D)
    assert torch.allclose(block(x), x, atol=1e-5)


# ── FirstOrderPFDRationalBlock ────────────────────────────────────────────────

def make_first_order_pfd_cfg():
    return RBFFFNConfig(d_model=D, n_heads=H, dropout=0.0,
                        model_type="first_order_pfd_rational", pfd_n=4)


def test_first_order_pfd_rational_block_shape():
    block = FirstOrderPFDRationalBlock(make_first_order_pfd_cfg())
    x = torch.randn(B, N, D)
    assert block(x).shape == (B, N, D)


def test_first_order_pfd_rational_block_gradient_flow():
    block = FirstOrderPFDRationalBlock(make_first_order_pfd_cfg())
    x = torch.randn(B, N, D)
    block(x).sum().backward()
    assert block.ffn.phi.grad is not None


def test_first_order_pfd_rational_block_residual_connection():
    """Zero FFN and attn output projections → output equals input."""
    block = FirstOrderPFDRationalBlock(make_first_order_pfd_cfg())
    with torch.no_grad():
        block.ffn.down_proj.weight.zero_()
        block.attn.o_proj.weight.zero_()
    x = torch.randn(B, N, D)
    assert torch.allclose(block(x), x, atol=1e-5)
