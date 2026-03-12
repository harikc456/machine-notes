import torch
import pytest
from rbf_ffn.config import RBFFFNConfig
from rbf_ffn.models.transformer_block import RBFTransformerBlock

D, H, B, N = 16, 4, 2, 10


def make_block(variant: str = "G0") -> RBFTransformerBlock:
    cfg = RBFFFNConfig(d_model=D, n_heads=H, gate_variant=variant)
    return RBFTransformerBlock(cfg)


def test_output_shape():
    block = make_block()
    x = torch.randn(B, N, D)
    out = block(x)
    assert out.shape == (B, N, D)


@pytest.mark.parametrize("variant", ["G0", "G1A", "G1B", "G2"])
def test_all_variants_produce_correct_shape(variant):
    block = make_block(variant)
    x = torch.randn(B, N, D)
    assert block(x).shape == (B, N, D)


def test_gradient_flows_to_input():
    block = make_block()
    x = torch.randn(B, N, D, requires_grad=True)
    block(x).sum().backward()
    assert x.grad is not None


def test_residual_connection_present():
    """With all weights zeroed, output should equal input (residual identity)."""
    block = make_block()
    with torch.no_grad():
        block.ffn.down_proj.weight.zero_()
        block.ffn.down_proj.bias.zero_()
        block.attn.out_proj.weight.zero_()
        block.attn.out_proj.bias.zero_()
    x = torch.randn(B, N, D)
    out = block(x)
    assert torch.allclose(out, x, atol=1e-5)
