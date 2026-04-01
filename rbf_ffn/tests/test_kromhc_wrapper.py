# rbf_ffn/tests/test_kromhc_wrapper.py
import torch
import pytest
from rbf_ffn.config import ModelConfig
from rbf_ffn.models.transformer_block import LlamaBlock, KromHCWrapper

D, H, B, N = 32, 4, 2, 16


def make_wrapper() -> KromHCWrapper:
    cfg = ModelConfig(d_model=D, n_heads=H, dropout=0.0)
    inner = LlamaBlock(cfg)
    return KromHCWrapper(inner, cfg)


def test_output_shape():
    wrapper = make_wrapper()
    x = torch.randn(B, N, D)
    out, H_mat = wrapper(x)
    assert out.shape == (B, N, D)


def test_H_shape():
    wrapper = make_wrapper()
    x = torch.randn(B, N, D)
    _, H_mat = wrapper(x)
    assert H_mat.shape == (B, N, H, H)


def test_returns_tuple():
    wrapper = make_wrapper()
    result = wrapper(torch.randn(B, N, D))
    assert isinstance(result, tuple) and len(result) == 2


def test_gradient_flows():
    wrapper = make_wrapper()
    x = torch.randn(B, N, D, requires_grad=True)
    out, _ = wrapper(x)
    out.sum().backward()
    assert x.grad is not None
    assert wrapper.mixer_proj.weight.grad is not None
    for gen in wrapper.head_mixer.weight_gens:
        for p in gen.parameters():
            assert p.grad is not None


def test_zero_mixer_proj_is_identity_of_inner_block():
    """When mixer_proj weights are zero, wrapper output == inner_block output.
    dropout=0.0 is required for the deterministic equality assertion to hold."""
    cfg = ModelConfig(d_model=D, n_heads=H, dropout=0.0)
    inner = LlamaBlock(cfg)
    wrapper = KromHCWrapper(inner, cfg)
    with torch.no_grad():
        wrapper.mixer_proj.weight.zero_()
    x = torch.randn(B, N, D)
    wrapper_out, _ = wrapper(x)
    inner_out = inner(x)
    assert torch.allclose(wrapper_out, inner_out, atol=1e-5)
