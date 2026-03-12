# rbf_ffn/tests/test_llama_ffn.py
import torch
import pytest
from rbf_ffn.config import RBFFFNConfig
from rbf_ffn.models.llama_ffn import SwiGLUFFN

B, N, D = 2, 10, 64
FFN_HIDDEN = 172  # 8/3 * 64 ≈ 170, rounded to nearest even


@pytest.fixture
def ffn():
    cfg = RBFFFNConfig(d_model=D, ffn_hidden=FFN_HIDDEN)
    return SwiGLUFFN(cfg)


def test_output_shape(ffn):
    x = torch.randn(B, N, D)
    assert ffn(x).shape == (B, N, D)


def test_no_bias_on_any_projection(ffn):
    for name in ("gate_proj", "up_proj", "down_proj"):
        assert getattr(ffn, name).bias is None, f"{name} has unexpected bias"


def test_gradient_flows(ffn):
    x = torch.randn(B, N, D, requires_grad=True)
    ffn(x).sum().backward()
    assert x.grad is not None
    for name in ("gate_proj", "up_proj", "down_proj"):
        assert getattr(ffn, name).weight.grad is not None


def test_projection_shapes(ffn):
    assert ffn.gate_proj.in_features  == D
    assert ffn.gate_proj.out_features == FFN_HIDDEN
    assert ffn.up_proj.in_features    == D
    assert ffn.up_proj.out_features   == FFN_HIDDEN
    assert ffn.down_proj.in_features  == FFN_HIDDEN
    assert ffn.down_proj.out_features == D
