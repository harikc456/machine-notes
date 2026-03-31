# rbf_ffn/tests/test_llama_ffn.py
import torch
import pytest
from rbf_ffn.config import ModelConfig
from rbf_ffn.models.llama_ffn import SwiGLUFFN

B, N, D = 2, 10, 64
FFN_HIDDEN = 172  # 8/3 * 64 ≈ 170, rounded to nearest even


@pytest.fixture
def ffn():
    cfg = ModelConfig(d_model=D, ffn_hidden=FFN_HIDDEN)
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


import torch.nn as nn
from rbf_ffn.models.kronecker_linear import KroneckerDeltaLinear

DELTA_RANK = 4


@pytest.fixture
def delta_ffn():
    cfg = ModelConfig(d_model=D, ffn_hidden=FFN_HIDDEN,
                      kronecker_delta_mlp=True, kronecker_delta_rank=DELTA_RANK)
    return SwiGLUFFN(cfg)


def test_delta_output_shape(delta_ffn):
    x = torch.randn(B, N, D)
    assert delta_ffn(x).shape == (B, N, D)


def test_delta_gate_proj_is_linear(delta_ffn):
    """gate_proj must stay nn.Linear when kronecker_delta_mlp=True."""
    assert type(delta_ffn.gate_proj) is nn.Linear


def test_delta_up_proj_is_kronecker_delta(delta_ffn):
    assert isinstance(delta_ffn.up_proj, KroneckerDeltaLinear)


def test_delta_down_proj_is_kronecker_delta(delta_ffn):
    assert isinstance(delta_ffn.down_proj, KroneckerDeltaLinear)


def test_delta_rank_propagated(delta_ffn):
    assert delta_ffn.up_proj.delta_rank == DELTA_RANK
    assert delta_ffn.down_proj.delta_rank == DELTA_RANK


def test_delta_gradient_flows(delta_ffn):
    x = torch.randn(B, N, D, requires_grad=True)
    delta_ffn(x).sum().backward()
    assert x.grad is not None
    assert delta_ffn.gate_proj.weight.grad is not None
    assert delta_ffn.up_proj.A.grad is not None
    assert delta_ffn.down_proj.A.grad is not None


def test_kronecker_delta_takes_precedence_over_kronecker_mlp():
    """If both flags are True, kronecker_delta_mlp wins."""
    cfg = ModelConfig(d_model=D, ffn_hidden=FFN_HIDDEN,
                      kronecker_mlp=True, kronecker_delta_mlp=True,
                      kronecker_delta_rank=DELTA_RANK)
    ffn = SwiGLUFFN(cfg)
    assert isinstance(ffn.up_proj, KroneckerDeltaLinear)
    assert isinstance(ffn.down_proj, KroneckerDeltaLinear)
    assert type(ffn.gate_proj) is nn.Linear
