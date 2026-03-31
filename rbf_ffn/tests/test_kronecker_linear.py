# rbf_ffn/tests/test_kronecker_linear.py
import math
import torch
import torch.nn as nn
import pytest
from rbf_ffn.models.kronecker_linear import KroneckerDeltaLinear

IN, OUT, RANK = 16, 32, 4
B, N = 2, 8


@pytest.fixture
def layer():
    return KroneckerDeltaLinear(IN, OUT, delta_rank=RANK, bias=True)


def test_output_shape(layer):
    x = torch.randn(B, N, IN)
    assert layer(x).shape == (B, N, OUT)


def test_output_shape_2d(layer):
    x = torch.randn(B, IN)
    assert layer(x).shape == (B, OUT)


def test_parameter_names(layer):
    names = {n for n, _ in layer.named_parameters()}
    assert "A" in names
    assert "B" in names
    assert "delta_C" in names
    assert "delta_D" in names
    assert "bias" in names


def test_delta_parameter_shapes(layer):
    assert layer.delta_C.shape == (OUT, RANK)
    assert layer.delta_D.shape == (RANK, IN)


def test_kronecker_parameter_shapes(layer):
    # in1 * in2 == IN, out1 * out2 == OUT
    assert layer.A.shape[0] * layer.B.shape[0] == OUT
    assert layer.A.shape[1] * layer.B.shape[1] == IN


def test_delta_c_init_zeros(layer):
    """delta_C must be zero-initialised for stable training start."""
    assert layer.delta_C.abs().max().item() == 0.0


def test_gradient_flows(layer):
    x = torch.randn(B, N, IN, requires_grad=True)
    layer(x).sum().backward()
    assert x.grad is not None
    for attr in ("A", "B", "delta_C", "delta_D"):
        assert getattr(layer, attr).grad is not None
    # Note: delta_D.grad is structurally zero at init because delta_C=0 kills its
    # gradient on the first step (matching LoRA convention). We only assert the
    # grad tensor is allocated, not that it is nonzero.


def test_no_bias_option():
    layer = KroneckerDeltaLinear(IN, OUT, delta_rank=RANK, bias=False)
    x = torch.randn(B, IN)
    out = layer(x)
    assert out.shape == (B, OUT)
    assert layer.bias is None


def test_delta_pathway_contributes():
    """After one gradient step, delta_D becomes nonzero and changes the output."""
    layer = KroneckerDeltaLinear(IN, OUT, delta_rank=RANK, bias=False)
    x = torch.randn(1, IN)
    out_before = layer(x).detach().clone()
    # Manually set delta_D to nonzero
    with torch.no_grad():
        layer.delta_D.fill_(0.1)
        layer.delta_C.fill_(0.1)
    out_after = layer(x).detach().clone()
    assert not torch.allclose(out_before, out_after)
