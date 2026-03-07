import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest
from geometry.lorentz import project_to_hyperboloid, lorentz_inner
from models.lorentz_attention import LorentzMultiheadAttention

MANIFOLD_EPS = 1e-3


def on_hyperboloid(x):
    inner = lorentz_inner(x.reshape(-1, x.shape[-1]), x.reshape(-1, x.shape[-1]))
    return torch.allclose(inner, torch.full_like(inner, -1.0), atol=MANIFOLD_EPS)


def make_hyperbolic_input(B, N, d_model):
    x_s = torch.randn(B, N, d_model) * 0.3
    return project_to_hyperboloid(x_s)   # (B, N, d_model+1)


def test_output_shape():
    attn = LorentzMultiheadAttention(d_model=64, n_heads=4)
    x = make_hyperbolic_input(2, 16, 64)
    out = attn(x)
    assert out.shape == x.shape, f"expected {x.shape}, got {out.shape}"


def test_output_on_manifold():
    attn = LorentzMultiheadAttention(d_model=64, n_heads=4)
    x = make_hyperbolic_input(2, 16, 64)
    out = attn(x)
    assert on_hyperboloid(out), "attention output must lie on hyperboloid"


def test_attention_weights_sum_to_one():
    attn = LorentzMultiheadAttention(d_model=32, n_heads=2, return_attn_weights=True)
    attn.eval()               # disable dropout so weights are unscaled
    x = make_hyperbolic_input(2, 8, 32)
    _, weights = attn(x)     # weights: (B, n_heads, N, N)
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_no_nan_in_output():
    attn = LorentzMultiheadAttention(d_model=64, n_heads=4)
    x = make_hyperbolic_input(4, 32, 64)
    out = attn(x)
    assert not torch.isnan(out).any(), "NaN detected in attention output"
