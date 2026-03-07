import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from geometry.lorentz import project_to_hyperboloid, lorentz_inner
from models.lorentz_block import LorentzTransformerBlock

MANIFOLD_EPS = 1e-3


def on_hyperboloid(x):
    inner = lorentz_inner(x.reshape(-1, x.shape[-1]), x.reshape(-1, x.shape[-1]))
    return torch.allclose(inner, torch.full_like(inner, -1.0), atol=MANIFOLD_EPS)


def test_block_output_shape():
    block = LorentzTransformerBlock(d_model=64, n_heads=4, mlp_ratio=4)
    x_s = torch.randn(2, 16, 64) * 0.3
    x   = project_to_hyperboloid(x_s)    # (2, 16, 65)
    out = block(x)
    assert out.shape == x.shape, f"expected {x.shape}, got {out.shape}"


def test_block_output_on_manifold():
    block = LorentzTransformerBlock(d_model=64, n_heads=4, mlp_ratio=4)
    x_s = torch.randn(2, 16, 64) * 0.3
    x   = project_to_hyperboloid(x_s)
    out = block(x)
    assert on_hyperboloid(out)


def test_block_no_nan():
    block = LorentzTransformerBlock(d_model=64, n_heads=4, mlp_ratio=4)
    x_s = torch.randn(4, 32, 64) * 0.3
    x   = project_to_hyperboloid(x_s)
    out = block(x)
    assert not torch.isnan(out).any()
