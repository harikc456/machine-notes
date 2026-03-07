import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest
from geometry.lorentz import (
    lorentz_inner, project_to_hyperboloid, exp_map_origin,
    log_map_origin, lorentz_normalize, lorentz_distance,
)

MANIFOLD_EPS = 1e-4


def on_hyperboloid(x: torch.Tensor) -> bool:
    """Check ⟨x, x⟩_L = -1 for all points."""
    inner = lorentz_inner(x.reshape(-1, x.shape[-1]), x.reshape(-1, x.shape[-1]))
    return torch.allclose(inner, torch.full_like(inner, -1.0), atol=MANIFOLD_EPS)


def test_project_to_hyperboloid_satisfies_constraint():
    x_s = torch.randn(32, 64)
    x = project_to_hyperboloid(x_s)
    assert x.shape == (32, 65), f"expected (32, 65), got {x.shape}"
    assert on_hyperboloid(x), "projected points must satisfy ⟨x,x⟩_L = -1"


def test_lorentz_inner_at_origin():
    o = torch.zeros(1, 65)
    o[:, 0] = 1.0
    result = lorentz_inner(o, o)
    assert torch.allclose(result, torch.tensor([-1.0]), atol=1e-6)


def test_exp_log_roundtrip():
    """exp_o(log_o(x)) should recover x for points on hyperboloid."""
    x_s = torch.randn(16, 64) * 0.5
    x = project_to_hyperboloid(x_s)
    u = log_map_origin(x)
    x_rec = exp_map_origin(u)
    assert torch.allclose(x, x_rec, atol=1e-4), \
        f"roundtrip error: {(x - x_rec).abs().max():.6f}"


def test_distance_is_non_negative():
    x_s = torch.randn(8, 64) * 0.3
    y_s = torch.randn(8, 64) * 0.3
    x = project_to_hyperboloid(x_s)
    y = project_to_hyperboloid(y_s)
    d = lorentz_distance(x, y)
    assert (d >= 0).all(), "distances must be non-negative"


def test_distance_is_zero_for_same_point():
    x_s = torch.randn(4, 64) * 0.3
    x = project_to_hyperboloid(x_s)
    d = lorentz_distance(x, x)
    assert torch.allclose(d, torch.zeros_like(d), atol=1e-3)


def test_lorentz_normalize_preserves_manifold():
    x_s = torch.randn(8, 64) * 0.3
    x = project_to_hyperboloid(x_s)
    x_perturbed = x + torch.randn_like(x) * 0.01
    x_norm = lorentz_normalize(x_perturbed)
    assert on_hyperboloid(x_norm)


# ── Layer tests ────────────────────────────────────────────────────────────────

from models.lorentz_layers import LorentzLinear, LorentzLayerNorm, LorentzCentroid


def test_lorentz_linear_output_on_manifold():
    layer = LorentzLinear(in_features=64, out_features=128)
    x_s = torch.randn(8, 32, 64) * 0.3
    x   = project_to_hyperboloid(x_s)   # (8, 32, 65)
    y   = layer(x)                       # (8, 32, 129)
    assert y.shape == (8, 32, 129), f"expected (8,32,129), got {y.shape}"
    assert on_hyperboloid(y), "LorentzLinear output must lie on hyperboloid"


def test_lorentz_layer_norm_output_on_manifold():
    norm = LorentzLayerNorm(64)
    x_s  = torch.randn(8, 32, 64) * 0.3
    x    = project_to_hyperboloid(x_s)
    y    = norm(x)
    assert y.shape == x.shape
    assert on_hyperboloid(y)


def test_lorentz_centroid_output_on_manifold():
    B, N, D = 4, 16, 65
    x_s     = torch.randn(B, N, D - 1) * 0.3
    x       = project_to_hyperboloid(x_s)              # (B, N, D)
    weights = torch.softmax(torch.randn(B, N), dim=-1) # (B, N)
    y       = LorentzCentroid.apply_weights(x, weights) # (B, D)
    assert y.shape == (B, D)
    assert on_hyperboloid(y)
