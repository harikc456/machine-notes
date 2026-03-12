import torch
import pytest
from rbf_ffn.models.rbf_layer import RBFLayer

CENTERS = [-1.0, -0.5, 0.0, 0.5, 1.0]
K = len(CENTERS)
D, B, N = 8, 2, 10


# ── σ-A (global) — regression: existing behaviour must be unchanged ──────────

def test_sigma_a_output_shape():
    layer = RBFLayer(d_model=D, centers=CENTERS, sigma_init=0.5, sigma_variant="global")
    assert layer(torch.randn(B, N, D)).shape == (B, N, D * K)


def test_sigma_a_param_count():
    layer = RBFLayer(d_model=D, centers=CENTERS, sigma_init=0.5, sigma_variant="global")
    sigma_params = sum(p.numel() for n, p in layer.named_parameters() if "sigma" in n)
    assert sigma_params == 1


# ── σ-B (per-center) ─────────────────────────────────────────────────────────

def test_sigma_b_output_shape():
    layer = RBFLayer(d_model=D, centers=CENTERS, sigma_init=0.5, sigma_variant="per_center")
    assert layer(torch.randn(B, N, D)).shape == (B, N, D * K)


def test_sigma_b_param_count():
    """σ-B must have exactly K learnable σ parameters."""
    layer = RBFLayer(d_model=D, centers=CENTERS, sigma_init=0.5, sigma_variant="per_center")
    sigma_params = sum(p.numel() for n, p in layer.named_parameters() if "sigma" in n)
    assert sigma_params == K


def test_sigma_b_all_positive():
    layer = RBFLayer(d_model=D, centers=CENTERS, sigma_init=0.5, sigma_variant="per_center")
    sigmas = layer.sigma  # (K,)
    assert sigmas.shape == (K,)
    assert sigmas.min().item() > 0.0


def test_sigma_b_init_value():
    layer = RBFLayer(d_model=D, centers=CENTERS, sigma_init=0.5, sigma_variant="per_center")
    assert torch.allclose(layer.sigma, torch.full((K,), 0.5), atol=1e-5)


def test_sigma_b_gradient_flows():
    layer = RBFLayer(d_model=D, centers=CENTERS, sigma_init=0.5, sigma_variant="per_center")
    x = torch.randn(B, N, D, requires_grad=True)
    layer(x).sum().backward()
    assert layer.sigma_raw.grad is not None
    assert x.grad is not None


# ── σ-C (per-dim-per-center) ─────────────────────────────────────────────────

def test_sigma_c_output_shape():
    layer = RBFLayer(d_model=D, centers=CENTERS, sigma_init=0.5, sigma_variant="per_dim")
    assert layer(torch.randn(B, N, D)).shape == (B, N, D * K)


def test_sigma_c_param_count():
    """σ-C must have exactly d_model * K learnable σ parameters."""
    layer = RBFLayer(d_model=D, centers=CENTERS, sigma_init=0.5, sigma_variant="per_dim")
    sigma_params = sum(p.numel() for n, p in layer.named_parameters() if "sigma" in n)
    assert sigma_params == D * K


def test_sigma_c_all_positive():
    layer = RBFLayer(d_model=D, centers=CENTERS, sigma_init=0.5, sigma_variant="per_dim")
    sigmas = layer.sigma  # (d_model, K)
    assert sigmas.shape == (D, K)
    assert sigmas.min().item() > 0.0


def test_sigma_c_init_value():
    layer = RBFLayer(d_model=D, centers=CENTERS, sigma_init=0.5, sigma_variant="per_dim")
    assert torch.allclose(layer.sigma, torch.full((D, K), 0.5), atol=1e-5)


def test_sigma_c_gradient_flows():
    layer = RBFLayer(d_model=D, centers=CENTERS, sigma_init=0.5, sigma_variant="per_dim")
    x = torch.randn(B, N, D, requires_grad=True)
    layer(x).sum().backward()
    assert layer.sigma_raw.grad is not None
    assert x.grad is not None


def test_invalid_sigma_variant_raises():
    with pytest.raises(ValueError, match="Unknown sigma_variant"):
        RBFLayer(d_model=D, centers=CENTERS, sigma_variant="bad")
