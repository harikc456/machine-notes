import torch
import pytest
from rbf_ffn.models.rbf_layer import RBFLayer


CENTERS = [-1.0, -0.5, 0.0, 0.5, 1.0]
K = len(CENTERS)


@pytest.fixture
def layer():
    return RBFLayer(d_model=8, centers=CENTERS, sigma_init=0.5)


def test_output_shape(layer):
    x = torch.randn(2, 10, 8)
    out = layer(x)
    assert out.shape == (2, 10, 8 * K)


def test_output_range(layer):
    """RBF outputs must be in (0, 1]."""
    x = torch.randn(4, 16, 8)
    out = layer(x)
    assert out.min() > 0.0
    assert out.max() <= 1.0 + 1e-6


def test_kernel_value_at_center():
    """φ_k(c_k) should equal 1.0 (peak of Gaussian)."""
    layer = RBFLayer(d_model=1, centers=[0.0], sigma_init=0.5)
    x = torch.zeros(1, 1, 1)  # exactly at center 0.0
    out = layer(x)            # (1, 1, 1)
    assert abs(out.item() - 1.0) < 1e-6


def test_sigma_stays_positive(layer):
    """σ = softplus(σ_raw) must always be positive."""
    assert layer.sigma.item() > 0.0
    # Force sigma_raw very negative; softplus still > 0
    with torch.no_grad():
        layer.sigma_raw.fill_(-100.0)
    assert layer.sigma.item() > 0.0


def test_sigma_init_value():
    """σ should equal sigma_init at construction."""
    layer = RBFLayer(d_model=4, centers=CENTERS, sigma_init=0.5)
    assert abs(layer.sigma.item() - 0.5) < 1e-5


def test_centers_not_in_grad(layer):
    """Centers must be a buffer, not a parameter."""
    param_names = {n for n, _ in layer.named_parameters()}
    assert "centers" not in param_names


def test_gradient_flows_through_sigma(layer):
    """Gradient must reach sigma_raw during backprop."""
    x = torch.randn(2, 4, 8)
    out = layer(x)
    out.sum().backward()
    assert layer.sigma_raw.grad is not None


def test_gradient_flows_through_input(layer):
    """Gradient must reach input x (needed for backprop through FFN)."""
    x = torch.randn(2, 4, 8, requires_grad=True)
    out = layer(x)
    out.sum().backward()
    assert x.grad is not None
