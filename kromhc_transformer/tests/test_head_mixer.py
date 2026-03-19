import pytest
import torch
from kromhc_transformer.models.head_mixer import KromHCHeadMixer

def test_head_mixer_output_shape():
    mixer = KromHCHeadMixer(n_heads=8, head_dim=64)
    x = torch.randn(32, 8, 64)
    out, H = mixer(x)
    assert out.shape == (32, 8, 64)
    assert H.shape == (32, 8, 8)

def test_head_mixer_doubly_stochastic_rows():
    mixer = KromHCHeadMixer(n_heads=8, head_dim=64)
    x = torch.randn(16, 8, 64)
    _, H = mixer(x)
    row_sums = H.sum(dim=2)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

def test_head_mixer_doubly_stochastic_cols():
    mixer = KromHCHeadMixer(n_heads=8, head_dim=64)
    x = torch.randn(16, 8, 64)
    _, H = mixer(x)
    col_sums = H.sum(dim=1)
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-5)

def test_head_mixer_gradient_flow():
    mixer = KromHCHeadMixer(n_heads=8, head_dim=64)
    x = torch.randn(8, 8, 64, requires_grad=True)
    out, _ = mixer(x)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().max() > 0

def test_head_mixer_n_heads_2():
    mixer = KromHCHeadMixer(n_heads=2, head_dim=32)
    x = torch.randn(4, 2, 32)
    out, H = mixer(x)
    assert out.shape == (4, 2, 32)
    assert H.shape == (4, 2, 2)

def test_head_mixer_n_heads_4():
    mixer = KromHCHeadMixer(n_heads=4, head_dim=32)
    x = torch.randn(4, 4, 32)
    out, H = mixer(x)
    assert out.shape == (4, 4, 32)
    assert H.shape == (4, 4, 4)

def test_head_mixer_n_heads_16():
    mixer = KromHCHeadMixer(n_heads=16, head_dim=32)
    x = torch.randn(4, 16, 32)
    out, H = mixer(x)
    assert out.shape == (4, 16, 32)
    assert H.shape == (4, 16, 16)

def test_head_mixer_non_power_of_2_raises():
    with pytest.raises(AssertionError):
        KromHCHeadMixer(n_heads=6, head_dim=32)
