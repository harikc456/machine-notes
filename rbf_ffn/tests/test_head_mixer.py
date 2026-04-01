import torch
import pytest
from rbf_ffn.models.head_mixer import KromHCHeadMixer

BS, N_HEADS, HEAD_DIM = 4, 8, 32


@pytest.fixture
def mixer():
    return KromHCHeadMixer(n_heads=N_HEADS, head_dim=HEAD_DIM)


def test_output_shapes(mixer):
    x = torch.randn(BS, N_HEADS, HEAD_DIM)
    mixed, H = mixer(x)
    assert mixed.shape == (BS, N_HEADS, HEAD_DIM)
    assert H.shape == (BS, N_HEADS, N_HEADS)


def test_H_row_sums_to_one(mixer):
    """H must be row-stochastic (each row sums to 1)."""
    x = torch.randn(BS, N_HEADS, HEAD_DIM)
    _, H = mixer(x)
    row_sums = H.sum(dim=-1)  # (BS, N_HEADS)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_H_col_sums_to_one(mixer):
    """H must be col-stochastic (each col sums to 1)."""
    x = torch.randn(BS, N_HEADS, HEAD_DIM)
    _, H = mixer(x)
    col_sums = H.sum(dim=-2)  # (BS, N_HEADS)
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-5)


def test_H_nonnegative(mixer):
    """All entries of H must be >= 0."""
    x = torch.randn(BS, N_HEADS, HEAD_DIM)
    _, H = mixer(x)
    assert (H >= -1e-6).all()


def test_gradient_flows_through_mixed(mixer):
    x = torch.randn(BS, N_HEADS, HEAD_DIM, requires_grad=True)
    mixed, _ = mixer(x)
    mixed.sum().backward()
    assert x.grad is not None
    for gen in mixer.weight_gens:
        for p in gen.parameters():
            assert p.grad is not None


def test_requires_power_of_two():
    with pytest.raises(AssertionError):
        KromHCHeadMixer(n_heads=6, head_dim=32)


def test_identity_when_equal_weights():
    """When both heads are weighted equally, mixing is the average permutation.
    This is a smoke test: mixed output should have finite values."""
    mixer = KromHCHeadMixer(n_heads=4, head_dim=16)
    x = torch.randn(2, 4, 16)
    mixed, H = mixer(x)
    assert torch.isfinite(mixed).all()
    assert torch.isfinite(H).all()


def test_d_context_override():
    """Custom d_context changes MLP input dim without error."""
    mixer = KromHCHeadMixer(n_heads=4, head_dim=16, d_context=8)
    x = torch.randn(2, 4, 16)
    mixed, H = mixer(x)
    assert mixed.shape == (2, 4, 16)
