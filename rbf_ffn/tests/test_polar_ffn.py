# rbf_ffn/tests/test_polar_ffn.py
import torch
import pytest
from rbf_ffn.config import ModelConfig

B, N, D = 2, 10, 32
FFN_HIDDEN = 86


def make_cfg():
    return ModelConfig(
        d_model=D, n_heads=4, n_layers=2, seq_len=N,
        ffn_hidden=FFN_HIDDEN, dropout=0.0, model_type="polar_mlp",
    )


@pytest.fixture
def ffn():
    # Lazy import: AdaptivePolarMLP doesn't exist yet (TDD). This lets pytest
    # collect the test file without crashing; tests fail at fixture setup time.
    from rbf_ffn.models.polar_ffn import AdaptivePolarMLP
    return AdaptivePolarMLP(make_cfg())


def test_output_shape(ffn):
    x = torch.randn(B, N, D)
    assert ffn(x).shape == (B, N, D)


def test_no_bias_on_down_proj(ffn):
    assert ffn.down_proj.bias is None


def test_parameter_shapes(ffn):
    assert ffn.keys.shape == (FFN_HIDDEN, D)
    assert ffn.thresholds.shape == (FFN_HIDDEN,)
    assert ffn.down_proj.weight.shape == (D, FFN_HIDDEN)


def test_threshold_init(ffn):
    assert torch.allclose(ffn.thresholds, torch.full((FFN_HIDDEN,), 0.7), atol=1e-6)


def test_gradient_flows(ffn):
    x = torch.randn(B, N, D, requires_grad=True)
    ffn(x).sum().backward()
    assert x.grad is not None
    assert ffn.keys.grad is not None
    assert ffn.thresholds.grad is not None
    assert ffn.down_proj.weight.grad is not None


def test_gate_behavior(ffn):
    """Tokens aligned with a key direction (cos_sim > threshold) should produce
    larger output magnitude than misaligned tokens (cos_sim << threshold)."""
    with torch.no_grad():
        # Set all thresholds impossibly high (cos_sim can never reach 2.0)
        # so all neurons produce ~0 output
        ffn.thresholds[:] = 2.0

        # Key 0: threshold set to 0.7, direction along dim 0
        direction = torch.zeros(D)
        direction[0] = 1.0
        ffn.keys[0] = direction
        ffn.thresholds[0] = 0.7

        # Aligned input: cos_sim with key 0 ~= 1.0 > 0.7 → gate ~= 1 → active
        x_aligned = torch.zeros(1, 1, D)
        x_aligned[0, 0, 0] = 1.0

        # Misaligned input: cos_sim with key 0 ~= 0 < 0.7 → gate ~= 0 → silent
        x_misaligned = torch.zeros(1, 1, D)
        x_misaligned[0, 0, 1] = 1.0

        out_aligned    = ffn(x_aligned)
        out_misaligned = ffn(x_misaligned)

        # Aligned input should produce a meaningfully larger output
        assert out_aligned.abs().sum() > out_misaligned.abs().sum()
