import torch
import torch.nn as nn
import pytest
from rbf_ffn.config import RBFFFNConfig
from rbf_ffn.models.rbf_ffn import RBFFFN

D, B, N = 16, 2, 10
CENTERS = [-1.0, -0.5, 0.0, 0.5, 1.0]
K = 5


def make_ffn(gate_variant: str) -> RBFFFN:
    cfg = RBFFFNConfig(d_model=D, K=K, centers=CENTERS, gate_variant=gate_variant)
    return RBFFFN(cfg)


@pytest.mark.parametrize("variant", ["G0", "G1A", "G1B"])
def test_output_shape_dk_variants(variant):
    """G0, G1A, G1B all output (B, N, d_model) after down projection."""
    ffn = make_ffn(variant)
    x = torch.randn(B, N, D)
    out = ffn(x)
    assert out.shape == (B, N, D)


def test_output_shape_g2():
    """G2 also outputs (B, N, d_model) — down proj is d_model → d_model."""
    ffn = make_ffn("G2")
    x = torch.randn(B, N, D)
    out = ffn(x)
    assert out.shape == (B, N, D)


@pytest.mark.parametrize("variant", ["G0", "G1A", "G1B", "G2"])
def test_gradient_flows_all_variants(variant):
    ffn = make_ffn(variant)
    x = torch.randn(B, N, D, requires_grad=True)
    ffn(x).sum().backward()
    assert x.grad is not None


def test_invalid_gate_variant_raises():
    with pytest.raises(ValueError, match="Unknown gate_variant"):
        cfg = RBFFFNConfig(d_model=D, gate_variant="INVALID")
        RBFFFN(cfg)


@pytest.mark.parametrize("variant", ["G0", "G1A", "G1B", "G2"])
def test_down_proj_input_dim(variant):
    """
    G0/G1A/G1B down proj: d_model*K → d_model.
    G2 down proj: d_model → d_model (Sinkhorn collapses K).
    """
    ffn = make_ffn(variant)
    expected_in = D * K if variant != "G2" else D
    assert ffn.down_proj.in_features == expected_in
    assert ffn.down_proj.out_features == D


def test_internal_norm_is_rmsnorm():
    """RBFFFN must use nn.RMSNorm internally, not LayerNorm."""
    ffn = make_ffn("G0")
    assert isinstance(ffn.norm, nn.RMSNorm), f"Expected RMSNorm, got {type(ffn.norm)}"


def test_down_proj_has_no_bias():
    """down_proj must have bias=False to match Llama no-bias convention."""
    for variant in ["G0", "G1A", "G1B", "G2"]:
        ffn = make_ffn(variant)
        assert ffn.down_proj.bias is None, f"{variant}: down_proj still has bias"
