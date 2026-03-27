# rbf_ffn/tests/test_polar_attention.py
import torch
import pytest
from rbf_ffn.config import ModelConfig
from rbf_ffn.models.attention import PolarAttention

B, N, D, H = 2, 16, 64, 4
HEAD_DIM = D // H


@pytest.fixture
def cfg():
    return ModelConfig(d_model=D, n_heads=H, dropout=0.0)


def test_polar_attn_output_shape(cfg):
    attn = PolarAttention(cfg)
    x = torch.randn(B, N, D)
    assert attn(x).shape == (B, N, D)


def test_polar_attn_no_bias(cfg):
    attn = PolarAttention(cfg)
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert getattr(attn, name).bias is None, f"{name} has unexpected bias"


def test_polar_attn_learnable_scales(cfg):
    attn = PolarAttention(cfg)
    assert attn.q_scale.shape == (H,)
    assert attn.k_scale.shape == (H,)
    assert attn.q_scale.requires_grad
    assert attn.k_scale.requires_grad


def test_polar_attn_causal_mask(cfg):
    """Output at position 0 must not depend on positions 1+."""
    attn = PolarAttention(cfg)
    attn.eval()
    x = torch.randn(1, N, D)
    out_full = attn(x)

    x_corrupt = x.clone()
    x_corrupt[:, 1:, :] = torch.randn_like(x_corrupt[:, 1:, :])
    out_corrupt = attn(x_corrupt)

    assert torch.allclose(out_full[:, 0, :], out_corrupt[:, 0, :], atol=1e-5)


def test_polar_attn_gradient_flows(cfg):
    attn = PolarAttention(cfg)
    x = torch.randn(B, N, D, requires_grad=True)
    attn(x).sum().backward()
    assert x.grad is not None
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert getattr(attn, name).weight.grad is not None
    assert attn.q_scale.grad is not None
    assert attn.k_scale.grad is not None


def test_polar_attn_scale_params_go_to_adamw(cfg):
    """q_scale and k_scale are 1-D; build_optimizer_groups must put them in AdamW."""
    from rbf_ffn.models.model import CausalLM, build_optimizer_groups
    model_cfg = ModelConfig(d_model=D, n_heads=H, n_layers=2, model_type="polar_attn")
    model = CausalLM(model_cfg)
    muon_params, adamw_params = build_optimizer_groups(model)
    muon_ids  = {id(p) for p in muon_params}
    adamw_ids = {id(p) for p in adamw_params}
    for block in model.blocks:
        assert id(block.attn.q_scale) not in muon_ids
        assert id(block.attn.q_scale) in adamw_ids
        assert id(block.attn.k_scale) not in muon_ids
        assert id(block.attn.k_scale) in adamw_ids
