from __future__ import annotations
import torch
import pytest

from flow_matching.config import FlowConfig


def _tiny_model():
    from flow_matching.model import DiT
    cfg = FlowConfig(d_model=64, n_heads=2, n_layers=2, patch_size=4)
    return DiT(cfg).eval()


def test_euler_sample_output_shape():
    from flow_matching.sample import euler_sample
    model  = _tiny_model()
    B      = 4
    y      = torch.randint(0, 100, (B,))
    device = torch.device("cpu")

    with torch.no_grad():
        out = euler_sample(model, y, cfg_scale=1.0, n_steps=2, device=device)

    assert out.shape == (B, 3, 32, 32), f"Expected ({B},3,32,32), got {out.shape}"


def test_euler_sample_cfg_scale_zero_equals_uncond():
    """cfg_scale=0 → guided velocity = uncond velocity."""
    from flow_matching.sample import euler_sample
    model  = _tiny_model()
    B      = 2
    y      = torch.zeros(B, dtype=torch.long)
    device = torch.device("cpu")

    torch.manual_seed(0)
    with torch.no_grad():
        out_cfg0 = euler_sample(model, y, cfg_scale=0.0, n_steps=2, device=device)

    torch.manual_seed(0)
    with torch.no_grad():
        # With cfg_scale=0, guided = uncond; equivalent to passing null token
        y_null = torch.full_like(y, 100)
        out_uncond = euler_sample(model, y_null, cfg_scale=0.0, n_steps=2, device=device)

    assert torch.allclose(out_cfg0, out_uncond, atol=1e-5)


def test_euler_sample_null_class_accepted():
    from flow_matching.sample import euler_sample
    model  = _tiny_model()
    y      = torch.full((2,), 100)  # null tokens
    with torch.no_grad():
        out = euler_sample(model, y, cfg_scale=3.0, n_steps=2, device=torch.device("cpu"))
    assert out.shape == (2, 3, 32, 32)
