from __future__ import annotations

from kv_quant.config import QuantConfig


def test_quantconfig_defaults():
    cfg = QuantConfig()
    assert cfg.method == "turboquant"
    assert cfg.bits == 4
    assert cfg.qjl_dim == 32
    assert cfg.calibration_path is None
    assert cfg.signal_bit_boost == 2.0


def test_quantconfig_custom():
    cfg = QuantConfig(method="spectralquant", bits=2, calibration_path="foo.pt")
    assert cfg.method == "spectralquant"
    assert cfg.bits == 2


import torch
from kv_quant.ops.rotation import make_rotation, rotate, unrotate


def test_rotation_orthogonal():
    torch.manual_seed(0)
    R = make_rotation(64)
    assert torch.allclose(R @ R.T, torch.eye(64), atol=1e-5)


def test_rotate_unrotate_roundtrip():
    torch.manual_seed(0)
    d, H = 32, 4
    R = torch.stack([make_rotation(d) for _ in range(H)])
    h = torch.randn(2, H, 10, d)
    assert torch.allclose(unrotate(rotate(h, R), R), h, atol=1e-5)


def test_rotate_shape():
    torch.manual_seed(0)
    d, H = 16, 3
    R = torch.stack([make_rotation(d) for _ in range(H)])
    h = torch.randn(1, H, 5, d)
    assert rotate(h, R).shape == (1, H, 5, d)
