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


from kv_quant.ops.scalar_quant import quantize, dequantize


def test_quantize_dtypes():
    h = torch.randn(4, 64)
    h_int, scale = quantize(h, 4)
    assert h_int.dtype == torch.int8
    assert scale.dtype == torch.float16
    assert scale.shape == (*h.shape[:-1], 1)


def test_quantize_dequantize_roundtrip():
    torch.manual_seed(0)
    h = torch.randn(8, 128)
    for bits in [2, 4, 7]:
        h_int, scale = quantize(h, bits)
        h_rec = dequantize(h_int, scale, bits)
        n_levels = 2 ** bits
        # Max error bounded by quantization step times max scale
        step = 2.0 / n_levels
        max_err = (h - h_rec).abs().max().item()
        max_scale = scale.max().item()
        assert max_err <= step * max_scale + 1e-4, f"bits={bits}: max_err={max_err:.4f} > {step * max_scale:.4f}"


def test_quantize_clamps_to_range():
    h = torch.tensor([[100.0, -100.0, 0.5]])
    h_int, scale = quantize(h, 4)
    h_rec = dequantize(h_int, scale, 4)
    # Reconstructed values should be within [-max_val, max_val]
    max_val = h.abs().max().item()
    assert h_rec.abs().max().item() <= max_val + 1e-3
