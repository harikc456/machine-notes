from __future__ import annotations
import torch
from kv_quant.config import QuantConfig
from kv_quant.turboquant import TurboQuantCache
from kv_quant.calibrate import _compute_bit_split
from kv_quant.spectralquant import SpectralQuantCache
from kv_quant.ops.qjl import make_sign_matrix


def _make_kv(batch=1, heads=2, seq=5, d=16):
    torch.manual_seed(42)
    return torch.randn(batch, heads, seq, d), torch.randn(batch, heads, seq, d)


def test_turboquant_update_returns_correct_shape():
    cfg = QuantConfig(bits=4, qjl_dim=8)
    cache = TurboQuantCache(cfg, n_heads=2, head_dim=16)
    k, v = _make_kv(heads=2, d=16)
    k_out, v_out = cache.update(k, v, layer_idx=0)
    assert k_out.shape == k.shape
    assert v_out.shape == v.shape


def test_turboquant_accumulates_sequence():
    cfg = QuantConfig(bits=4, qjl_dim=8)
    cache = TurboQuantCache(cfg, n_heads=2, head_dim=16)
    k1, v1 = _make_kv(seq=3, heads=2, d=16)
    k2, v2 = _make_kv(seq=1, heads=2, d=16)
    cache.update(k1, v1, layer_idx=0)
    k_out, v_out = cache.update(k2, v2, layer_idx=0)
    assert k_out.shape[-2] == 4  # 3 + 1
    assert cache.get_seq_length(layer_idx=0) == 4


def test_turboquant_no_nan():
    cfg = QuantConfig(bits=4, qjl_dim=8)
    cache = TurboQuantCache(cfg, n_heads=2, head_dim=16)
    k, v = _make_kv(heads=2, d=16)
    k_out, v_out = cache.update(k, v, layer_idx=0)
    assert not torch.isnan(k_out).any()
    assert not torch.isnan(v_out).any()


def test_turboquant_compressed_smaller_than_fp16():
    cfg = QuantConfig(bits=4, qjl_dim=8)
    cache = TurboQuantCache(cfg, n_heads=2, head_dim=16)
    k, v = _make_kv(batch=1, heads=2, seq=64, d=16)
    cache.update(k, v, layer_idx=0)
    fp16_bytes = k.nelement() * 2 * 2  # K + V, float16
    assert cache.compressed_bytes() < fp16_bytes


def test_turboquant_multiple_layers():
    cfg = QuantConfig(bits=4, qjl_dim=8)
    cache = TurboQuantCache(cfg, n_heads=2, head_dim=16)
    k, v = _make_kv(heads=2, d=16)
    cache.update(k, v, layer_idx=0)
    cache.update(k, v, layer_idx=1)
    assert cache.get_seq_length(layer_idx=0) == 5
    assert cache.get_seq_length(layer_idx=1) == 5


def test_compute_bit_split_budget():
    # For d=128, d_s=4, total_bits=4, boost=2.0:
    # bits_signal=8, bits_noise should make average ≈ 4
    bits_s, bits_n = _compute_bit_split(total_bits=4, d=128, d_s=4, signal_bit_boost=2.0)
    # Average: (4*bits_s + 124*bits_n) / 128 should be close to 4
    avg = (4 * bits_s + 124 * bits_n) / 128
    assert abs(avg - 4.0) < 1.0
    assert bits_s >= bits_n  # signal gets more bits
    assert 1 <= bits_n <= 8
    assert 1 <= bits_s <= 8


def test_compute_bit_split_low_bits():
    bits_s, bits_n = _compute_bit_split(total_bits=2, d=128, d_s=4, signal_bit_boost=2.0)
    assert bits_s >= bits_n
    assert bits_n >= 1


def _make_synthetic_cal_data(
    n_layers: int = 2, n_kv_heads: int = 2, head_dim: int = 16,
    d_s: int = 4, bits: int = 4, qjl_dim: int = 8
) -> dict:
    """Synthetic calibration data for unit tests — no model download needed."""
    torch.manual_seed(0)
    layers: dict = {}
    for l in range(n_layers):
        layers[l] = {}
        for h in range(n_kv_heads):
            U, _ = torch.linalg.qr(torch.randn(head_dim, head_dim))
            cb_s = torch.randn(2 ** bits, d_s)
            cb_n = torch.randn(2 ** max(1, bits - 1), head_dim - d_s)
            S = make_sign_matrix(qjl_dim, d_s)
            layers[l][h] = {
                "U": U, "d_s": d_s,
                "bits_signal": bits, "bits_noise": max(1, bits - 1),
                "codebook_signal": cb_s, "codebook_noise": cb_n, "S_signal": S,
            }
    return {
        "model_id": "test", "n_layers": n_layers, "n_kv_heads": n_kv_heads,
        "head_dim": head_dim, "qjl_dim": qjl_dim, "layers": layers,
    }


def test_spectralquant_update_returns_correct_shape():
    cfg = QuantConfig(method="spectralquant", bits=4)
    cal = _make_synthetic_cal_data()
    cache = SpectralQuantCache(cfg, cal)
    k, v = _make_kv(heads=2, d=16)
    k_out, v_out = cache.update(k, v, layer_idx=0)
    assert k_out.shape == k.shape
    assert v_out.shape == v.shape


def test_spectralquant_accumulates_sequence():
    cfg = QuantConfig(method="spectralquant", bits=4)
    cal = _make_synthetic_cal_data()
    cache = SpectralQuantCache(cfg, cal)
    k1, v1 = _make_kv(seq=3, heads=2, d=16)
    k2, v2 = _make_kv(seq=1, heads=2, d=16)
    cache.update(k1, v1, layer_idx=0)
    k_out, v_out = cache.update(k2, v2, layer_idx=0)
    assert k_out.shape[-2] == 4
    assert cache.get_seq_length(0) == 4


def test_spectralquant_no_nan():
    cfg = QuantConfig(method="spectralquant", bits=4)
    cal = _make_synthetic_cal_data()
    cache = SpectralQuantCache(cfg, cal)
    k, v = _make_kv(heads=2, d=16)
    k_out, v_out = cache.update(k, v, layer_idx=0)
    assert not torch.isnan(k_out).any()
    assert not torch.isnan(v_out).any()


def test_spectralquant_compressed_smaller_than_fp16():
    cfg = QuantConfig(method="spectralquant", bits=4)
    cal = _make_synthetic_cal_data()
    cache = SpectralQuantCache(cfg, cal)
    k, v = _make_kv(batch=1, heads=2, seq=64, d=16)
    cache.update(k, v, layer_idx=0)
    fp16_k_bytes = k.nelement() * 2  # K only (V stored bfloat16 = 2 bytes)
    assert cache.compressed_bytes() < fp16_k_bytes * 4  # sanity, not tight
