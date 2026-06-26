from __future__ import annotations
import torch
from kv_quant.config import QuantConfig
from kv_quant.turboquant import TurboQuantCache


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
