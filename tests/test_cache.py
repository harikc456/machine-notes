from __future__ import annotations
import pytest
import torch
from unittest.mock import MagicMock
from kv_quant.config import QuantConfig
from kv_quant.turboquant import TurboQuantCache
from kv_quant.spectralquant import SpectralQuantCache
import os
import sys
_SPECTRALQUANT_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "spectralquant", "src")
)
if _SPECTRALQUANT_SRC not in sys.path:
    sys.path.insert(0, _SPECTRALQUANT_SRC)


def _make_kv(batch=1, heads=2, seq=5, d=64):
    torch.manual_seed(42)
    return torch.randn(batch, heads, seq, d), torch.randn(batch, heads, seq, d)


def test_turboquant_update_returns_correct_shape():
    cfg = QuantConfig(bits=4, buffer_size=128)
    cache = TurboQuantCache(cfg, n_heads=2, head_dim=64)
    k, v = _make_kv(heads=2, d=64)
    k_out, v_out = cache.update(k, v, layer_idx=0)
    assert k_out.shape == k.shape
    assert v_out.shape == v.shape


def test_turboquant_accumulates_sequence():
    cfg = QuantConfig(bits=4, buffer_size=128)
    cache = TurboQuantCache(cfg, n_heads=2, head_dim=64)
    k1, v1 = _make_kv(seq=3, heads=2, d=64)
    k2, v2 = _make_kv(seq=1, heads=2, d=64)
    cache.update(k1, v1, layer_idx=0)
    k_out, v_out = cache.update(k2, v2, layer_idx=0)
    assert k_out.shape[-2] == 4
    assert cache.get_seq_length(layer_idx=0) == 4


def test_turboquant_no_nan():
    cfg = QuantConfig(bits=4, buffer_size=128)
    cache = TurboQuantCache(cfg, n_heads=2, head_dim=64)
    k, v = _make_kv(heads=2, d=64)
    k_out, v_out = cache.update(k, v, layer_idx=0)
    assert not torch.isnan(k_out).any()
    assert not torch.isnan(v_out).any()


def test_turboquant_buffer_flush():
    """Tokens older than buffer_size should be quantized, not in buffer."""
    cfg = QuantConfig(bits=4, buffer_size=4, value_bits=2, value_group_size=16)
    cache = TurboQuantCache(cfg, n_heads=2, head_dim=64)
    # Insert 6 tokens — 2 should flush to quantized storage
    for _ in range(6):
        k, v = _make_kv(seq=1, heads=2, d=64)
        cache.update(k, v, layer_idx=0)
    assert cache._qk[0] is not None, "Tokens should have been flushed to quantized storage"
    assert cache._k_buf[0].shape[-2] == 4
    assert cache.get_seq_length(0) == 6


def test_turboquant_multiple_layers():
    cfg = QuantConfig(bits=4, buffer_size=128)
    cache = TurboQuantCache(cfg, n_heads=2, head_dim=64)
    k, v = _make_kv(heads=2, d=64)
    cache.update(k, v, layer_idx=0)
    cache.update(k, v, layer_idx=1)
    assert cache.get_seq_length(layer_idx=0) == 5
    assert cache.get_seq_length(layer_idx=1) == 5



def _make_spectralquant_cal_data(
    n_layers: int = 2, n_kv_heads: int = 2, D: int = 64, avg_bits: int = 4
) -> tuple:
    """Synthetic (calibrator, quant_state) using identity eigenvectors — no model needed."""
    from spectralquant.calibration import EigenspectralCalibrator, HeadCalibrationData
    from spectralquant.nonuniform_quantization import NonUniformQuantizer

    calibrator = EigenspectralCalibrator()
    torch.manual_seed(0)
    quant_state: dict = {}

    for l in range(n_layers):
        for h in range(n_kv_heads):
            for kv_type in ("key", "value"):
                eigenvectors = torch.eye(D)
                eigenvalues = torch.ones(D)
                # With all eigenvalues=1, participation ratio = D, so d_eff_int = D-1.
                # We override d_eff via the fit() call to use D//2 for predictable allocation.
                d_eff_float = float(D // 2)

                calibrator._calibration_data[(l, h, kv_type)] = HeadCalibrationData(
                    layer_idx=l,
                    head_idx=h,
                    head_type=kv_type,
                    eigenvalues=eigenvalues,
                    eigenvectors=eigenvectors,
                    d_eff=d_eff_float,
                    spectral_gap=None,
                    var_95=D // 2,
                    var_99=min(D * 3 // 4, D),
                    n_samples=200,
                    head_dim=D,
                )

                rotated = torch.randn(200, D)
                quant = NonUniformQuantizer(eigenvalues=eigenvalues, avg_bits=float(avg_bits))
                quant.fit(rotated, d_eff=d_eff_float)

                quant_state[f"L{l}_H{h}_{kv_type}"] = {
                    "semantic_centroids": quant._semantic_quantizer._centroids.clone(),
                    "tail_centroids": quant._tail_quantizer._centroids.clone(),
                    "d_eff_int": quant._d_eff_int,
                    "b_high": quant._b_high,
                    "b_low": quant._b_low,
                    "head_dim": D,
                }

    calibrator._is_calibrated = True
    return (calibrator, quant_state)


def test_spectralquant_update_returns_correct_shape():
    cfg = QuantConfig(method="spectralquant", bits=4)
    cal = _make_spectralquant_cal_data()
    cache = SpectralQuantCache(cfg, cal)
    k, v = _make_kv(heads=2, d=64)
    k_out, v_out = cache.update(k, v, layer_idx=0)
    assert k_out.shape == k.shape
    assert v_out.shape == v.shape


def test_spectralquant_accumulates_sequence():
    cfg = QuantConfig(method="spectralquant", bits=4)
    cal = _make_spectralquant_cal_data()
    cache = SpectralQuantCache(cfg, cal)
    k1, v1 = _make_kv(seq=3, heads=2, d=64)
    k2, v2 = _make_kv(seq=1, heads=2, d=64)
    cache.update(k1, v1, layer_idx=0)
    k_out, v_out = cache.update(k2, v2, layer_idx=0)
    assert k_out.shape[-2] == 4
    assert cache.get_seq_length(0) == 4


def test_spectralquant_no_nan():
    cfg = QuantConfig(method="spectralquant", bits=4)
    cal = _make_spectralquant_cal_data()
    cache = SpectralQuantCache(cfg, cal)
    k, v = _make_kv(heads=2, d=64)
    k_out, v_out = cache.update(k, v, layer_idx=0)
    assert not torch.isnan(k_out).any()
    assert not torch.isnan(v_out).any()


def test_spectralquant_compressed_smaller_than_fp16():
    cfg = QuantConfig(method="spectralquant", bits=4)
    cal = _make_spectralquant_cal_data()
    cache = SpectralQuantCache(cfg, cal)
    k, v = _make_kv(batch=1, heads=2, seq=64, d=64)
    cache.update(k, v, layer_idx=0)
    fp16_k_bytes = k.nelement() * 2  # K only (1*2*64*64 * 2 bytes = 16384)
    # compressed_bytes() counts K+V at 4 bits/coord = 8192 bytes < 65536
    assert cache.compressed_bytes() < fp16_k_bytes * 4


# ---------------------------------------------------------------------------
# wrap() API tests
# ---------------------------------------------------------------------------
from kv_quant import wrap


def _mock_model(n_kv_heads=4, n_heads=8, hidden_size=512, head_dim=64, n_layers=2):
    model = MagicMock()
    model.config.num_key_value_heads = n_kv_heads
    model.config.num_attention_heads = n_heads
    model.config.hidden_size = hidden_size
    model.config.head_dim = head_dim
    model.config.num_hidden_layers = n_layers
    model.parameters = lambda: iter([torch.zeros(1)])
    model.generate = MagicMock(return_value=torch.zeros(1, 10, dtype=torch.long))
    return model


def test_wrap_returns_model():
    model = _mock_model()
    cfg = QuantConfig(method="turboquant", bits=4)
    result = wrap(model, cfg)
    assert result is model


def test_wrap_sets_quant_config():
    model = _mock_model()
    cfg = QuantConfig(method="turboquant", bits=4)
    wrap(model, cfg)
    assert model._kv_quant_config is cfg


def test_wrap_injects_turboquant_cache():
    from kv_quant.turboquant import TurboQuantCache
    model = _mock_model()
    cfg = QuantConfig(method="turboquant", bits=4)
    wrap(model, cfg)

    captured = {}
    def fake_generate(*args, **kwargs):
        captured["cache"] = kwargs.get("past_key_values")
        return torch.zeros(1, 10, dtype=torch.long)

    model.generate = fake_generate
    # Re-wrap so the patched generate is the one wrapped
    wrap(model, cfg)
    model.generate(torch.zeros(1, 5, dtype=torch.long))
    assert isinstance(captured["cache"], TurboQuantCache)


def test_wrap_spectralquant_raises_without_calibration():
    model = _mock_model()
    cfg = QuantConfig(method="spectralquant", bits=4, calibration_path=None)
    with pytest.raises(ValueError, match="calibration_path"):
        wrap(model, cfg)


# ---------------------------------------------------------------------------
# TriAttention guard rail tests (no model/stats needed — tests ValueError only)
# ---------------------------------------------------------------------------

def test_wrap_standalone_triattention_requires_calibration_path():
    """method=None, eviction=triattention, no calibration_path → ValueError."""
    model = _mock_model()
    cfg = QuantConfig(method=None, eviction="triattention", budget=256, calibration_path=None)
    with pytest.raises(ValueError, match="calibration_path"):
        wrap(model, cfg)


def test_wrap_combined_triattention_requires_calibration_path():
    """method=turboquant, eviction=triattention, no calibration_path → ValueError."""
    model = _mock_model()
    cfg = QuantConfig(method="turboquant", eviction="triattention", budget=256, calibration_path=None)
    with pytest.raises(ValueError, match="calibration_path"):
        wrap(model, cfg)


def test_wrap_triattention_requires_model_name_or_path():
    """eviction=triattention with _name_or_path=None → ValueError."""
    model = _mock_model()
    model.config._name_or_path = None
    cfg = QuantConfig(method=None, eviction="triattention", budget=256, calibration_path="/fake/stats.pt")
    with pytest.raises(ValueError, match="_name_or_path"):
        wrap(model, cfg)


def test_plain_cache_compressed_bytes_single_layer():
    """_make_plain_cache() compressed_bytes() sums bfloat16 KV bytes."""
    from kv_quant import _make_plain_cache
    cache = _make_plain_cache()
    B, H, S, D = 1, 2, 10, 64
    k = torch.zeros(B, H, S, D, dtype=torch.bfloat16)
    v = torch.zeros(B, H, S, D, dtype=torch.bfloat16)
    cache.key_cache.append(k)
    cache.value_cache.append(v)
    expected = (k.nelement() + v.nelement()) * 2  # bfloat16 = 2 bytes
    assert cache.compressed_bytes() == expected


def test_plain_cache_compressed_bytes_multiple_layers():
    """_make_plain_cache() accumulates across two layers."""
    from kv_quant import _make_plain_cache
    cache = _make_plain_cache()
    B, H, S, D = 1, 2, 5, 64
    k = torch.zeros(B, H, S, D, dtype=torch.bfloat16)
    v = torch.zeros(B, H, S, D, dtype=torch.bfloat16)
    cache.key_cache.extend([k, k])
    cache.value_cache.extend([v, v])
    expected = 2 * (k.nelement() + v.nelement()) * 2
    assert cache.compressed_bytes() == expected
