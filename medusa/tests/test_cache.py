from __future__ import annotations
import torch
import pytest


def _import_helpers():
    from medusa.cache import _quantise_int8, _dequantise_int8
    return _quantise_int8, _dequantise_int8


def test_quantise_shape():
    q = _import_helpers()[0]
    t = torch.randn(4, 32)
    q_t, scales = q(t)
    assert q_t.shape == t.shape
    assert q_t.dtype == torch.int8
    assert scales.shape == (4,)


def test_quantise_range():
    q, _ = _import_helpers()
    t = torch.randn(8, 16)
    q_t, _ = q(t)
    assert q_t.abs().max().item() <= 127


def test_roundtrip_close():
    q, dq = _import_helpers()
    torch.manual_seed(42)
    t = torch.randn(6, 64) * 0.5
    q_t, scales = q(t)
    t_rec = dq(q_t, scales)
    assert t_rec.shape == t.shape
    assert (t - t_rec).abs().max().item() < 0.02


def test_zero_tensor():
    q, dq = _import_helpers()
    t = torch.zeros(2, 8)
    q_t, scales = q(t)
    t_rec = dq(q_t, scales)
    assert torch.allclose(t_rec, t)
