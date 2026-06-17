import torch
import tempfile
from pathlib import Path
import pytest
from mtp_draft.cache import _quantise_int8, _dequantise_int8


def test_quantise_roundtrip():
    x = torch.randn(4, 2048)
    q, scale = _quantise_int8(x)
    assert q.dtype == torch.int8
    x_hat = _dequantise_int8(q, scale)
    # Max absolute error should be small relative to data range
    assert (x - x_hat).abs().max() < x.abs().max() * 0.02


def test_quantise_int8_range():
    x = torch.randn(4, 2048) * 10
    q, scale = _quantise_int8(x)
    assert q.abs().max() <= 127


def test_scale_positive():
    x = torch.randn(4, 2048)
    _, scale = _quantise_int8(x)
    assert scale > 0
