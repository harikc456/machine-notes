"""Tests for kv_quant.calibrate — validates the new spectralquant save format."""
from __future__ import annotations
import os, sys, tempfile
import torch
import pytest

_SPECTRALQUANT_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "spectralquant", "src")
)
if _SPECTRALQUANT_SRC not in sys.path:
    sys.path.insert(0, _SPECTRALQUANT_SRC)


def _make_fake_kv(n_layers=2, n_kv_heads=2, n_tokens=50, head_dim=64):
    """Synthetic K/V tensors simulating DynamicCache output."""
    torch.manual_seed(0)
    # all_keys[l] = list of (n_kv_heads, seq, head_dim) tensors
    all_keys = [[torch.randn(n_kv_heads, n_tokens, head_dim)] for _ in range(n_layers)]
    all_vals = [[torch.randn(n_kv_heads, n_tokens, head_dim)] for _ in range(n_layers)]
    return all_keys, all_vals


def test_save_format_files_exist():
    """calibrate() produces three files: .pt, _meta.json, _quantizers.pt."""
    from kv_quant.calibrate import _calibrate_from_kv
    all_keys, all_vals = _make_fake_kv()
    with tempfile.TemporaryDirectory() as tmpdir:
        base = os.path.join(tmpdir, "cal")
        _calibrate_from_kv(all_keys, all_vals, head_dim=64, bits=4, base_path=base)
        assert os.path.exists(base + ".pt")
        assert os.path.exists(base + "_meta.json")
        assert os.path.exists(base + "_quantizers.pt")


def test_quant_state_keys():
    """quant_states dict has keys 'L{l}_H{h}_key' and 'L{l}_H{h}_value'."""
    from kv_quant.calibrate import _calibrate_from_kv
    all_keys, all_vals = _make_fake_kv(n_layers=1, n_kv_heads=2)
    with tempfile.TemporaryDirectory() as tmpdir:
        base = os.path.join(tmpdir, "cal")
        _calibrate_from_kv(all_keys, all_vals, head_dim=64, bits=4, base_path=base)
        qs = torch.load(base + "_quantizers.pt", map_location="cpu", weights_only=True)
        assert "L0_H0_key" in qs
        assert "L0_H1_key" in qs
        assert "L0_H0_value" in qs


def test_quant_state_fields():
    """Each quant_state entry has the required fields."""
    from kv_quant.calibrate import _calibrate_from_kv
    all_keys, all_vals = _make_fake_kv(n_layers=1, n_kv_heads=1)
    with tempfile.TemporaryDirectory() as tmpdir:
        base = os.path.join(tmpdir, "cal")
        _calibrate_from_kv(all_keys, all_vals, head_dim=64, bits=4, base_path=base)
        qs = torch.load(base + "_quantizers.pt", map_location="cpu", weights_only=True)
        state = qs["L0_H0_key"]
        assert "semantic_centroids" in state
        assert "tail_centroids" in state
        assert "d_eff_int" in state
        assert "b_high" in state
        assert "b_low" in state
        assert "head_dim" in state
        assert state["head_dim"] == 64
        assert 1 <= state["b_high"] <= 8
        assert 1 <= state["b_low"] <= 8


def test_calibrator_can_be_reloaded():
    """EigenspectralCalibrator.load() can reload the .pt + _meta.json."""
    from kv_quant.calibrate import _calibrate_from_kv
    from spectralquant.calibration import EigenspectralCalibrator
    all_keys, all_vals = _make_fake_kv(n_layers=1, n_kv_heads=1)
    with tempfile.TemporaryDirectory() as tmpdir:
        base = os.path.join(tmpdir, "cal")
        _calibrate_from_kv(all_keys, all_vals, head_dim=64, bits=4, base_path=base)
        cal = EigenspectralCalibrator()
        cal.load(base)
        assert cal._is_calibrated
        hcd = cal.get(0, 0, "key")
        assert hcd is not None
        assert hcd.eigenvectors.shape == (64, 64)
        assert hcd.eigenvalues.shape == (64,)
