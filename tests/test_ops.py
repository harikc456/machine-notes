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
