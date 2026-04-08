"""Tests for SIGRegConfig use_residual and norm_type fields."""
import pytest
from sigreg.config import SIGRegConfig


def test_config_defaults_are_plain():
    cfg = SIGRegConfig()
    assert cfg.use_residual is False
    assert cfg.norm_type == "none"


def test_config_accepts_valid_norm_types():
    for nt in ("none", "rmsnorm", "layernorm"):
        cfg = SIGRegConfig(norm_type=nt)
        assert cfg.norm_type == nt


def test_config_rejects_invalid_norm_type():
    with pytest.raises(AssertionError):
        SIGRegConfig(norm_type="batchnorm")


def test_config_accepts_use_residual_true():
    cfg = SIGRegConfig(use_residual=True)
    assert cfg.use_residual is True
