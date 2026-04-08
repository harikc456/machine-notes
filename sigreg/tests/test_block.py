"""Tests for TransformerBlock — all four use_residual × norm_type combinations."""
import torch
import pytest
from sigreg.config import SIGRegConfig
from sigreg.models.block import TransformerBlock


def _cfg(**kwargs) -> SIGRegConfig:
    """Tiny config for fast tests."""
    return SIGRegConfig(
        d_model=32, n_heads=2, ffn_hidden=64,
        vocab_size=64, seq_len=8,
        **kwargs,
    )


def _x(cfg: SIGRegConfig) -> torch.Tensor:
    return torch.randn(2, cfg.seq_len, cfg.d_model)


def test_plain_output_shape():
    cfg = _cfg()
    out = TransformerBlock(cfg)(_x(cfg))
    assert out.shape == (2, cfg.seq_len, cfg.d_model)


def test_residual_only_output_shape():
    cfg = _cfg(use_residual=True)
    out = TransformerBlock(cfg)(_x(cfg))
    assert out.shape == (2, cfg.seq_len, cfg.d_model)


def test_rmsnorm_no_residual_output_shape():
    cfg = _cfg(norm_type="rmsnorm")
    out = TransformerBlock(cfg)(_x(cfg))
    assert out.shape == (2, cfg.seq_len, cfg.d_model)


def test_prenorm_residual_rmsnorm_output_shape():
    cfg = _cfg(use_residual=True, norm_type="rmsnorm")
    out = TransformerBlock(cfg)(_x(cfg))
    assert out.shape == (2, cfg.seq_len, cfg.d_model)


def test_prenorm_residual_layernorm_output_shape():
    cfg = _cfg(use_residual=True, norm_type="layernorm")
    out = TransformerBlock(cfg)(_x(cfg))
    assert out.shape == (2, cfg.seq_len, cfg.d_model)


def test_plain_has_no_norm_modules():
    block = TransformerBlock(_cfg())
    assert block.norm_attn is None
    assert block.norm_ffn is None


def test_rmsnorm_block_has_norm_modules():
    block = TransformerBlock(_cfg(norm_type="rmsnorm"))
    assert block.norm_attn is not None
    assert block.norm_ffn is not None


def test_layernorm_block_has_norm_modules():
    block = TransformerBlock(_cfg(norm_type="layernorm"))
    assert block.norm_attn is not None
    assert block.norm_ffn is not None
