"""Smoke tests for SIGRegCausalLM with new residual/norm flags."""
import torch
from sigreg.config import SIGRegConfig
from sigreg.models.model import SIGRegCausalLM


def _cfg(**kwargs) -> SIGRegConfig:
    return SIGRegConfig(
        d_model=32, n_heads=2, n_layers=3, ffn_hidden=64,
        vocab_size=64, seq_len=8,
        **kwargs,
    )


def _tokens(cfg: SIGRegConfig) -> torch.Tensor:
    return torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))


def test_plain_model_forward():
    cfg = _cfg()
    model = SIGRegCausalLM(cfg)
    logits, hidden = model(_tokens(cfg), collect_hidden=True)
    assert logits.shape == (2, cfg.seq_len, cfg.vocab_size)
    assert len(hidden) == cfg.n_layers


def test_prenorm_residual_model_forward():
    cfg = _cfg(use_residual=True, norm_type="rmsnorm")
    model = SIGRegCausalLM(cfg)
    logits, hidden = model(_tokens(cfg), collect_hidden=True)
    assert logits.shape == (2, cfg.seq_len, cfg.vocab_size)
    assert len(hidden) == cfg.n_layers


def test_plain_model_collect_hidden_false():
    cfg = _cfg()
    model = SIGRegCausalLM(cfg)
    logits, hidden = model(_tokens(cfg), collect_hidden=False)
    assert logits.shape == (2, cfg.seq_len, cfg.vocab_size)
    assert hidden == []
