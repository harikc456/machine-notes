"""Tests for collect_hidden inference from sigreg_weight."""
import torch
from sigreg.config import SIGRegConfig
from sigreg.models.model import SIGRegCausalLM


def _cfg(**kwargs) -> SIGRegConfig:
    return SIGRegConfig(
        d_model=32, n_heads=2, n_layers=2, ffn_hidden=64,
        vocab_size=64, seq_len=8,
        **kwargs,
    )


def test_collect_hidden_false_when_weight_zero():
    """sigreg_weight=0.0 must produce collect_hidden=False → empty hidden list."""
    cfg = _cfg(sigreg_weight=0.0)
    assert (cfg.sigreg_weight > 0.0) is False  # expression used in train.py

    model = SIGRegCausalLM(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
    _, hidden = model(tokens, collect_hidden=(cfg.sigreg_weight > 0.0))

    assert hidden == []


def test_collect_hidden_true_when_weight_nonzero():
    """sigreg_weight>0 must produce collect_hidden=True → hidden states returned."""
    cfg = _cfg(sigreg_weight=0.1)
    assert (cfg.sigreg_weight > 0.0) is True  # expression used in train.py

    model = SIGRegCausalLM(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
    _, hidden = model(tokens, collect_hidden=(cfg.sigreg_weight > 0.0))

    assert len(hidden) == cfg.n_layers
