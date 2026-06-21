from __future__ import annotations
import torch
import pytest
from medusa.config import MedusaConfig
from medusa.models.medusa_model import MedusaHead, MedusaModel


VOCAB = 64
D_MODEL = 32


@pytest.fixture
def lm_weight():
    return torch.randn(VOCAB, D_MODEL)


@pytest.fixture
def cfg():
    return MedusaConfig(n_heads=3, d_model=D_MODEL)


def test_head_output_shape():
    head = MedusaHead(D_MODEL)
    h = torch.randn(4, D_MODEL)
    out = head(h)
    assert out.shape == (4, D_MODEL)


def test_head_w1_init_zero():
    head = MedusaHead(D_MODEL)
    assert head.W1.weight.abs().max().item() == 0.0


def test_model_lm_head_frozen(cfg, lm_weight):
    model = MedusaModel(cfg, lm_weight)
    assert torch.allclose(model.lm_head_weight, lm_weight)
    assert not model.lm_head_weight.requires_grad


def test_model_output_shape(cfg, lm_weight):
    model = MedusaModel(cfg, lm_weight)
    h = torch.randn(4, D_MODEL)
    out = model(h)
    assert out.shape == (4, cfg.n_heads, D_MODEL)


def test_model_get_logits_shape(cfg, lm_weight):
    model = MedusaModel(cfg, lm_weight)
    h = torch.randn(4, D_MODEL)
    logits = model.get_logits(h)
    assert logits.shape == (4, cfg.n_heads, VOCAB)


def test_model_head_count(cfg, lm_weight):
    model = MedusaModel(cfg, lm_weight)
    assert len(model.heads) == cfg.n_heads


def test_model_trainable_params(cfg, lm_weight):
    model = MedusaModel(cfg, lm_weight)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Each head: W1 only (d_model × d_model); lm_head_weight is a frozen buffer
    expected = cfg.n_heads * D_MODEL * D_MODEL
    assert n == expected
