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


def test_head_output_shape(lm_weight):
    head = MedusaHead(D_MODEL, lm_weight)
    h = torch.randn(4, D_MODEL)
    out = head(h)
    assert out.shape == (4, VOCAB)


def test_head_w1_init_zero(lm_weight):
    head = MedusaHead(D_MODEL, lm_weight)
    assert head.W1.weight.abs().max().item() == 0.0


def test_head_w2_init_from_lm_head(lm_weight):
    head = MedusaHead(D_MODEL, lm_weight)
    assert torch.allclose(head.W2.weight, lm_weight.float())


def test_model_output_shape(cfg, lm_weight):
    model = MedusaModel(cfg, lm_weight)
    h = torch.randn(4, D_MODEL)
    out = model(h)
    assert out.shape == (4, cfg.n_heads, VOCAB)


def test_model_head_count(cfg, lm_weight):
    model = MedusaModel(cfg, lm_weight)
    assert len(model.heads) == cfg.n_heads


def test_model_trainable_params(cfg, lm_weight):
    model = MedusaModel(cfg, lm_weight)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Each head: W1 (d_model*d_model) + W2 (vocab*d_model)
    expected = cfg.n_heads * (D_MODEL * D_MODEL + VOCAB * D_MODEL)
    assert n == expected
