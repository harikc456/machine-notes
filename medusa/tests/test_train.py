from __future__ import annotations
import math
import torch
import pytest

from medusa.config import MedusaConfig
from medusa.models.medusa_model import MedusaModel


VOCAB = 64
D_MODEL = 32
N_HEADS = 3
B = 4


@pytest.fixture
def cfg():
    return MedusaConfig(n_heads=N_HEADS, d_model=D_MODEL, lambda_decay=0.8, grad_clip=1.0)


@pytest.fixture
def model(cfg):
    lm_w = torch.randn(VOCAB, D_MODEL)
    return MedusaModel(cfg, lm_w)


def test_training_step_returns_scalar(cfg, model):
    from medusa.train import training_step
    from torch.optim import AdamW
    opt = AdamW(model.parameters(), lr=1e-3)
    hidden = torch.randn(B, D_MODEL)
    targets = torch.randint(0, VOCAB, (B, N_HEADS))
    loss = training_step(model, hidden, targets, cfg, [opt])
    assert loss.shape == ()
    assert loss.item() > 0


def test_training_step_updates_w1(cfg, model):
    from medusa.train import training_step
    from torch.optim import AdamW
    opt = AdamW(model.parameters(), lr=1e-3)
    w_before = model.heads[0].W1.weight.clone()
    hidden = torch.randn(B, D_MODEL)
    targets = torch.randint(0, VOCAB, (B, N_HEADS))
    training_step(model, hidden, targets, cfg, [opt])
    assert not torch.equal(model.heads[0].W1.weight, w_before)


def test_training_step_ignores_minus100(cfg, model):
    from medusa.train import training_step
    from torch.optim import AdamW
    opt = AdamW(model.parameters(), lr=1e-3)
    hidden = torch.randn(B, D_MODEL)
    targets = torch.full((B, N_HEADS), -100, dtype=torch.long)
    loss = training_step(model, hidden, targets, cfg, [opt])
    assert not torch.isnan(loss)


def test_make_lr_lambda():
    from medusa.train import make_lr_lambda
    fn = make_lr_lambda(warmup_steps=100, total_steps=1000)
    assert fn(0) == pytest.approx(0.0)
    assert fn(50) == pytest.approx(0.5)
    assert fn(100) == pytest.approx(1.0)
    assert 0.0 < fn(550) < 1.0
    assert fn(1000) == pytest.approx(0.0, abs=1e-6)
