"""Smoke tests for mtp_draft.train — all use synthetic data, no teacher loading."""
from __future__ import annotations
import math

import pytest
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from mtp_draft.config import MTPConfig
from mtp_draft.models.draft_model import MTPDraftModel
from mtp_draft.train import make_lr_lambda, training_step

VOCAB = 100
D_TEACHER = 32
B = 2


@pytest.fixture
def cfg():
    return MTPConfig(
        d_draft=32,
        n_blocks=1,
        ffn_hidden=64,
        n_heads=2,
        d_teacher=D_TEACHER,
        max_draft=3,
        teacher_layers=[0, 1, 2, 3],
        lora_rank=2,
        dropout=0.0,
    )


@pytest.fixture
def model(cfg):
    emb_w = torch.randn(VOCAB, D_TEACHER)
    lm_w = torch.randn(VOCAB, D_TEACHER)
    return MTPDraftModel(cfg, emb_w, lm_w)


# ---------------------------------------------------------------------------
# training_step tests
# ---------------------------------------------------------------------------


def test_training_step_returns_finite_loss(model, cfg):
    """One forward+backward step must not produce NaN or Inf."""
    hiddens = torch.randn(B, len(cfg.teacher_layers), cfg.d_teacher)
    ctx_ids = torch.randint(0, VOCAB, (B, cfg.max_prompt_len))
    targets = torch.randint(0, VOCAB, (B, cfg.max_draft))
    opt = AdamW(model.trainable_parameters(), lr=1e-3)

    loss = training_step(model, hiddens, ctx_ids, targets, cfg, opt)
    assert loss.ndim == 0, "loss must be a scalar"
    assert torch.isfinite(loss), f"loss is not finite: {loss.item()}"


def test_train_reduces_loss(cfg):
    """Loss at step 3 should be lower than at step 1 with a large LR on small data."""
    torch.manual_seed(0)
    emb_w = torch.randn(VOCAB, D_TEACHER)
    lm_w = torch.randn(VOCAB, D_TEACHER)
    m = MTPDraftModel(cfg, emb_w, lm_w)

    hiddens = torch.randn(B, len(cfg.teacher_layers), cfg.d_teacher)
    ctx_ids = torch.randint(0, VOCAB, (B, cfg.max_prompt_len))
    targets = torch.randint(0, VOCAB, (B, cfg.max_draft))
    opt = AdamW(m.trainable_parameters(), lr=1e-2)

    losses = []
    for _ in range(3):
        loss = training_step(m, hiddens, ctx_ids, targets, cfg, opt)
        losses.append(loss.item())

    assert losses[2] < losses[0], (
        f"Expected loss to decrease: step1={losses[0]:.4f}, step3={losses[2]:.4f}"
    )


# ---------------------------------------------------------------------------
# make_lr_lambda tests
# ---------------------------------------------------------------------------


def test_lr_lambda_warmup():
    """At step 0, lr_lambda should be 0; at step warmup_steps, it should be 1.0."""
    warmup_steps = 10
    total_steps = 100
    lr_lambda = make_lr_lambda(warmup_steps, total_steps)

    assert lr_lambda(0) == pytest.approx(0.0), "lr_lambda(0) must be 0"
    assert lr_lambda(warmup_steps) == pytest.approx(1.0), (
        "lr_lambda(warmup_steps) must be 1.0"
    )


def test_lr_lambda_cosine_decay():
    """After warmup, lr decays smoothly and reaches ~0 at total_steps."""
    warmup_steps = 10
    total_steps = 100
    lr_lambda = make_lr_lambda(warmup_steps, total_steps)

    # Should be monotonically decreasing after warmup
    vals = [lr_lambda(s) for s in range(warmup_steps, total_steps + 1)]
    for i in range(1, len(vals)):
        assert vals[i] <= vals[i - 1] + 1e-9, (
            f"LR not monotonically decreasing at step {warmup_steps + i}: "
            f"{vals[i - 1]:.4f} -> {vals[i]:.4f}"
        )

    # At total_steps, lr should be approximately 0
    assert lr_lambda(total_steps) == pytest.approx(0.0, abs=1e-6), (
        f"lr_lambda(total_steps) should be ~0, got {lr_lambda(total_steps)}"
    )
