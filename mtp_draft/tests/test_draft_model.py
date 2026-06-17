import torch
import pytest
from mtp_draft.config import MTPConfig
from mtp_draft.models.draft_model import MTPDraftModel

VOCAB, D_TEACHER = 200, 64
B, SEQ_LEN = 2, 32

@pytest.fixture
def cfg():
    return MTPConfig(
        d_draft=64, n_blocks=2, ffn_hidden=128, n_heads=4,
        d_teacher=D_TEACHER, max_draft=4,
        teacher_layers=[0, 1, 2, 3],  # 4 layers = power of 2
        lora_rank=4,
    )

@pytest.fixture
def model(cfg):
    emb_w = torch.randn(VOCAB, D_TEACHER)
    lm_w = torch.randn(VOCAB, D_TEACHER)
    return MTPDraftModel(cfg, emb_w, lm_w)

@pytest.fixture
def inputs(cfg):
    hiddens = torch.randn(B, len(cfg.teacher_layers), cfg.d_teacher)
    ctx_ids = torch.randint(0, VOCAB, (B, SEQ_LEN))
    return hiddens, ctx_ids


def test_output_shape(model, inputs, cfg):
    hiddens, ctx_ids = inputs
    out = model(hiddens, ctx_ids)
    assert out.shape == (B, cfg.max_draft, VOCAB)


def test_frozen_embedding_no_grad(model, inputs):
    hiddens, ctx_ids = inputs
    model(hiddens, ctx_ids).sum().backward()
    assert model.token_embedding.weight.grad is None


def test_frozen_lm_head_weight_no_grad(model, inputs):
    hiddens, ctx_ids = inputs
    model(hiddens, ctx_ids).sum().backward()
    assert model.lm_head.weight.grad is None


def test_trainable_params_have_grad(model, inputs):
    hiddens, ctx_ids = inputs
    model(hiddens, ctx_ids).sum().backward()
    for p in model.trainable_parameters():
        assert p.grad is not None


def test_param_count_under_50m(model):
    n = sum(p.numel() for p in model.trainable_parameters())
    assert n < 50_000_000, f"Trainable params {n:,} exceed 50M"
