from __future__ import annotations
import torch
import pytest

from medusa.cache import _extract_conversation
from medusa.config import MedusaConfig


def _import_helpers():
    from medusa.cache import _quantise_int8, _dequantise_int8
    return _quantise_int8, _dequantise_int8


class _FakeTok:
    """Minimal tokenizer stub for testing _extract_conversation."""
    def encode(self, text, add_special_tokens=True):
        return list(range(len(text.split())))

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return messages[0]["content"]


@pytest.fixture
def tok():
    return _FakeTok()


@pytest.fixture
def base_cfg():
    return MedusaConfig(max_seq_len=20, max_answer_len=5, n_heads=3)


def test_extract_conversation_basic(tok, base_cfg):
    example = {"conversations": [
        {"from": "human", "value": "hello world"},
        {"from": "gpt", "value": "hi there friend"},
    ]}
    prompt_ids, answer_ids = _extract_conversation(example, tok, base_cfg)
    assert len(prompt_ids) > 0
    assert len(answer_ids) > 0


def test_extract_conversation_missing_turns_returns_empty(tok, base_cfg):
    example = {"conversations": [{"from": "human", "value": "hello"}]}
    prompt_ids, answer_ids = _extract_conversation(example, tok, base_cfg)
    assert prompt_ids == []
    assert answer_ids == []


def test_extract_conversation_prompt_truncated(tok, base_cfg):
    long_answer = " ".join([f"word{i}" for i in range(base_cfg.max_answer_len)])
    example = {"conversations": [
        {"from": "human", "value": " ".join([f"w{i}" for i in range(100)])},
        {"from": "gpt", "value": long_answer},
    ]}
    prompt_ids, answer_ids = _extract_conversation(example, tok, base_cfg)
    total = len(prompt_ids) + min(base_cfg.max_answer_len, len(answer_ids))
    assert total <= base_cfg.max_seq_len


def test_quantise_shape():
    q = _import_helpers()[0]
    t = torch.randn(4, 32)
    q_t, scales = q(t)
    assert q_t.shape == t.shape
    assert q_t.dtype == torch.int8
    assert scales.shape == (4,)


def test_quantise_range():
    q, _ = _import_helpers()
    t = torch.randn(8, 16)
    q_t, _ = q(t)
    assert q_t.abs().max().item() <= 127


def test_roundtrip_close():
    q, dq = _import_helpers()
    torch.manual_seed(42)
    t = torch.randn(6, 64) * 0.5
    q_t, scales = q(t)
    t_rec = dq(q_t, scales)
    assert t_rec.shape == t.shape
    assert (t - t_rec).abs().max().item() < 0.02


def test_zero_tensor():
    q, dq = _import_helpers()
    t = torch.zeros(2, 8)
    q_t, scales = q(t)
    t_rec = dq(q_t, scales)
    assert torch.allclose(t_rec, t)
