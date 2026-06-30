"""
Tests for superbpe_data.py.

Tokenizer training tests use a tiny corpus and tiny vocab sizes so they
complete in < 1 second. Data pipeline tests mock the tokenizer and I/O.
"""
from __future__ import annotations
import torch
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# ── Tiny-corpus helpers ───────────────────────────────────────────────────────

_TINY_TEXTS = [
    "the quick brown fox jumps over the lazy dog",
    "pack my box with five dozen liquor jugs",
    "how vexingly quick daft zebras jump",
    "sphinx of black quartz judge my vow",
] * 100   # repeat for sufficient frequency counts

_TRANSITION = 50    # small enough for < 1s training
_VOCAB_SIZE = 70    # 20 superword slots above the subword stage


# ── _train_stage1 ─────────────────────────────────────────────────────────────

def test_train_stage1_reaches_target_vocab_size():
    from rbf_ffn.superbpe_data import _train_stage1
    tok = _train_stage1(_TINY_TEXTS, transition=_TRANSITION)
    assert tok.get_vocab_size() == _TRANSITION


def test_train_stage1_encodes_text():
    from rbf_ffn.superbpe_data import _train_stage1
    tok = _train_stage1(_TINY_TEXTS, transition=_TRANSITION)
    enc = tok.encode("the quick brown fox")
    assert len(enc.ids) > 0
    assert all(isinstance(i, int) for i in enc.ids)


def test_train_stage1_ids_within_vocab():
    from rbf_ffn.superbpe_data import _train_stage1
    tok = _train_stage1(_TINY_TEXTS, transition=_TRANSITION)
    enc = tok.encode("hello world")
    assert all(0 <= i < _TRANSITION for i in enc.ids)


# ── _train_stage2 ─────────────────────────────────────────────────────────────

def test_train_stage2_reaches_target_vocab_size():
    from rbf_ffn.superbpe_data import _train_stage1, _train_stage2
    tok1 = _train_stage1(_TINY_TEXTS, transition=_TRANSITION)
    tok2 = _train_stage2(tok1, _TINY_TEXTS, vocab_size=_VOCAB_SIZE)
    assert tok2.get_vocab_size() == _VOCAB_SIZE


def test_train_stage2_encodes_text():
    from rbf_ffn.superbpe_data import _train_stage1, _train_stage2
    tok1 = _train_stage1(_TINY_TEXTS, transition=_TRANSITION)
    tok2 = _train_stage2(tok1, _TINY_TEXTS, vocab_size=_VOCAB_SIZE)
    enc = tok2.encode("the quick brown fox")
    assert len(enc.ids) > 0
    assert all(0 <= i < _VOCAB_SIZE for i in enc.ids)


def test_train_stage2_ids_within_extended_vocab():
    from rbf_ffn.superbpe_data import _train_stage1, _train_stage2
    tok1 = _train_stage1(_TINY_TEXTS, transition=_TRANSITION)
    tok2 = _train_stage2(tok1, _TINY_TEXTS, vocab_size=_VOCAB_SIZE)
    enc = tok2.encode("pack my box with five dozen")
    assert all(0 <= i < _VOCAB_SIZE for i in enc.ids)


# ── _build_superbpe_tokenizer ─────────────────────────────────────────────────

def test_build_superbpe_tokenizer_creates_cache_files(tmp_path):
    from rbf_ffn.superbpe_data import _build_superbpe_tokenizer
    with patch("rbf_ffn.superbpe_data._load_wikitext_split_texts", return_value=_TINY_TEXTS), \
         patch("rbf_ffn.superbpe_data._VOCAB_SIZE", _VOCAB_SIZE), \
         patch("rbf_ffn.superbpe_data._TRANSITION", _TRANSITION):
        _build_superbpe_tokenizer(tmp_path)
    stage1 = tmp_path / "superbpe48576_tokenizer" / "stage1.json"
    stage2 = tmp_path / "superbpe48576_tokenizer" / "stage2.json"
    assert stage1.exists()
    assert stage2.exists()


def test_build_superbpe_tokenizer_loads_from_cache(tmp_path):
    from rbf_ffn.superbpe_data import _build_superbpe_tokenizer
    with patch("rbf_ffn.superbpe_data._load_wikitext_split_texts", return_value=_TINY_TEXTS), \
         patch("rbf_ffn.superbpe_data._VOCAB_SIZE", _VOCAB_SIZE), \
         patch("rbf_ffn.superbpe_data._TRANSITION", _TRANSITION):
        tok_first = _build_superbpe_tokenizer(tmp_path)
        # Second call must load from cache without re-training
        with patch("rbf_ffn.superbpe_data._train_stage1") as mock_s1, \
             patch("rbf_ffn.superbpe_data._train_stage2") as mock_s2:
            tok_second = _build_superbpe_tokenizer(tmp_path)
            mock_s1.assert_not_called()
            mock_s2.assert_not_called()
    assert tok_second.get_vocab_size() == tok_first.get_vocab_size()


# ── _load_split ───────────────────────────────────────────────────────────────

def test_load_split_returns_correct_shape(tmp_path):
    from rbf_ffn.superbpe_data import _load_split, _train_stage1
    tok = _train_stage1(_TINY_TEXTS, transition=_TRANSITION)
    texts = _TINY_TEXTS[:10]
    with patch("rbf_ffn.superbpe_data._load_wikitext_split_texts", return_value=texts), \
         patch("rbf_ffn.superbpe_data._CACHE_DIR", tmp_path):
        chunks = _load_split("train", seq_len=16, tokenizer=tok)
    assert chunks.ndim == 2
    assert chunks.shape[1] == 16
    assert chunks.dtype == torch.long


def test_load_split_caches_to_disk(tmp_path):
    from rbf_ffn.superbpe_data import _load_split, _train_stage1
    tok = _train_stage1(_TINY_TEXTS, transition=_TRANSITION)
    texts = _TINY_TEXTS[:10]
    with patch("rbf_ffn.superbpe_data._load_wikitext_split_texts", return_value=texts), \
         patch("rbf_ffn.superbpe_data._CACHE_DIR", tmp_path):
        _load_split("train", seq_len=16, tokenizer=tok)
        cache_file = tmp_path / "train_superbpe48576_16.pt"
        assert cache_file.exists()


# ── get_dataloaders ───────────────────────────────────────────────────────────

def _fake_tokenizer():
    """Minimal mock tokenizer for DataLoader tests."""
    tok = MagicMock()
    tok.encode_batch.return_value = [MagicMock(ids=[0, 1, 2, 3]) for _ in range(10)]
    return tok


def test_get_dataloaders_returns_three_loaders(tmp_path):
    from rbf_ffn.superbpe_data import get_dataloaders
    from rbf_ffn.config import ModelConfig
    cfg = ModelConfig(tokenizer="superbpe48576", vocab_size=48576, seq_len=8, batch_size=2)
    fake_chunks = torch.zeros(32, 8, dtype=torch.long)
    with patch("rbf_ffn.superbpe_data._build_superbpe_tokenizer", return_value=_fake_tokenizer()), \
         patch("rbf_ffn.superbpe_data._load_split", return_value=fake_chunks):
        train_loader, val_loader, test_loader = get_dataloaders(cfg)
    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None


def test_get_dataloaders_batch_size(tmp_path):
    from rbf_ffn.superbpe_data import get_dataloaders
    from rbf_ffn.config import ModelConfig
    cfg = ModelConfig(tokenizer="superbpe48576", vocab_size=48576, seq_len=8, batch_size=4)
    fake_chunks = torch.zeros(32, 8, dtype=torch.long)
    with patch("rbf_ffn.superbpe_data._build_superbpe_tokenizer", return_value=_fake_tokenizer()), \
         patch("rbf_ffn.superbpe_data._load_split", return_value=fake_chunks):
        train_loader, _, _ = get_dataloaders(cfg)
    assert train_loader.batch_size == 4
