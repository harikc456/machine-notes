"""
Tests for the data pipeline. These tests do NOT download WikiText-103.
They verify the chunking logic and Dataset contract using synthetic data.
"""
import torch
import pytest
from rbf_ffn.data import chunk_tokens, TokenDataset


def test_chunk_tokens_basic():
    tokens = list(range(20))
    chunks = chunk_tokens(tokens, seq_len=5)
    assert chunks.shape == (4, 5)
    assert chunks[0].tolist() == [0, 1, 2, 3, 4]
    assert chunks[3].tolist() == [15, 16, 17, 18, 19]


def test_chunk_tokens_discards_remainder():
    tokens = list(range(22))   # 22 tokens, seq_len=5 → 4 full chunks, 2 discarded
    chunks = chunk_tokens(tokens, seq_len=5)
    assert chunks.shape == (4, 5)


def test_chunk_tokens_exact_multiple():
    tokens = list(range(15))
    chunks = chunk_tokens(tokens, seq_len=5)
    assert chunks.shape == (3, 5)


def test_token_dataset_len():
    data = torch.arange(40).view(8, 5)   # 8 sequences of length 5
    ds = TokenDataset(data)
    assert len(ds) == 8


def test_token_dataset_getitem():
    data = torch.arange(40).view(8, 5)
    ds = TokenDataset(data)
    item = ds[0]
    assert item.shape == (5,)
    assert item.dtype == torch.long


def test_token_dataset_values():
    data = torch.arange(40, dtype=torch.long).view(8, 5)
    ds = TokenDataset(data)
    assert ds[2].tolist() == [10, 11, 12, 13, 14]


from unittest.mock import patch
from rbf_ffn.data import get_dataloaders
from rbf_ffn.config import RBFFFNConfig


def _fake_load_split(split, seq_len):
    """Synthetic tensor so tests don't touch disk or network."""
    return torch.zeros(32, seq_len, dtype=torch.long)


def test_train_loader_has_persistent_workers():
    cfg = RBFFFNConfig(seq_len=16, batch_size=4)
    with patch("rbf_ffn.data._load_split", side_effect=_fake_load_split):
        train_loader, _, _ = get_dataloaders(cfg)
    assert train_loader.persistent_workers is True


def test_train_loader_has_prefetch_factor():
    cfg = RBFFFNConfig(seq_len=16, batch_size=4)
    with patch("rbf_ffn.data._load_split", side_effect=_fake_load_split):
        train_loader, _, _ = get_dataloaders(cfg)
    assert train_loader.prefetch_factor == 2


def test_val_and_test_loaders_unchanged():
    """Val and test loaders must NOT have persistent_workers enabled."""
    cfg = RBFFFNConfig(seq_len=16, batch_size=4)
    with patch("rbf_ffn.data._load_split", side_effect=_fake_load_split):
        _, val_loader, test_loader = get_dataloaders(cfg)
    assert val_loader.persistent_workers is False
    assert test_loader.persistent_workers is False