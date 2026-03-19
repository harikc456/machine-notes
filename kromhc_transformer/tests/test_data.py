import pytest
import torch
from kromhc_transformer.data import get_dataloaders, chunk_tokens
from kromhc_transformer.config import KromHCConfig

def test_chunk_tokens():
    tokens = list(range(100))
    chunks = chunk_tokens(tokens, seq_len=10)
    assert chunks.shape == (10, 10)
    assert chunks.dtype == torch.long

def test_chunk_tokens_discards_remainder():
    tokens = list(range(105))  # 105 tokens, seq_len=10 → 10 chunks of 10, 5 discarded
    chunks = chunk_tokens(tokens, seq_len=10)
    assert chunks.shape == (10, 10)

def test_get_dataloaders():
    """Smoke test: dataloaders load without error."""
    cfg = KromHCConfig(seq_len=512, batch_size=8)
    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None
    batch = next(iter(train_loader))
    assert batch.shape == (8, 512)
    assert batch.dtype == torch.long
