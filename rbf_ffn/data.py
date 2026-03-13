"""
WikiText-103 data pipeline.

Usage:
    from rbf_ffn.data import get_dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(cfg)

On first call, downloads and tokenises WikiText-103 (~5 minutes).
Subsequent calls load from cache in rbf_ffn/data_cache/.
"""
from __future__ import annotations
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

_CACHE_DIR = Path(__file__).parent / "data_cache"


def chunk_tokens(tokens: list[int], seq_len: int) -> torch.Tensor:
    """
    Split a flat token list into non-overlapping chunks of seq_len.
    The trailing remainder (< seq_len tokens) is discarded.

    Returns: LongTensor of shape (n_chunks, seq_len)
    """
    t = torch.tensor(tokens, dtype=torch.long)
    n_chunks = len(t) // seq_len
    return t[: n_chunks * seq_len].view(n_chunks, seq_len)


class TokenDataset(Dataset):
    """Simple wrapper around a (N, seq_len) LongTensor."""

    def __init__(self, data: torch.Tensor):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


def _load_split(split: str, seq_len: int) -> torch.Tensor:
    """
    Load a tokenised split from cache, or build and cache it.

    split: "train" | "validation" | "test"
    """
    _CACHE_DIR.mkdir(exist_ok=True)
    cache_file = _CACHE_DIR / f"{split}_r50k_{seq_len}.pt"

    if cache_file.exists():
        return torch.load(cache_file, weights_only=True)

    # Lazy imports so the rest of the module works without these packages
    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("r50k_base")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)

    texts = [row["text"] for row in dataset if row["text"].strip() != ""]
    full_text = "\n".join(texts)
    tokens = enc.encode(full_text)

    chunks = chunk_tokens(tokens, seq_len)
    torch.save(chunks, cache_file)
    print(f"Cached {len(chunks):,} sequences → {cache_file}")
    return chunks


def get_dataloaders(cfg) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Return (train_loader, val_loader, test_loader) for WikiText-103.

    cfg must have: seq_len, batch_size
    """
    g = torch.Generator()
    g.manual_seed(cfg.seed)

    def _make_loader(split: str, shuffle: bool, drop_last: bool) -> DataLoader:
        data = _load_split(split, cfg.seq_len)
        ds = TokenDataset(data)
        return DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=4,
            pin_memory=True,
            generator=g if shuffle else None,
        )

    train_loader = _make_loader("train",      shuffle=True,  drop_last=True)
    val_loader   = _make_loader("validation", shuffle=False, drop_last=False)
    test_loader  = _make_loader("test",       shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader