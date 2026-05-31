"""
WikiText-103 data pipeline for I-DLM.

Reuses rbf_ffn's token cache (same seq_len → same cache file).
Cache lives at rbf_ffn/data_cache/.
"""
from __future__ import annotations
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

# Reuse rbf_ffn's existing cache to avoid re-downloading
_CACHE_DIR = Path(__file__).parent.parent / "rbf_ffn" / "data_cache"


def chunk_tokens(tokens: list[int], seq_len: int) -> torch.Tensor:
    t = torch.tensor(tokens, dtype=torch.long)
    n = len(t) // seq_len
    return t[: n * seq_len].view(n, seq_len)


class TokenDataset(Dataset):
    def __init__(self, data: torch.Tensor):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


def _load_split(split: str, seq_len: int) -> torch.Tensor:
    _CACHE_DIR.mkdir(exist_ok=True)
    cache_file = _CACHE_DIR / f"{split}_r50k_{seq_len}.pt"
    if cache_file.exists():
        return torch.load(cache_file, weights_only=True)

    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("r50k_base")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    texts = [row["text"] for row in dataset if row["text"].strip() != ""]
    tokens = enc.encode("\n".join(texts))
    chunks = chunk_tokens(tokens, seq_len)
    torch.save(chunks, cache_file)
    print(f"Cached {len(chunks):,} sequences → {cache_file}")
    return chunks


def get_dataloaders(cfg) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader). cfg needs seq_len, batch_size, seed."""
    g = torch.Generator()
    g.manual_seed(cfg.seed)

    def _make(split: str, shuffle: bool, drop_last: bool,
               persistent: bool = False, prefetch: int | None = None) -> DataLoader:
        ds = TokenDataset(_load_split(split, cfg.seq_len))
        return DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=shuffle, drop_last=drop_last,
            num_workers=4, pin_memory=True,
            generator=g if shuffle else None,
            persistent_workers=persistent,
            prefetch_factor=prefetch,
        )

    return (
        _make("train",      shuffle=True,  drop_last=True,  persistent=True, prefetch=2),
        _make("validation", shuffle=False, drop_last=False),
        _make("test",       shuffle=False, drop_last=False),
    )
