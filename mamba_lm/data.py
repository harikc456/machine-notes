"""
WikiText-103 data pipeline for mamba_lm.

On first call, downloads and tokenises WikiText-103 (~5 minutes).
Subsequent calls load from cache in mamba_lm/data_cache/.
"""
from __future__ import annotations
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

_CACHE_DIR = Path(__file__).parent / "data_cache"


def chunk_tokens(tokens: list[int], seq_len: int) -> torch.Tensor:
    t = torch.tensor(tokens, dtype=torch.long)
    n_chunks = len(t) // seq_len
    return t[: n_chunks * seq_len].view(n_chunks, seq_len)


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
    g = torch.Generator()
    g.manual_seed(cfg.seed)

    def _make(split: str, shuffle: bool, drop_last: bool,
               persistent: bool = False, prefetch: int | None = None) -> DataLoader:
        data = _load_split(split, cfg.seq_len)
        return DataLoader(
            TokenDataset(data),
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=4,
            pin_memory=True,
            generator=g if shuffle else None,
            persistent_workers=persistent,
            prefetch_factor=prefetch,
        )

    return (
        _make("train",      shuffle=True,  drop_last=True,  persistent=True, prefetch=2),
        _make("validation", shuffle=False, drop_last=False),
        _make("test",       shuffle=False, drop_last=False),
    )
