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

    import os
    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("r50k_base")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    texts = [row["text"] for row in dataset if row["text"].strip() != ""]
    # encode_batch parallelises across threads; flatten and join with newline token
    nl_tok = enc.encode("\n")[0]
    encoded = enc.encode_batch(texts, num_threads=os.cpu_count() or 4)
    tokens: list[int] = []
    for toks in encoded:
        tokens.extend(toks)
        tokens.append(nl_tok)
    chunks = chunk_tokens(tokens, seq_len)
    torch.save(chunks, cache_file)
    print(f"Cached {len(chunks):,} sequences → {cache_file}")
    return chunks


def get_dataloaders(cfg) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader). cfg needs seq_len, batch_size, seed."""
    g = torch.Generator()
    g.manual_seed(cfg.seed)

    def _make(split: str, shuffle: bool, drop_last: bool) -> DataLoader:
        ds = TokenDataset(_load_split(split, cfg.seq_len))
        # Dataset is fully in-memory: num_workers=0 avoids spawning worker processes
        # and the IPC overhead they introduce for simple index lookups.
        return DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=shuffle, drop_last=drop_last,
            num_workers=0, pin_memory=True,
            generator=g if shuffle else None,
        )

    return (
        _make("train",      shuffle=True,  drop_last=True),
        _make("validation", shuffle=False, drop_last=False),
        _make("test",       shuffle=False, drop_last=False),
    )
