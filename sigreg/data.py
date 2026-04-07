"""
WikiText-103 data pipeline for sigreg experiments.

Uses a custom BPE tokenizer with vocab_size=65536, trained on the WikiText-103
training split on first use and cached to rbf_ffn/data_cache/.

Usage:
    from sigreg.data import get_dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(cfg)

cfg must have: seq_len (int), batch_size (int), seed (int)

On first call, this trains the BPE tokenizer (~5 min) and tokenises each split.
Subsequent calls load from cache instantly.
"""
from __future__ import annotations
from pathlib import Path

import torch
from tokenizers import ByteLevelBPETokenizer
from torch.utils.data import DataLoader

from rbf_ffn.data import chunk_tokens, TokenDataset

# Shared cache directory — reuses the same location as rbf_ffn to avoid
# downloading WikiText-103 twice.
_CACHE_DIR = Path(__file__).parent.parent / "rbf_ffn" / "data_cache"

_VOCAB_SIZE = 65536


def _load_wikitext_split_texts(split: str) -> list[str]:
    """Download a WikiText-103 split and return non-empty text rows."""
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    return [row["text"] for row in dataset if row["text"].strip()]


def _build_tokenizer(cache_dir: Path) -> ByteLevelBPETokenizer:
    """
    Return a ByteLevelBPETokenizer with vocab_size=65536.

    Loads from cache_dir/bpe65536_tokenizer/ if already trained;
    otherwise trains on WikiText-103 train split and saves to that directory.
    """
    tok_dir = cache_dir / "bpe65536_tokenizer"
    vocab_file = tok_dir / "vocab.json"
    merges_file = tok_dir / "merges.txt"

    if vocab_file.exists() and merges_file.exists():
        return ByteLevelBPETokenizer(str(vocab_file), str(merges_file))

    print("Training BPE tokenizer (vocab_size=65536) on WikiText-103 train split…")
    texts = _load_wikitext_split_texts("train")
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        texts,
        vocab_size=_VOCAB_SIZE,
        min_frequency=2,
        special_tokens=[],
    )
    tok_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_model(str(tok_dir))
    print(f"Tokenizer saved → {tok_dir}")
    return tokenizer


def _load_split(split: str, seq_len: int, tokenizer: ByteLevelBPETokenizer) -> torch.Tensor:
    """
    Load a tokenised WikiText-103 split as a (N, seq_len) LongTensor.

    Builds and caches to _CACHE_DIR/{split}_bpe65536_{seq_len}.pt on first call.
    """
    _CACHE_DIR.mkdir(exist_ok=True)
    cache_file = _CACHE_DIR / f"{split}_bpe65536_{seq_len}.pt"

    if cache_file.exists():
        return torch.load(cache_file, weights_only=True)

    texts = _load_wikitext_split_texts(split)
    full_text = "\n".join(texts)
    tokens = tokenizer.encode(full_text).ids

    chunks = chunk_tokens(tokens, seq_len)
    torch.save(chunks, cache_file)
    print(f"Cached {len(chunks):,} sequences → {cache_file}")
    return chunks


def get_dataloaders(cfg) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Return (train_loader, val_loader, test_loader) for WikiText-103.

    cfg must have: seq_len, batch_size, seed
    """
    tokenizer = _build_tokenizer(_CACHE_DIR)

    g = torch.Generator()
    g.manual_seed(cfg.seed)

    def _make_loader(
        split: str,
        shuffle: bool,
        drop_last: bool,
        persistent_workers: bool = False,
        prefetch_factor: int | None = None,
    ) -> DataLoader:
        data = _load_split(split, cfg.seq_len, tokenizer)
        ds = TokenDataset(data)
        return DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=4,
            pin_memory=True,
            generator=g if shuffle else None,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

    train_loader = _make_loader("train", shuffle=True, drop_last=True,
                                persistent_workers=True, prefetch_factor=2)
    val_loader   = _make_loader("validation", shuffle=False, drop_last=False)
    test_loader  = _make_loader("test",       shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader
