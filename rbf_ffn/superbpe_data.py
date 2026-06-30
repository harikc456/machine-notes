"""
WikiText-103 data pipeline using a two-stage SuperBPE tokenizer.

SuperBPE (arXiv:2503.13423) trains BPE in two stages:
  Stage 1 (0 → t): standard BPE with whitespace pretokenization (subwords)
  Stage 2 (t → T): BPE without whitespace constraint (superwords / multi-word tokens)

Transition point t=43718, total vocab T=48576 (~90/10 split, per paper recommendation).

Usage:
    from rbf_ffn.superbpe_data import get_dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(cfg)

cfg must have: seq_len, batch_size, seed
"""
from __future__ import annotations
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel, Whitespace, Sequence as PreTokenizerSequence
from torch.utils.data import DataLoader

from rbf_ffn.data import chunk_tokens, TokenDataset

_CACHE_DIR = Path(__file__).parent / "data_cache"

_VOCAB_SIZE = 48576
_TRANSITION = 43718   # Stage 1 boundary (~90% of total vocab)


def _load_wikitext_split_texts(split: str) -> list[str]:
    """Download a WikiText-103 split and return non-empty text rows."""
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    return [row["text"] for row in dataset if row["text"].strip()]


def _train_stage1(texts: list[str], transition: int) -> Tokenizer:
    """Train Stage 1 BPE: whitespace-bounded, vocab size = transition."""
    tok = Tokenizer(BPE())
    tok.pre_tokenizer = PreTokenizerSequence([Whitespace(), ByteLevel(add_prefix_space=False)])
    trainer = BpeTrainer(vocab_size=transition, min_frequency=2, special_tokens=[])
    tok.train_from_iterator(texts, trainer)
    return tok


def _train_stage2(stage1: Tokenizer, texts: list[str], vocab_size: int) -> Tokenizer:
    """Train Stage 2 BPE: no whitespace boundary, warm-starting from Stage 1.

    Loads Stage 1 vocab+merges via serialisation, swaps the pre-tokenizer to
    ByteLevel-only, then continues BPE until vocab_size is reached.
    """
    try:
        tok2 = Tokenizer.from_str(stage1.to_str())
    except Exception as e:
        import warnings
        warnings.warn(
            f"Stage 2 warm-start from Stage 1 failed ({e}); "
            "falling back to fresh BPE with ByteLevel-only pre-tokenizer.",
            RuntimeWarning,
        )
        tok2 = Tokenizer(BPE())
    tok2.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=[])
    tok2.train_from_iterator(texts, trainer)
    return tok2


def _build_superbpe_tokenizer(cache_dir: Path) -> Tokenizer:
    """Return the Stage 2 SuperBPE tokenizer (vocab_size=48576).

    Trains Stage 1 then Stage 2 on WikiText-103 train split on first call;
    subsequent calls load from cache instantly.
    """
    tok_dir = cache_dir / "superbpe48576_tokenizer"
    stage1_file = tok_dir / "stage1.json"
    stage2_file = tok_dir / "stage2.json"

    if stage2_file.exists():
        return Tokenizer.from_file(str(stage2_file))

    tok_dir.mkdir(parents=True, exist_ok=True)
    texts = _load_wikitext_split_texts("train")

    if not stage1_file.exists():
        print(f"Training SuperBPE Stage 1 (t={_TRANSITION})...")
        tok1 = _train_stage1(texts, _TRANSITION)
        tok1.save(str(stage1_file))
        print(f"Stage 1 saved -> {stage1_file}")
    else:
        tok1 = Tokenizer.from_file(str(stage1_file))

    print(f"Training SuperBPE Stage 2 (T={_VOCAB_SIZE})...")
    tok2 = _train_stage2(tok1, texts, _VOCAB_SIZE)
    tok2.save(str(stage2_file))
    print(f"Stage 2 saved -> {stage2_file}")
    return tok2


def _load_split(split: str, seq_len: int, tokenizer: Tokenizer) -> torch.Tensor:
    """Tokenise a WikiText-103 split and return (N, seq_len) LongTensor.

    Caches to _CACHE_DIR/{split}_superbpe48576_{seq_len}.pt on first call.
    """
    _CACHE_DIR.mkdir(exist_ok=True)
    cache_file = _CACHE_DIR / f"{split}_superbpe48576_{seq_len}.pt"

    if cache_file.exists():
        return torch.load(cache_file, weights_only=True)

    texts = _load_wikitext_split_texts(split)
    all_tokens: list[int] = []
    batch_size = 1000
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encodings = tokenizer.encode_batch(batch)
        for enc in encodings:
            all_tokens.extend(enc.ids)
        if (i // batch_size + 1) % 20 == 0:
            print(f"  Encoded {i + len(batch):,}/{len(texts):,} texts -> {len(all_tokens):,} tokens")

    chunks = chunk_tokens(all_tokens, seq_len)
    torch.save(chunks, cache_file)
    print(f"Cached {len(chunks):,} sequences -> {cache_file}")
    return chunks


def get_dataloaders(cfg) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader) for WikiText-103.

    cfg must have: seq_len, batch_size, seed
    """
    tokenizer = _build_superbpe_tokenizer(_CACHE_DIR)

    g = torch.Generator()
    g.manual_seed(cfg.seed)

    def _make_loader(
        split: str,
        shuffle: bool,
        drop_last: bool,
        num_workers: int = 0,
    ) -> DataLoader:
        data = _load_split(split, cfg.seq_len, tokenizer)
        ds = TokenDataset(data)
        return DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=True,
            generator=g if shuffle else None,
        )

    train_loader = _make_loader("train",      shuffle=True,  drop_last=True)
    val_loader   = _make_loader("validation", shuffle=False, drop_last=False)
    test_loader  = _make_loader("test",       shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader
