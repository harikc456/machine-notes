# SIGReg BPE-65536 Tokenizer Design

**Date:** 2026-04-07  
**Status:** Approved

## Overview

Replace the `r50k_base` tokenizer (tiktoken, ~50k vocab) used by the `sigreg` experiment with a custom BPE tokenizer trained on WikiText-103 with `vocab_size=65536`, matching `SIGRegConfig.vocab_size`.

## Architecture

A new `sigreg/data.py` replaces the import of `rbf_ffn.data` in `sigreg/train.py`. It is structurally identical to `rbf_ffn/data.py` but swaps the tokenizer. Nothing in `rbf_ffn/` changes.

```
sigreg/train.py
    └── sigreg/data.py          (new)
            ├── _build_tokenizer()   lazy: trains + caches BPE-65536
            └── _load_split()        lazy: tokenises + caches split tensors
```

## Tokenizer Training

- **Library:** `tokenizers.ByteLevelBPETokenizer` (HuggingFace, already installed)
- **Training corpus:** WikiText-103 training split (downloaded via `datasets`, same as existing pipeline)
- **Vocab size:** 65536
- **Save location:** `rbf_ffn/data_cache/bpe65536_tokenizer/` (`vocab.json` + `merges.txt`)
- **Trigger:** First call to `get_dataloaders`; subsequent runs load from cache

## Data Caching

Split tensors are cached as `{split}_bpe65536_{seq_len}.pt` in `rbf_ffn/data_cache/`, distinct from the existing `{split}_r50k_{seq_len}.pt` files. The two pipelines never collide.

## Changes

| File | Change |
|---|---|
| `sigreg/data.py` | **New** — `_build_tokenizer`, `_load_split`, `get_dataloaders` |
| `sigreg/train.py` | 1-line import swap: `sigreg.data` instead of `rbf_ffn.data` |
| `sigreg/README.md` | Update tokenizer reference from `r50k_base` to custom BPE-65536 |

No changes to `rbf_ffn/`.

## Interface Contract

`sigreg.data.get_dataloaders(cfg)` has the same signature and return type as `rbf_ffn.data.get_dataloaders(cfg)`: takes any object with `seq_len`, `batch_size`, `seed` attributes; returns `(train_loader, val_loader, test_loader)`.
