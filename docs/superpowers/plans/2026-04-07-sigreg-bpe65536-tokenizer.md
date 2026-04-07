# SIGReg BPE-65536 Tokenizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `sigreg/data.py` that lazily trains a BPE tokenizer with vocab size 65536 on WikiText-103 and wires it into the sigreg training pipeline.

**Architecture:** A new `sigreg/data.py` mirrors `rbf_ffn/data.py` but uses `ByteLevelBPETokenizer` (HuggingFace `tokenizers`) instead of `tiktoken r50k_base`. Tokenizer and split tensors are cached in `rbf_ffn/data_cache/`. `sigreg/train.py` changes one import line.

**Tech Stack:** Python, PyTorch, `tokenizers` (HuggingFace), `datasets` (HuggingFace), pytest

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `sigreg/data.py` | **Create** | Lazy BPE tokenizer build + WikiText-103 dataloaders |
| `sigreg/tests/__init__.py` | **Create** | Marks sigreg/tests as a package |
| `sigreg/tests/test_data.py` | **Create** | Unit tests for sigreg/data.py (no network/disk) |
| `sigreg/train.py` | **Modify** | Swap import from `rbf_ffn.data` → `sigreg.data` |
| `sigreg/README.md` | **Modify** | Update tokenizer reference |
| `pyproject.toml` | **Modify** | Add `sigreg/tests` to pytest testpaths |

---

### Task 1: Set up test infrastructure

**Files:**
- Create: `sigreg/tests/__init__.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Create the tests package**

```bash
mkdir -p sigreg/tests
touch sigreg/tests/__init__.py
```

- [ ] **Step 2: Add sigreg/tests to pyproject.toml testpaths**

In `pyproject.toml`, update `[tool.pytest.ini_options]`:

```toml
[tool.pytest.ini_options]
testpaths = ["rbf_ffn/tests", "kromhc_transformer/tests", "grokking/tests", "sigreg/tests"]
```

- [ ] **Step 3: Verify pytest discovers the new directory**

```bash
pytest sigreg/tests/ --collect-only
```

Expected: `no tests ran` (empty directory), no errors.

- [ ] **Step 4: Commit**

```bash
git add sigreg/tests/__init__.py pyproject.toml
git commit -m "chore(sigreg): add tests directory and register with pytest"
```

---

### Task 2: Write failing tests for sigreg/data.py

**Files:**
- Create: `sigreg/tests/test_data.py`

- [ ] **Step 1: Write the test file**

Create `sigreg/tests/test_data.py`:

```python
"""
Tests for sigreg/data.py.

All tests are offline — no network calls, no disk I/O to rbf_ffn/data_cache/.
_build_tokenizer and _load_split are mocked where needed.
"""
import torch
import pytest
from unittest.mock import MagicMock, patch


# ── _build_tokenizer ──────────────────────────────────────────────────────────

def test_build_tokenizer_loads_from_cache_when_files_exist(tmp_path):
    """If vocab.json and merges.txt exist, load without retraining."""
    tok_dir = tmp_path / "bpe65536_tokenizer"
    tok_dir.mkdir()
    (tok_dir / "vocab.json").write_text('{"a": 0}')
    (tok_dir / "merges.txt").write_text("#version: 0.2\n")

    mock_tok = MagicMock()
    with patch("sigreg.data.ByteLevelBPETokenizer", return_value=mock_tok) as MockTok:
        from sigreg.data import _build_tokenizer
        result = _build_tokenizer(tmp_path)

    MockTok.assert_called_once_with(
        str(tok_dir / "vocab.json"),
        str(tok_dir / "merges.txt"),
    )
    assert result is mock_tok


def test_build_tokenizer_trains_and_saves_when_no_cache(tmp_path):
    """If no cache files, train on WikiText-103 and save."""
    mock_tok = MagicMock()
    fake_dataset = [{"text": "hello world"}, {"text": "  "}, {"text": "foo bar"}]

    with patch("sigreg.data.ByteLevelBPETokenizer", return_value=mock_tok) as MockTok, \
         patch("sigreg.data._load_wikitext_train_texts", return_value=["hello world", "foo bar"]):
        from sigreg.data import _build_tokenizer
        result = _build_tokenizer(tmp_path)

    # Constructor called with no args (fresh tokenizer)
    MockTok.assert_called_once_with()
    mock_tok.train_from_iterator.assert_called_once_with(
        ["hello world", "foo bar"],
        vocab_size=65536,
        min_frequency=2,
        special_tokens=[],
    )
    tok_dir = tmp_path / "bpe65536_tokenizer"
    mock_tok.save_model.assert_called_once_with(str(tok_dir))
    assert result is mock_tok


# ── _load_split ───────────────────────────────────────────────────────────────

def test_load_split_uses_bpe65536_cache_filename(tmp_path):
    """Cache file must be named {split}_bpe65536_{seq_len}.pt."""
    mock_tok = MagicMock()
    mock_tok.encode.return_value.ids = list(range(100))

    with patch("sigreg.data._CACHE_DIR", tmp_path), \
         patch("sigreg.data._load_wikitext_split_texts", return_value=["hello world"]):
        from sigreg.data import _load_split
        _load_split("train", seq_len=10, tokenizer=mock_tok)

    assert (tmp_path / "train_bpe65536_10.pt").exists()


def test_load_split_loads_from_cache_when_pt_exists(tmp_path):
    """If the .pt file exists, return it without re-tokenising."""
    cached = torch.zeros(5, 10, dtype=torch.long)
    cache_file = tmp_path / "validation_bpe65536_10.pt"
    torch.save(cached, cache_file)

    mock_tok = MagicMock()
    with patch("sigreg.data._CACHE_DIR", tmp_path):
        from sigreg.data import _load_split
        result = _load_split("validation", seq_len=10, tokenizer=mock_tok)

    mock_tok.encode.assert_not_called()
    assert result.shape == (5, 10)


def test_load_split_returns_long_tensor(tmp_path):
    """Returned tensor must be dtype=torch.long."""
    mock_tok = MagicMock()
    mock_tok.encode.return_value.ids = list(range(50))

    with patch("sigreg.data._CACHE_DIR", tmp_path), \
         patch("sigreg.data._load_wikitext_split_texts", return_value=["text"]):
        from sigreg.data import _load_split
        result = _load_split("test", seq_len=10, tokenizer=mock_tok)

    assert result.dtype == torch.long


# ── get_dataloaders ───────────────────────────────────────────────────────────

def _fake_load_split(split, seq_len, tokenizer):
    return torch.zeros(32, seq_len, dtype=torch.long)


class _Cfg:
    seq_len = 16
    batch_size = 4
    seed = 42


def test_get_dataloaders_returns_three_loaders():
    mock_tok = MagicMock()
    with patch("sigreg.data._build_tokenizer", return_value=mock_tok), \
         patch("sigreg.data._load_split", side_effect=_fake_load_split):
        from sigreg.data import get_dataloaders
        result = get_dataloaders(_Cfg())
    assert len(result) == 3


def test_train_loader_has_persistent_workers_and_prefetch():
    mock_tok = MagicMock()
    with patch("sigreg.data._build_tokenizer", return_value=mock_tok), \
         patch("sigreg.data._load_split", side_effect=_fake_load_split):
        from sigreg.data import get_dataloaders
        train_loader, _, _ = get_dataloaders(_Cfg())
    assert train_loader.persistent_workers is True
    assert train_loader.prefetch_factor == 2


def test_val_and_test_loaders_have_no_persistent_workers():
    mock_tok = MagicMock()
    with patch("sigreg.data._build_tokenizer", return_value=mock_tok), \
         patch("sigreg.data._load_split", side_effect=_fake_load_split):
        from sigreg.data import get_dataloaders
        _, val_loader, test_loader = get_dataloaders(_Cfg())
    assert val_loader.persistent_workers is False
    assert test_loader.persistent_workers is False
```

- [ ] **Step 2: Run tests — expect ImportError (module not yet written)**

```bash
pytest sigreg/tests/test_data.py -v
```

Expected: `ImportError: No module named 'sigreg.data'` or similar — confirms tests are wired up and failing for the right reason.

- [ ] **Step 3: Commit the failing tests**

```bash
git add sigreg/tests/test_data.py
git commit -m "test(sigreg): add failing tests for BPE-65536 data pipeline"
```

---

### Task 3: Implement sigreg/data.py

**Files:**
- Create: `sigreg/data.py`

- [ ] **Step 1: Write sigreg/data.py**

Create `sigreg/data.py`:

```python
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


def _load_wikitext_train_texts() -> list[str]:
    """Download WikiText-103 train split and return non-empty text rows."""
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    return [row["text"] for row in dataset if row["text"].strip()]


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
    texts = _load_wikitext_train_texts()
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
```

- [ ] **Step 2: Run the tests**

```bash
pytest sigreg/tests/test_data.py -v
```

Expected: all 9 tests pass.

- [ ] **Step 3: Commit**

```bash
git add sigreg/data.py
git commit -m "feat(sigreg): add BPE-65536 data pipeline with lazy tokenizer training"
```

---

### Task 4: Wire sigreg/train.py to sigreg/data.py

**Files:**
- Modify: `sigreg/train.py:36`

- [ ] **Step 1: Swap the import**

In `sigreg/train.py`, change line 36 from:

```python
from rbf_ffn.data import get_dataloaders
```

to:

```python
from sigreg.data import get_dataloaders
```

- [ ] **Step 2: Run the full test suite to confirm nothing broke**

```bash
pytest rbf_ffn/tests/ sigreg/tests/ -v
```

Expected: all tests pass. `rbf_ffn` tests are unaffected.

- [ ] **Step 3: Commit**

```bash
git add sigreg/train.py
git commit -m "feat(sigreg): use sigreg.data (BPE-65536) instead of rbf_ffn.data (r50k)"
```

---

### Task 5: Update sigreg/README.md

**Files:**
- Modify: `sigreg/README.md`

- [ ] **Step 1: Update the tokenizer reference**

In `sigreg/README.md`, replace the Setup section's tokenizer note:

Find (line ~40):
```
The WikiText-103 dataset is reused from `rbf_ffn`. On first run it will be downloaded, tokenised with `r50k_base`, and cached to `rbf_ffn/data_cache/`. Subsequent runs load from cache instantly.
```

Replace with:
```
The WikiText-103 dataset is shared with `rbf_ffn`. On first run the BPE-65536 tokenizer is trained on the WikiText-103 training split (~5 minutes), then each split is tokenised and cached to `rbf_ffn/data_cache/`. Subsequent runs load from cache instantly.
```

- [ ] **Step 2: Update the Config Reference table's `vocab_size` row note**

Find the `vocab_size` row in the Config Reference table:
```
| `vocab_size` | 65536 | Vocabulary size (must match tokeniser) |
```

Update to:
```
| `vocab_size` | 65536 | Vocabulary size — matches the custom BPE-65536 tokeniser trained in `sigreg/data.py` |
```

- [ ] **Step 3: Commit**

```bash
git add sigreg/README.md
git commit -m "docs(sigreg): update tokenizer references from r50k_base to BPE-65536"
```
