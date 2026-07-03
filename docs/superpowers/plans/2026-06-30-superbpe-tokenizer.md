# SuperBPE Tokenizer Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a two-stage SuperBPE tokenizer (vocab_size=48576, transition=43718) to `rbf_ffn` and pretrain a baseline XSA+SwiGLU model with it on WikiText-103.

**Architecture:** A new `rbf_ffn/superbpe_data.py` module handles tokenizer training (Stage 1: BPE with whitespace boundary → Stage 2: BPE without whitespace boundary) and the WikiText-103 data pipeline. `ModelConfig` gains a `tokenizer` field; `train.py` dispatches to the right data loader based on it.

**Tech Stack:** `tokenizers` (HuggingFace), `datasets` (HuggingFace), PyTorch, pytest

## Global Constraints

- Python 3.10+
- All new code in `rbf_ffn/` package
- No new top-level dependencies: `tokenizers` is already available (used in `sigreg/data.py`)
- Cache files go in `rbf_ffn/data_cache/` — shared with existing r50k and bpe65536 caches
- Tokenizer cache: `data_cache/superbpe48576_tokenizer/stage1.json` and `stage2.json`
- Token chunk cache: `data_cache/{split}_superbpe48576_{seq_len}.pt`
- All tests must pass without network or GPU (mock expensive calls)
- Run tests with: `pytest rbf_ffn/tests/ -v`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `rbf_ffn/config.py` | Modify | Add `tokenizer` field + validation |
| `rbf_ffn/superbpe_data.py` | Create | Two-stage tokenizer training + WikiText-103 pipeline |
| `rbf_ffn/train.py` | Modify | Dispatch `get_dataloaders` by tokenizer; `_superbpe` naming tag |
| `rbf_ffn/configs/baseline_xsa_superbpe.yaml` | Create | Experiment config |
| `rbf_ffn/tests/test_config.py` | Modify | Tests for tokenizer field + validation |
| `rbf_ffn/tests/test_superbpe_data.py` | Create | Tests for tokenizer training + data pipeline |
| `rbf_ffn/tests/test_train.py` | Modify | Tests for dispatch + naming tag |

---

## Task 1: `ModelConfig` tokenizer field + validation

**Files:**
- Modify: `rbf_ffn/config.py`
- Modify: `rbf_ffn/tests/test_config.py`

**Interfaces:**
- Produces: `ModelConfig.tokenizer: str = "r50k"` — accepted values: `"r50k"` | `"superbpe48576"`

- [ ] **Step 1: Write the failing tests**

Append to `rbf_ffn/tests/test_config.py`:

```python
# ── tokenizer field ───────────────────────────────────────────────────────────

def test_tokenizer_default_is_r50k():
    cfg = ModelConfig()
    assert cfg.tokenizer == "r50k"


def test_tokenizer_superbpe_with_correct_vocab_size_is_valid():
    cfg = ModelConfig(tokenizer="superbpe48576", vocab_size=48576)
    assert cfg.tokenizer == "superbpe48576"
    assert cfg.vocab_size == 48576


def test_tokenizer_superbpe_with_wrong_vocab_size_raises():
    with pytest.raises(ValueError, match="vocab_size must be 48576"):
        ModelConfig(tokenizer="superbpe48576", vocab_size=50257)


def test_tokenizer_unknown_value_raises():
    with pytest.raises(ValueError, match="Unknown tokenizer"):
        ModelConfig(tokenizer="gpt4_turbo")


def test_tokenizer_yaml_roundtrip(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("tokenizer: superbpe48576\nvocab_size: 48576\n")
    cfg = load_config(p)
    assert cfg.tokenizer == "superbpe48576"
    assert cfg.vocab_size == 48576
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest rbf_ffn/tests/test_config.py::test_tokenizer_default_is_r50k rbf_ffn/tests/test_config.py::test_tokenizer_superbpe_with_correct_vocab_size_is_valid rbf_ffn/tests/test_config.py::test_tokenizer_superbpe_with_wrong_vocab_size_raises rbf_ffn/tests/test_config.py::test_tokenizer_unknown_value_raises rbf_ffn/tests/test_config.py::test_tokenizer_yaml_roundtrip -v
```

Expected: 5 failures — `ModelConfig` has no `tokenizer` field yet.

- [ ] **Step 3: Add `tokenizer` field and validation to `ModelConfig`**

In `rbf_ffn/config.py`, add after the `mup_init_std` field (before `__post_init__`):

```python
    # Tokenizer
    tokenizer: str = "r50k"    # "r50k" | "superbpe48576"
```

In `__post_init__`, add before the closing (after the mup validation block):

```python
        _valid_tokenizers = {"r50k", "superbpe48576"}
        if self.tokenizer not in _valid_tokenizers:
            raise ValueError(
                f"Unknown tokenizer '{self.tokenizer}'. Valid values: {sorted(_valid_tokenizers)}"
            )
        if self.tokenizer == "superbpe48576" and self.vocab_size != 48576:
            raise ValueError(
                f"vocab_size must be 48576 when tokenizer='superbpe48576', got {self.vocab_size}"
            )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest rbf_ffn/tests/test_config.py::test_tokenizer_default_is_r50k rbf_ffn/tests/test_config.py::test_tokenizer_superbpe_with_correct_vocab_size_is_valid rbf_ffn/tests/test_config.py::test_tokenizer_superbpe_with_wrong_vocab_size_raises rbf_ffn/tests/test_config.py::test_tokenizer_unknown_value_raises rbf_ffn/tests/test_config.py::test_tokenizer_yaml_roundtrip -v
```

Expected: 5 PASSED.

- [ ] **Step 5: Verify no existing tests broke**

```bash
pytest rbf_ffn/tests/test_config.py -v
```

Expected: all previously passing tests still PASS.

- [ ] **Step 6: Commit**

```bash
git add rbf_ffn/config.py rbf_ffn/tests/test_config.py
git commit -m "feat(rbf_ffn): add tokenizer field to ModelConfig with superbpe48576 validation"
```

---

## Task 2: `superbpe_data.py` — two-stage tokenizer + data pipeline

**Files:**
- Create: `rbf_ffn/superbpe_data.py`
- Create: `rbf_ffn/tests/test_superbpe_data.py`

**Interfaces:**
- Consumes: `rbf_ffn.data.chunk_tokens(tokens: list[int], seq_len: int) -> torch.Tensor`
- Consumes: `rbf_ffn.data.TokenDataset`
- Produces: `_train_stage1(texts: list[str], transition: int) -> Tokenizer`
- Produces: `_train_stage2(stage1: Tokenizer, texts: list[str], vocab_size: int) -> Tokenizer`
- Produces: `_build_superbpe_tokenizer(cache_dir: Path) -> Tokenizer`
- Produces: `_load_split(split: str, seq_len: int, tokenizer: Tokenizer) -> torch.Tensor`
- Produces: `get_dataloaders(cfg) -> tuple[DataLoader, DataLoader, DataLoader]`

Note on Stage 2 warm-start: `Tokenizer.to_str()` serializes to a JSON string and `Tokenizer.from_str()` deserializes — this lets Stage 2 load Stage 1's vocab+merges in memory. `BpeTrainer` will continue from the existing model, adding only the remaining merges needed to reach `vocab_size`.

- [ ] **Step 1: Write the failing tests**

Create `rbf_ffn/tests/test_superbpe_data.py`:

```python
"""
Tests for superbpe_data.py.

Tokenizer training tests use a tiny corpus and tiny vocab sizes so they
complete in < 1 second. Data pipeline tests mock the tokenizer and I/O.
"""
from __future__ import annotations
import torch
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# ── Tiny-corpus helpers ───────────────────────────────────────────────────────

_TINY_TEXTS = [
    "the quick brown fox jumps over the lazy dog",
    "pack my box with five dozen liquor jugs",
    "how vexingly quick daft zebras jump",
    "sphinx of black quartz judge my vow",
] * 100   # repeat for sufficient frequency counts

_TRANSITION = 50    # small enough for < 1s training
_VOCAB_SIZE = 70    # 20 superword slots above the subword stage


# ── _train_stage1 ─────────────────────────────────────────────────────────────

def test_train_stage1_reaches_target_vocab_size():
    from rbf_ffn.superbpe_data import _train_stage1
    tok = _train_stage1(_TINY_TEXTS, transition=_TRANSITION)
    assert tok.get_vocab_size() == _TRANSITION


def test_train_stage1_encodes_text():
    from rbf_ffn.superbpe_data import _train_stage1
    tok = _train_stage1(_TINY_TEXTS, transition=_TRANSITION)
    enc = tok.encode("the quick brown fox")
    assert len(enc.ids) > 0
    assert all(isinstance(i, int) for i in enc.ids)


def test_train_stage1_ids_within_vocab():
    from rbf_ffn.superbpe_data import _train_stage1
    tok = _train_stage1(_TINY_TEXTS, transition=_TRANSITION)
    enc = tok.encode("hello world")
    assert all(0 <= i < _TRANSITION for i in enc.ids)


# ── _train_stage2 ─────────────────────────────────────────────────────────────

def test_train_stage2_reaches_target_vocab_size():
    from rbf_ffn.superbpe_data import _train_stage1, _train_stage2
    tok1 = _train_stage1(_TINY_TEXTS, transition=_TRANSITION)
    tok2 = _train_stage2(tok1, _TINY_TEXTS, vocab_size=_VOCAB_SIZE)
    assert tok2.get_vocab_size() == _VOCAB_SIZE


def test_train_stage2_encodes_text():
    from rbf_ffn.superbpe_data import _train_stage1, _train_stage2
    tok1 = _train_stage1(_TINY_TEXTS, transition=_TRANSITION)
    tok2 = _train_stage2(tok1, _TINY_TEXTS, vocab_size=_VOCAB_SIZE)
    enc = tok2.encode("the quick brown fox")
    assert len(enc.ids) > 0
    assert all(0 <= i < _VOCAB_SIZE for i in enc.ids)


def test_train_stage2_ids_within_extended_vocab():
    from rbf_ffn.superbpe_data import _train_stage1, _train_stage2
    tok1 = _train_stage1(_TINY_TEXTS, transition=_TRANSITION)
    tok2 = _train_stage2(tok1, _TINY_TEXTS, vocab_size=_VOCAB_SIZE)
    enc = tok2.encode("pack my box with five dozen")
    assert all(0 <= i < _VOCAB_SIZE for i in enc.ids)


# ── _build_superbpe_tokenizer ─────────────────────────────────────────────────

def test_build_superbpe_tokenizer_creates_cache_files(tmp_path):
    from rbf_ffn.superbpe_data import _build_superbpe_tokenizer, _train_stage1, _train_stage2
    # Patch WikiText loading and the training helpers to use tiny corpus + tiny sizes
    with patch("rbf_ffn.superbpe_data._load_wikitext_split_texts", return_value=_TINY_TEXTS), \
         patch("rbf_ffn.superbpe_data._VOCAB_SIZE", _VOCAB_SIZE), \
         patch("rbf_ffn.superbpe_data._TRANSITION", _TRANSITION):
        _build_superbpe_tokenizer(tmp_path)
    stage1 = tmp_path / "superbpe48576_tokenizer" / "stage1.json"
    stage2 = tmp_path / "superbpe48576_tokenizer" / "stage2.json"
    assert stage1.exists()
    assert stage2.exists()


def test_build_superbpe_tokenizer_loads_from_cache(tmp_path):
    from rbf_ffn.superbpe_data import _build_superbpe_tokenizer
    with patch("rbf_ffn.superbpe_data._load_wikitext_split_texts", return_value=_TINY_TEXTS), \
         patch("rbf_ffn.superbpe_data._VOCAB_SIZE", _VOCAB_SIZE), \
         patch("rbf_ffn.superbpe_data._TRANSITION", _TRANSITION):
        tok_first = _build_superbpe_tokenizer(tmp_path)
        # Second call must load from cache without re-training
        with patch("rbf_ffn.superbpe_data._train_stage1") as mock_s1, \
             patch("rbf_ffn.superbpe_data._train_stage2") as mock_s2:
            tok_second = _build_superbpe_tokenizer(tmp_path)
            mock_s1.assert_not_called()
            mock_s2.assert_not_called()
    assert tok_second.get_vocab_size() == tok_first.get_vocab_size()


# ── _load_split ───────────────────────────────────────────────────────────────

def test_load_split_returns_correct_shape(tmp_path):
    from rbf_ffn.superbpe_data import _load_split, _train_stage1
    tok = _train_stage1(_TINY_TEXTS, transition=_TRANSITION)
    texts = _TINY_TEXTS[:10]
    with patch("rbf_ffn.superbpe_data._load_wikitext_split_texts", return_value=texts), \
         patch("rbf_ffn.superbpe_data._CACHE_DIR", tmp_path):
        chunks = _load_split("train", seq_len=16, tokenizer=tok)
    assert chunks.ndim == 2
    assert chunks.shape[1] == 16
    assert chunks.dtype == torch.long


def test_load_split_caches_to_disk(tmp_path):
    from rbf_ffn.superbpe_data import _load_split, _train_stage1
    tok = _train_stage1(_TINY_TEXTS, transition=_TRANSITION)
    texts = _TINY_TEXTS[:10]
    with patch("rbf_ffn.superbpe_data._load_wikitext_split_texts", return_value=texts), \
         patch("rbf_ffn.superbpe_data._CACHE_DIR", tmp_path):
        _load_split("train", seq_len=16, tokenizer=tok)
        cache_file = tmp_path / "train_superbpe48576_16.pt"
        assert cache_file.exists()


# ── get_dataloaders ───────────────────────────────────────────────────────────

def _fake_tokenizer():
    """Minimal mock tokenizer for DataLoader tests."""
    tok = MagicMock()
    tok.encode_batch.return_value = [MagicMock(ids=[0, 1, 2, 3]) for _ in range(10)]
    return tok


def test_get_dataloaders_returns_three_loaders(tmp_path):
    from rbf_ffn.superbpe_data import get_dataloaders
    from rbf_ffn.config import ModelConfig
    cfg = ModelConfig(tokenizer="superbpe48576", vocab_size=48576, seq_len=8, batch_size=2)
    fake_chunks = torch.zeros(32, 8, dtype=torch.long)
    with patch("rbf_ffn.superbpe_data._build_superbpe_tokenizer", return_value=_fake_tokenizer()), \
         patch("rbf_ffn.superbpe_data._load_split", return_value=fake_chunks):
        train_loader, val_loader, test_loader = get_dataloaders(cfg)
    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None


def test_get_dataloaders_batch_size(tmp_path):
    from rbf_ffn.superbpe_data import get_dataloaders
    from rbf_ffn.config import ModelConfig
    cfg = ModelConfig(tokenizer="superbpe48576", vocab_size=48576, seq_len=8, batch_size=4)
    fake_chunks = torch.zeros(32, 8, dtype=torch.long)
    with patch("rbf_ffn.superbpe_data._build_superbpe_tokenizer", return_value=_fake_tokenizer()), \
         patch("rbf_ffn.superbpe_data._load_split", return_value=fake_chunks):
        train_loader, _, _ = get_dataloaders(cfg)
    assert train_loader.batch_size == 4
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest rbf_ffn/tests/test_superbpe_data.py -v
```

Expected: all fail with `ModuleNotFoundError: No module named 'rbf_ffn.superbpe_data'`.

- [ ] **Step 3: Create `rbf_ffn/superbpe_data.py`**

```python
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
    tok2 = Tokenizer.from_str(stage1.to_str())
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
        print(f"Training SuperBPE Stage 1 (t={_TRANSITION})…")
        tok1 = _train_stage1(texts, _TRANSITION)
        tok1.save(str(stage1_file))
        print(f"Stage 1 saved → {stage1_file}")
    else:
        tok1 = Tokenizer.from_file(str(stage1_file))

    print(f"Training SuperBPE Stage 2 (T={_VOCAB_SIZE})…")
    tok2 = _train_stage2(tok1, texts, _VOCAB_SIZE)
    tok2.save(str(stage2_file))
    print(f"Stage 2 saved → {stage2_file}")
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
            print(f"  Encoded {i + len(batch):,}/{len(texts):,} texts → {len(all_tokens):,} tokens")

    chunks = chunk_tokens(all_tokens, seq_len)
    torch.save(chunks, cache_file)
    print(f"Cached {len(chunks):,} sequences → {cache_file}")
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest rbf_ffn/tests/test_superbpe_data.py -v
```

Expected: all PASSED. The tokenizer training tests may each take 1–3 seconds on a tiny corpus — that is acceptable.

- [ ] **Step 5: Commit**

```bash
git add rbf_ffn/superbpe_data.py rbf_ffn/tests/test_superbpe_data.py
git commit -m "feat(rbf_ffn): add SuperBPE two-stage tokenizer and WikiText-103 data pipeline"
```

---

## Task 3: `train.py` dispatch + experiment naming tag

**Files:**
- Modify: `rbf_ffn/train.py`
- Modify: `rbf_ffn/tests/test_train.py`

**Interfaces:**
- Consumes: `ModelConfig.tokenizer` (from Task 1)
- Consumes: `rbf_ffn.superbpe_data.get_dataloaders` (from Task 2)

- [ ] **Step 1: Write the failing tests**

Append to `rbf_ffn/tests/test_train.py` (add `get_experiment_dir` to the existing import from `rbf_ffn.train`):

```python
from rbf_ffn.train import make_lr_lambda, train, apply_adaptive_weight_norm, get_experiment_dir


def test_get_experiment_dir_includes_superbpe_tag(tmp_path):
    cfg = ModelConfig(
        tokenizer="superbpe48576",
        vocab_size=48576,
        attn_type="xsa",
        ffn_type="swiglu",
        d_model=32,
    )
    with patch("rbf_ffn.train.Path") as mock_path:
        # We just want to check the name string, not create a real dir
        mock_path.return_value.__truediv__.return_value.__truediv__.return_value.mkdir = MagicMock()
        # Instead: call the function with a monkeypatch on Path.__file__ to use tmp_path
        pass
    # Simpler: check the name directly by inspecting get_experiment_dir source logic
    # We'll test via a real call and check the returned path name
    import rbf_ffn.train as train_mod
    original = train_mod.Path
    try:
        # Redirect experiments/ to tmp_path
        class _PatchedPath(original):
            def __new__(cls, *args, **kwargs):
                p = original(*args, **kwargs)
                return p
        with patch.object(train_mod, "__file__", str(tmp_path / "train.py")):
            result = get_experiment_dir(cfg)
    finally:
        pass
    assert "_superbpe" in result.name


def test_train_uses_superbpe_dataloaders(tmp_path):
    """When cfg.tokenizer='superbpe48576', train() calls superbpe_data.get_dataloaders."""
    cfg = _tiny_cfg(tokenizer="superbpe48576", vocab_size=48576)
    fake_train, fake_val, _ = _fake_loaders(cfg)
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(
        "tokenizer: superbpe48576\nvocab_size: 48576\nseq_len: 9\nbatch_size: 2\nn_epochs: 1\n"
    )
    with patch("rbf_ffn.superbpe_data.get_dataloaders", return_value=(fake_train, fake_val, fake_val)) as mock_sbpe, \
         patch("rbf_ffn.data.get_dataloaders") as mock_r50k:
        train(cfg, config_path)
        mock_sbpe.assert_called_once()
        mock_r50k.assert_not_called()


def test_train_uses_r50k_dataloaders_by_default(tmp_path):
    """Default tokenizer uses rbf_ffn.data.get_dataloaders (r50k)."""
    cfg = _tiny_cfg()  # tokenizer defaults to "r50k"
    fake_train, fake_val, _ = _fake_loaders(cfg)
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("seq_len: 9\nbatch_size: 2\nn_epochs: 1\n")
    with patch("rbf_ffn.data.get_dataloaders", return_value=(fake_train, fake_val, fake_val)) as mock_r50k, \
         patch("rbf_ffn.superbpe_data.get_dataloaders") as mock_sbpe:
        train(cfg, config_path)
        mock_r50k.assert_called_once()
        mock_sbpe.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest rbf_ffn/tests/test_train.py::test_get_experiment_dir_includes_superbpe_tag rbf_ffn/tests/test_train.py::test_train_uses_superbpe_dataloaders rbf_ffn/tests/test_train.py::test_train_uses_r50k_dataloaders_by_default -v
```

Expected: failures — `get_experiment_dir` not exported, no superbpe dispatch exists yet.

- [ ] **Step 3: Update `train.py` — remove top-level data import, add dispatch, add naming tag**

**3a.** Replace the top-level import in `rbf_ffn/train.py`:

```python
# Remove this line:
from rbf_ffn.data import get_dataloaders
```

**3b.** In `get_experiment_dir`, add `_superbpe` tag. Find the block that builds `norm_tags` (after the `if cfg.mup:` block, before `name = f"{stamp}..."`), and append:

```python
    if cfg.tokenizer == "superbpe48576":
        norm_tags += "_superbpe"
```

**3c.** In `train()`, replace the data-loading line:

```python
    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = get_dataloaders(cfg)
```

with:

```python
    # ── Data ──────────────────────────────────────────────────────────────────
    if cfg.tokenizer == "superbpe48576":
        from rbf_ffn.superbpe_data import get_dataloaders as _get_dataloaders
    else:
        from rbf_ffn.data import get_dataloaders as _get_dataloaders
    train_loader, val_loader, _ = _get_dataloaders(cfg)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest rbf_ffn/tests/test_train.py::test_get_experiment_dir_includes_superbpe_tag rbf_ffn/tests/test_train.py::test_train_uses_superbpe_dataloaders rbf_ffn/tests/test_train.py::test_train_uses_r50k_dataloaders_by_default -v
```

Expected: all PASSED.

- [ ] **Step 5: Verify no existing train tests broke**

```bash
pytest rbf_ffn/tests/test_train.py -v
```

Expected: all previously passing tests still PASS.

- [ ] **Step 6: Commit**

```bash
git add rbf_ffn/train.py rbf_ffn/tests/test_train.py
git commit -m "feat(rbf_ffn): dispatch to superbpe_data when tokenizer=superbpe48576; add _superbpe naming tag"
```

---

## Task 4: Experiment config

**Files:**
- Create: `rbf_ffn/configs/baseline_xsa_superbpe.yaml`
- Modify: `rbf_ffn/tests/test_config.py`

**Interfaces:**
- Consumes: `ModelConfig` with `tokenizer`, `vocab_size`, `attn_type`, `ffn_type` fields

- [ ] **Step 1: Write the failing test**

Append to `rbf_ffn/tests/test_config.py`:

```python
def test_baseline_xsa_superbpe_yaml_loads():
    cfg = load_config(CONFIGS_DIR / "baseline_xsa_superbpe.yaml")
    assert cfg.tokenizer == "superbpe48576"
    assert cfg.vocab_size == 48576
    assert cfg.attn_type == "xsa"
    assert cfg.ffn_type == "swiglu"
    assert cfg.qk_norm is True
    assert cfg.linear_weight_norm is True
    assert cfg.d_model == 256
    assert cfg.n_layers == 6
    assert cfg.seq_len == 512
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest rbf_ffn/tests/test_config.py::test_baseline_xsa_superbpe_yaml_loads -v
```

Expected: FAIL — file does not exist yet.

- [ ] **Step 3: Create the config file**

Create `rbf_ffn/configs/baseline_xsa_superbpe.yaml`:

```yaml
# XSA + SwiGLU baseline with SuperBPE tokenizer (vocab=48576, t=43718).
# Matches the current best-performing config family (XSA + qknorm + wnorm)
# with only the tokenizer and vocab_size changed.
attn_type: xsa
ffn_type: swiglu
vocab_size: 48576
tokenizer: superbpe48576
d_model: 256
n_heads: 8
n_layers: 6
seq_len: 512
batch_size: 32
n_epochs: 3
qk_norm: true
linear_weight_norm: true
seed: 42
muon_lr: 0.02
adamw_lr: 3.0e-4
adamw_wd: 0.1
warmup_ratio: 0.02
grad_clip: 1.0
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest rbf_ffn/tests/test_config.py::test_baseline_xsa_superbpe_yaml_loads -v
```

Expected: PASSED.

- [ ] **Step 5: Run full test suite to confirm nothing broke**

```bash
pytest rbf_ffn/tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add rbf_ffn/configs/baseline_xsa_superbpe.yaml rbf_ffn/tests/test_config.py
git commit -m "feat(rbf_ffn): add baseline_xsa_superbpe.yaml experiment config"
```

---

## Self-Review

**Spec coverage:**
- ✓ `superbpe_data.py` with two-stage tokenizer (Task 2)
- ✓ `ModelConfig.tokenizer` field + validation (Task 1)
- ✓ `train.py` dispatch + `_superbpe` naming tag (Task 3)
- ✓ `baseline_xsa_superbpe.yaml` config (Task 4)
- ✓ Tests: vocab size, no special tokens (covered via `tok.get_vocab_size()`), chunk shape, config validation, config YAML (Tasks 1–4)
- ✓ Cache filenames: `stage1.json`, `stage2.json`, `{split}_superbpe48576_{seq_len}.pt` (Task 2)
- ✓ Fallback: `_train_stage2` uses `Tokenizer.to_str()` / `from_str()` warm-start; BpeTrainer continues from existing vocab

**Type consistency:**
- `_train_stage1` returns `Tokenizer` → `_train_stage2` consumes `Tokenizer` ✓
- `_build_superbpe_tokenizer` returns `Tokenizer` → `_load_split` consumes `Tokenizer` ✓
- `get_dataloaders(cfg)` matches existing signature in `rbf_ffn.data` ✓
- `get_experiment_dir` already in `train.py` public scope; test imports it directly ✓

**Note on `test_train.py` import:** The test file already imports from `rbf_ffn.train`; Task 3 adds `get_experiment_dir` to that import. Also add `MagicMock` to the existing `from unittest.mock import patch` import. The `_tiny_cfg` and `_fake_loaders` helpers already exist in `test_train.py`.
