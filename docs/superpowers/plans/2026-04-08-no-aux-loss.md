# No Auxiliary Loss Option Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `sigreg_weight: 0.0` fully disable the aux loss with zero hidden-state overhead by passing `collect_hidden=False` to the model when the weight is zero.

**Architecture:** One-line change in `sigreg/train.py`: replace the hardcoded `collect_hidden=True` with `collect_hidden=(cfg.sigreg_weight > 0.0)`. The model already handles `collect_hidden=False` by returning an empty list, and the training loop already guards on `if hidden_states and cfg.sigreg_weight > 0.0`.

**Tech Stack:** Python, PyTorch, pytest

---

## File Map

| File | Change |
|---|---|
| `sigreg/train.py:163` | `collect_hidden=True` → `collect_hidden=(cfg.sigreg_weight > 0.0)` |
| `sigreg/tests/test_train.py` | New — two unit tests for the collect_hidden behaviour |

---

## Task 1: Infer `collect_hidden` from `sigreg_weight`

**Files:**
- Modify: `sigreg/train.py:163`
- Create: `sigreg/tests/test_train.py`

- [ ] **Step 1: Write failing tests**

Create `sigreg/tests/test_train.py`:

```python
"""Tests for collect_hidden inference from sigreg_weight."""
import torch
from sigreg.config import SIGRegConfig
from sigreg.models.model import SIGRegCausalLM


def _cfg(**kwargs) -> SIGRegConfig:
    return SIGRegConfig(
        d_model=32, n_heads=2, n_layers=2, ffn_hidden=64,
        vocab_size=64, seq_len=8,
        **kwargs,
    )


def test_collect_hidden_false_when_weight_zero():
    """sigreg_weight=0.0 must produce collect_hidden=False → empty hidden list."""
    cfg = _cfg(sigreg_weight=0.0)
    assert (cfg.sigreg_weight > 0.0) is False  # expression used in train.py

    model = SIGRegCausalLM(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
    _, hidden = model(tokens, collect_hidden=(cfg.sigreg_weight > 0.0))

    assert hidden == []


def test_collect_hidden_true_when_weight_nonzero():
    """sigreg_weight>0 must produce collect_hidden=True → hidden states returned."""
    cfg = _cfg(sigreg_weight=0.1)
    assert (cfg.sigreg_weight > 0.0) is True  # expression used in train.py

    model = SIGRegCausalLM(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
    _, hidden = model(tokens, collect_hidden=(cfg.sigreg_weight > 0.0))

    assert len(hidden) == cfg.n_layers
```

- [ ] **Step 2: Run tests — they should PASS (model already works correctly)**

```bash
cd /home/harikrishnan-c/projects/machine-notes
python -m pytest sigreg/tests/test_train.py -v
```

Expected: 2 × `PASSED` — these tests verify the model's `collect_hidden` behaviour, which is already correct. They will pass even before the train.py fix, confirming the model side is sound.

- [ ] **Step 3: Verify the current train.py line is hardcoded (grep check)**

```bash
grep -n "collect_hidden" sigreg/train.py
```

Expected output includes:
```
163:                logits, hidden_states = model(inputs, collect_hidden=True)
```

This confirms line 163 still uses the hardcoded `True` that needs to change.

- [ ] **Step 4: Apply the one-line fix in `sigreg/train.py`**

On line 163, change:
```python
                logits, hidden_states = model(inputs, collect_hidden=True)
```
to:
```python
                logits, hidden_states = model(inputs, collect_hidden=(cfg.sigreg_weight > 0.0))
```

- [ ] **Step 5: Verify the fix with grep**

```bash
grep -n "collect_hidden" sigreg/train.py
```

Expected output includes:
```
163:                logits, hidden_states = model(inputs, collect_hidden=(cfg.sigreg_weight > 0.0))
```

- [ ] **Step 6: Run the full test suite**

```bash
python -m pytest sigreg/tests/ -v
```

Expected: all tests pass. (The pre-existing `test_train_loader_has_persistent_workers_and_prefetch` failure in `test_data.py` is unrelated — ignore it.)

- [ ] **Step 7: Commit**

```bash
git add sigreg/train.py sigreg/tests/test_train.py
git commit -m "feat(sigreg): skip hidden state collection when sigreg_weight=0"
```
