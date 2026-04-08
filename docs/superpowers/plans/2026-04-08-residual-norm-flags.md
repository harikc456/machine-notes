# Residual Connections and Norm Flags Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `use_residual` and `norm_type` flags to `SIGRegConfig` and implement all four forward-pass combinations in `TransformerBlock` (renamed from `PlainBlock`).

**Architecture:** Two new fields on `SIGRegConfig` control skip connections and pre-norm layers. `PlainBlock` is renamed `TransformerBlock` and uses these flags to choose between four forward paths. `SIGRegCausalLM` only changes its import. Existing configs default to `false`/`"none"` — no behaviour change.

**Tech Stack:** Python 3.11, PyTorch, pytest, PyYAML

---

## File Map

| File | Change |
|---|---|
| `sigreg/config.py` | Add `use_residual: bool`, `norm_type: str`, update `__post_init__` validation |
| `sigreg/models/block.py` | Rename `PlainBlock` → `TransformerBlock`, add conditional norms and forward paths |
| `sigreg/models/model.py` | Update import and instantiation `PlainBlock` → `TransformerBlock` |
| `sigreg/configs/baseline.yaml` | Add `use_residual: false` and `norm_type: "none"` |
| `sigreg/configs/weak_loss.yaml` | Add `use_residual: false` and `norm_type: "none"` |
| `sigreg/tests/test_config.py` | New — config field and validation tests |
| `sigreg/tests/test_block.py` | New — all four `TransformerBlock` forward-path tests |
| `sigreg/tests/test_model.py` | New — `SIGRegCausalLM` smoke tests with new flags |

---

## Task 1: Config Fields + Validation Tests

**Files:**
- Modify: `sigreg/config.py`
- Create: `sigreg/tests/test_config.py`

- [ ] **Step 1: Write failing config tests**

Create `sigreg/tests/test_config.py`:

```python
"""Tests for SIGRegConfig use_residual and norm_type fields."""
import pytest
from sigreg.config import SIGRegConfig


def test_config_defaults_are_plain():
    cfg = SIGRegConfig()
    assert cfg.use_residual is False
    assert cfg.norm_type == "none"


def test_config_accepts_valid_norm_types():
    for nt in ("none", "rmsnorm", "layernorm"):
        cfg = SIGRegConfig(norm_type=nt)
        assert cfg.norm_type == nt


def test_config_rejects_invalid_norm_type():
    with pytest.raises(AssertionError):
        SIGRegConfig(norm_type="batchnorm")


def test_config_accepts_use_residual_true():
    cfg = SIGRegConfig(use_residual=True)
    assert cfg.use_residual is True
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/harikrishnan-c/projects/machine-notes
python -m pytest sigreg/tests/test_config.py -v
```

Expected: `FAILED` — `SIGRegConfig` has no `use_residual` or `norm_type` attribute.

- [ ] **Step 3: Add fields and validation to `sigreg/config.py`**

In the `SIGRegConfig` dataclass, add after the `tie_embeddings` field:

```python
    # ── Block architecture ────────────────────────────────────────────────────
    use_residual: bool = False      # add skip connections around attn and ffn
    norm_type: str = "none"         # "none" | "rmsnorm" | "layernorm"
```

In `__post_init__`, add after the existing `sigreg_layers` assert:

```python
        assert self.norm_type in ("none", "rmsnorm", "layernorm"), (
            f"norm_type must be 'none', 'rmsnorm', or 'layernorm', got '{self.norm_type}'"
        )
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest sigreg/tests/test_config.py -v
```

Expected: 4 × `PASSED`

- [ ] **Step 5: Commit**

```bash
git add sigreg/config.py sigreg/tests/test_config.py
git commit -m "feat(sigreg): add use_residual and norm_type config fields"
```

---

## Task 2: TransformerBlock with All Four Forward Paths

**Files:**
- Modify: `sigreg/models/block.py`
- Create: `sigreg/tests/test_block.py`

- [ ] **Step 1: Write failing block tests**

Create `sigreg/tests/test_block.py`:

```python
"""Tests for TransformerBlock — all four use_residual × norm_type combinations."""
import torch
import pytest
from sigreg.config import SIGRegConfig
from sigreg.models.block import TransformerBlock


def _cfg(**kwargs) -> SIGRegConfig:
    """Tiny config for fast tests."""
    return SIGRegConfig(
        d_model=32, n_heads=2, ffn_hidden=64,
        vocab_size=64, seq_len=8,
        **kwargs,
    )


def _x(cfg: SIGRegConfig) -> torch.Tensor:
    return torch.randn(2, cfg.seq_len, cfg.d_model)


def test_plain_output_shape():
    cfg = _cfg()
    out = TransformerBlock(cfg)(_x(cfg))
    assert out.shape == (2, cfg.seq_len, cfg.d_model)


def test_residual_only_output_shape():
    cfg = _cfg(use_residual=True)
    out = TransformerBlock(cfg)(_x(cfg))
    assert out.shape == (2, cfg.seq_len, cfg.d_model)


def test_rmsnorm_no_residual_output_shape():
    cfg = _cfg(norm_type="rmsnorm")
    out = TransformerBlock(cfg)(_x(cfg))
    assert out.shape == (2, cfg.seq_len, cfg.d_model)


def test_prenorm_residual_rmsnorm_output_shape():
    cfg = _cfg(use_residual=True, norm_type="rmsnorm")
    out = TransformerBlock(cfg)(_x(cfg))
    assert out.shape == (2, cfg.seq_len, cfg.d_model)


def test_prenorm_residual_layernorm_output_shape():
    cfg = _cfg(use_residual=True, norm_type="layernorm")
    out = TransformerBlock(cfg)(_x(cfg))
    assert out.shape == (2, cfg.seq_len, cfg.d_model)


def test_plain_has_no_norm_modules():
    block = TransformerBlock(_cfg())
    assert block.norm_attn is None
    assert block.norm_ffn is None


def test_rmsnorm_block_has_norm_modules():
    block = TransformerBlock(_cfg(norm_type="rmsnorm"))
    assert block.norm_attn is not None
    assert block.norm_ffn is not None


def test_layernorm_block_has_norm_modules():
    block = TransformerBlock(_cfg(norm_type="layernorm"))
    assert block.norm_attn is not None
    assert block.norm_ffn is not None
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest sigreg/tests/test_block.py -v
```

Expected: `ImportError` — `TransformerBlock` does not exist yet.

- [ ] **Step 3: Replace `PlainBlock` with `TransformerBlock` in `sigreg/models/block.py`**

Replace the entire `PlainBlock` class (the `# ── Block` section at the bottom of the file) with:

```python
# ── Block ─────────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    Transformer block with optional residual connections and pre-norm.

    Controlled by cfg.use_residual and cfg.norm_type:

        use_residual=False, norm_type="none"  → x = ffn(attn(x))
        use_residual=True,  norm_type="none"  → x = x + attn(x); x = x + ffn(x)
        use_residual=False, norm_type=*       → x = ffn(attn(norm_attn(x)))
        use_residual=True,  norm_type=*       → x = x + attn(norm_attn(x)); x = x + ffn(norm_ffn(x))

    When both use_residual and a norm are active, pre-norm ordering is used.
    """

    def __init__(self, cfg: SIGRegConfig):
        super().__init__()
        self.attn = CausalSelfAttention(cfg)
        self.ffn  = SwiGLUFFN(cfg)
        self.use_residual = cfg.use_residual

        if cfg.norm_type == "rmsnorm":
            self.norm_attn: nn.Module | None = nn.RMSNorm(cfg.d_model)
            self.norm_ffn:  nn.Module | None = nn.RMSNorm(cfg.d_model)
        elif cfg.norm_type == "layernorm":
            self.norm_attn = nn.LayerNorm(cfg.d_model)
            self.norm_ffn  = nn.LayerNorm(cfg.d_model)
        else:
            self.norm_attn = None
            self.norm_ffn  = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d_model) → (B, N, d_model)"""
        attn_in = self.norm_attn(x) if self.norm_attn is not None else x
        attn_out = self.attn(attn_in)
        x = x + attn_out if self.use_residual else attn_out

        ffn_in = self.norm_ffn(x) if self.norm_ffn is not None else x
        ffn_out = self.ffn(ffn_in)
        x = x + ffn_out if self.use_residual else ffn_out

        return x
```

- [ ] **Step 4: Run block tests**

```bash
python -m pytest sigreg/tests/test_block.py -v
```

Expected: 8 × `PASSED`

- [ ] **Step 5: Commit**

```bash
git add sigreg/models/block.py sigreg/tests/test_block.py
git commit -m "feat(sigreg): rename PlainBlock→TransformerBlock, add residual+norm forward paths"
```

---

## Task 3: Update Model Import

**Files:**
- Modify: `sigreg/models/model.py`
- Create: `sigreg/tests/test_model.py`

- [ ] **Step 1: Write failing model smoke tests**

Create `sigreg/tests/test_model.py`:

```python
"""Smoke tests for SIGRegCausalLM with new residual/norm flags."""
import torch
from sigreg.config import SIGRegConfig
from sigreg.models.model import SIGRegCausalLM


def _cfg(**kwargs) -> SIGRegConfig:
    return SIGRegConfig(
        d_model=32, n_heads=2, n_layers=3, ffn_hidden=64,
        vocab_size=64, seq_len=8,
        **kwargs,
    )


def _tokens(cfg: SIGRegConfig) -> torch.Tensor:
    return torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))


def test_plain_model_forward():
    cfg = _cfg()
    model = SIGRegCausalLM(cfg)
    logits, hidden = model(_tokens(cfg), collect_hidden=True)
    assert logits.shape == (2, cfg.seq_len, cfg.vocab_size)
    assert len(hidden) == cfg.n_layers


def test_prenorm_residual_model_forward():
    cfg = _cfg(use_residual=True, norm_type="rmsnorm")
    model = SIGRegCausalLM(cfg)
    logits, hidden = model(_tokens(cfg), collect_hidden=True)
    assert logits.shape == (2, cfg.seq_len, cfg.vocab_size)
    assert len(hidden) == cfg.n_layers


def test_plain_model_collect_hidden_false():
    cfg = _cfg()
    model = SIGRegCausalLM(cfg)
    logits, hidden = model(_tokens(cfg), collect_hidden=False)
    assert logits.shape == (2, cfg.seq_len, cfg.vocab_size)
    assert hidden == []
```

- [ ] **Step 2: Run to confirm it fails**

```bash
python -m pytest sigreg/tests/test_model.py -v
```

Expected: `ImportError` — `model.py` still imports `PlainBlock`.

- [ ] **Step 3: Update `sigreg/models/model.py`**

Change the import line from:
```python
from sigreg.models.block import PlainBlock
```
to:
```python
from sigreg.models.block import TransformerBlock
```

Change the `nn.ModuleList` line in `__init__` from:
```python
self.blocks = nn.ModuleList([PlainBlock(cfg) for _ in range(cfg.n_layers)])
```
to:
```python
self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
```

- [ ] **Step 4: Run full test suite**

```bash
python -m pytest sigreg/tests/ -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add sigreg/models/model.py sigreg/tests/test_model.py
git commit -m "feat(sigreg): update SIGRegCausalLM to use TransformerBlock"
```

---

## Task 4: Update YAML Configs

**Files:**
- Modify: `sigreg/configs/baseline.yaml`
- Modify: `sigreg/configs/weak_loss.yaml`

- [ ] **Step 1: Add flags to `baseline.yaml`**

Add after the `tie_embeddings: false` line:

```yaml
use_residual: false
norm_type: "none"
```

- [ ] **Step 2: Add flags to `weak_loss.yaml`**

Add after the `tie_embeddings: true` line:

```yaml
use_residual: false
norm_type: "none"
```

- [ ] **Step 3: Verify both configs load without error**

```bash
python -c "
from sigreg.config import load_config
from pathlib import Path
b = load_config(Path('sigreg/configs/baseline.yaml'))
w = load_config(Path('sigreg/configs/weak_loss.yaml'))
assert b.use_residual is False and b.norm_type == 'none'
assert w.use_residual is False and w.norm_type == 'none'
print('OK')
"
```

Expected: `OK`

- [ ] **Step 4: Run full test suite**

```bash
python -m pytest sigreg/tests/ -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add sigreg/configs/baseline.yaml sigreg/configs/weak_loss.yaml
git commit -m "feat(sigreg): add use_residual and norm_type to baseline configs"
```
