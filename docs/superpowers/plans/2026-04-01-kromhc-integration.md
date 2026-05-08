# KromHC Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate `kromhc_transformer/` into `rbf_ffn/` by adding KromHC head mixing as a composable, opt-in wrapper that can be stacked on any existing block type.

**Architecture:** A `KromHCWrapper` in `transformer_block.py` wraps any inner block and applies head mixing as an additive residual after the block output. `CausalLM` wraps each block if `cfg.use_kromhc=True` and returns `(logits, hs)` always, where `hs` is a list of per-layer H tensors (empty when not using KromHC). `kromhc_transformer/` is moved to `archive/`.

**Tech Stack:** PyTorch, pytest, YAML configs

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `rbf_ffn/models/head_mixer.py` | **Create** | `KromHCHeadMixer` — Kronecker-factored doubly-stochastic head mixing |
| `rbf_ffn/models/transformer_block.py` | **Modify** | Add `KromHCWrapper` class |
| `rbf_ffn/config.py` | **Modify** | Add `use_kromhc`, `kromhc_mixer_hidden` fields |
| `rbf_ffn/models/model.py` | **Modify** | Conditionally wrap blocks; change `forward()` to return `(logits, hs)` |
| `rbf_ffn/train.py` | **Modify** | Unpack `(logits, hs)`, add `collect_kromhc_stats()`, log H stats per epoch |
| `rbf_ffn/tests/test_head_mixer.py` | **Create** | Unit tests for `KromHCHeadMixer` |
| `rbf_ffn/tests/test_kromhc_wrapper.py` | **Create** | Unit tests for `KromHCWrapper` |
| `rbf_ffn/tests/test_model.py` | **Modify** | Update call sites from `logits = model(t)` to `logits, _ = model(t)` |
| `rbf_ffn/configs/baseline_kromhc.yaml` | **Create** | Baseline + KromHC |
| `rbf_ffn/configs/baseline_qk_norm_kromhc.yaml` | **Create** | Baseline + QK norm + KromHC |
| `rbf_ffn/configs/pfd_rationalglu_qk_norm_weight_norm_kromhc.yaml` | **Create** | Best FFN + KromHC |
| `rbf_ffn/OVERVIEW.md` | **Modify** | Add consolidation note |
| `archive/kromhc_transformer/` | **Create** (via git mv) | Archive of original implementation |

---

## Task 1: Port `head_mixer.py` and write tests

**Files:**
- Create: `rbf_ffn/tests/test_head_mixer.py`
- Create: `rbf_ffn/models/head_mixer.py`

- [ ] **Step 1: Write the failing tests**

```python
# rbf_ffn/tests/test_head_mixer.py
import math
import torch
import pytest
from rbf_ffn.models.head_mixer import KromHCHeadMixer

BS, N_HEADS, HEAD_DIM = 4, 8, 32


@pytest.fixture
def mixer():
    return KromHCHeadMixer(n_heads=N_HEADS, head_dim=HEAD_DIM)


def test_output_shapes(mixer):
    x = torch.randn(BS, N_HEADS, HEAD_DIM)
    mixed, H = mixer(x)
    assert mixed.shape == (BS, N_HEADS, HEAD_DIM)
    assert H.shape == (BS, N_HEADS, N_HEADS)


def test_H_row_sums_to_one(mixer):
    """H must be row-stochastic (each row sums to 1)."""
    x = torch.randn(BS, N_HEADS, HEAD_DIM)
    _, H = mixer(x)
    row_sums = H.sum(dim=-1)  # (BS, N_HEADS)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_H_col_sums_to_one(mixer):
    """H must be col-stochastic (each col sums to 1)."""
    x = torch.randn(BS, N_HEADS, HEAD_DIM)
    _, H = mixer(x)
    col_sums = H.sum(dim=-2)  # (BS, N_HEADS)
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-5)


def test_H_nonnegative(mixer):
    """All entries of H must be >= 0."""
    x = torch.randn(BS, N_HEADS, HEAD_DIM)
    _, H = mixer(x)
    assert (H >= -1e-6).all()


def test_gradient_flows_through_mixed(mixer):
    x = torch.randn(BS, N_HEADS, HEAD_DIM, requires_grad=True)
    mixed, _ = mixer(x)
    mixed.sum().backward()
    assert x.grad is not None
    for gen in mixer.weight_gens:
        for p in gen.parameters():
            assert p.grad is not None


def test_requires_power_of_two():
    with pytest.raises(AssertionError):
        KromHCHeadMixer(n_heads=6, head_dim=32)


def test_identity_when_equal_weights():
    """When both heads are weighted equally, mixing is the average permutation.
    This is a smoke test: mixed output should have finite values."""
    mixer = KromHCHeadMixer(n_heads=4, head_dim=16)
    x = torch.randn(2, 4, 16)
    mixed, H = mixer(x)
    assert torch.isfinite(mixed).all()
    assert torch.isfinite(H).all()


def test_d_context_override():
    """Custom d_context changes MLP input dim without error."""
    mixer = KromHCHeadMixer(n_heads=4, head_dim=16, d_context=8)
    x = torch.randn(2, 4, 16)
    mixed, H = mixer(x)
    assert mixed.shape == (2, 4, 16)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/harikrishnan-c/projects/machine-notes
python -m pytest rbf_ffn/tests/test_head_mixer.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError` — `head_mixer` does not exist yet.

- [ ] **Step 3: Create `rbf_ffn/models/head_mixer.py`**

```python
"""KromHC Head Mixer: Kronecker-factored doubly-stochastic head mixing."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import permutations


class KromHCHeadMixer(nn.Module):
    """
    Mixes attention head outputs using Kronecker-factored permutation matrices.

    For n_heads=8: factors=[2,2,2], builds 3 tiny 2×2 permutation matrices,
    Kronecker-chains them into an 8×8 doubly-stochastic matrix per token.

    Input:  (bs, n_heads, head_dim)
    Output: mixed (bs, n_heads, head_dim), H (bs, n_heads, n_heads)
    """

    def __init__(self, n_heads: int = 8, head_dim: int = 64, d_context: int = None):
        super().__init__()
        self.n = n_heads
        self.head_dim = head_dim
        if d_context is None:
            d_context = head_dim

        k = int(math.log2(n_heads))
        assert 2 ** k == n_heads, f"n_heads ({n_heads}) must be a power of 2"
        self.K = k

        self.perm_bases = nn.ParameterList()
        self.weight_gens = nn.ModuleList()

        for _ in range(k):
            basis = torch.zeros(2, 2, 2)
            for idx, p in enumerate(permutations(range(2))):
                for r, c in enumerate(p):
                    basis[idx, r, c] = 1.0
            self.perm_bases.append(nn.Parameter(basis, requires_grad=False))

            self.weight_gens.append(nn.Sequential(
                nn.Linear(d_context, 32, bias=False),
                nn.ReLU(),
                nn.Linear(32, 2, bias=False),
            ))

    def _batched_kronecker(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Batched Kronecker product: (bs, m, m) ⊗ (bs, p, p) → (bs, m*p, m*p)"""
        bs, m, _ = A.shape
        p = B.shape[1]
        return torch.einsum('b i j, b k l -> b i k j l', A, B).reshape(bs, m * p, m * p)

    def forward(self, x: torch.Tensor):
        """
        x: (bs, n_heads, head_dim)
        Returns: (mixed: same shape, H: (bs, n_heads, n_heads))
        """
        bs, n, d = x.shape
        assert n == self.n, f"Expected {self.n} heads, got {n}"

        context = x.mean(dim=1)  # (bs, head_dim)

        small_us = []
        for gen, basis in zip(self.weight_gens, self.perm_bases):
            logits = gen(context)                        # (bs, 2)
            a = F.softmax(logits, dim=-1)               # convex weights
            U = (a @ basis.view(2, -1)).view(bs, 2, 2)  # (bs, 2, 2)
            small_us.append(U)

        H = small_us[0]
        for U in small_us[1:]:
            H = self._batched_kronecker(H, U)  # (bs, n_heads, n_heads)

        out = torch.matmul(H, x)  # (bs, n_heads, head_dim)
        return out, H
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /home/harikrishnan-c/projects/machine-notes
python -m pytest rbf_ffn/tests/test_head_mixer.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/harikrishnan-c/projects/machine-notes
git add rbf_ffn/models/head_mixer.py rbf_ffn/tests/test_head_mixer.py
git commit -m "feat(models): add KromHCHeadMixer with Kronecker-factored head mixing"
```

---

## Task 2: Add `KromHCWrapper` to `transformer_block.py`

**Files:**
- Create: `rbf_ffn/tests/test_kromhc_wrapper.py`
- Modify: `rbf_ffn/models/transformer_block.py`

- [ ] **Step 1: Write the failing tests**

```python
# rbf_ffn/tests/test_kromhc_wrapper.py
import torch
import pytest
from rbf_ffn.config import ModelConfig
from rbf_ffn.models.transformer_block import LlamaBlock, KromHCWrapper

D, H, B, N = 32, 4, 2, 16


def make_wrapper() -> KromHCWrapper:
    cfg = ModelConfig(d_model=D, n_heads=H, dropout=0.0)
    inner = LlamaBlock(cfg)
    return KromHCWrapper(inner, cfg)


def test_output_shape():
    wrapper = make_wrapper()
    x = torch.randn(B, N, D)
    out, H_mat = wrapper(x)
    assert out.shape == (B, N, D)


def test_H_shape():
    wrapper = make_wrapper()
    x = torch.randn(B, N, D)
    _, H_mat = wrapper(x)
    assert H_mat.shape == (B, N, H, H)


def test_returns_tuple():
    wrapper = make_wrapper()
    result = wrapper(torch.randn(B, N, D))
    assert isinstance(result, tuple) and len(result) == 2


def test_gradient_flows():
    wrapper = make_wrapper()
    x = torch.randn(B, N, D, requires_grad=True)
    out, _ = wrapper(x)
    out.sum().backward()
    assert x.grad is not None


def test_zero_mixer_proj_is_identity_of_inner_block():
    """When mixer_proj weights are zero, wrapper output == inner_block output."""
    cfg = ModelConfig(d_model=D, n_heads=H, dropout=0.0)
    inner = LlamaBlock(cfg)
    wrapper = KromHCWrapper(inner, cfg)
    with torch.no_grad():
        wrapper.mixer_proj.weight.zero_()
    x = torch.randn(B, N, D)
    wrapper_out, _ = wrapper(x)
    inner_out = inner(x)
    assert torch.allclose(wrapper_out, inner_out, atol=1e-5)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/harikrishnan-c/projects/machine-notes
python -m pytest rbf_ffn/tests/test_kromhc_wrapper.py -v 2>&1 | head -20
```

Expected: `ImportError` — `KromHCWrapper` does not exist yet.

- [ ] **Step 3: Add `KromHCWrapper` to `transformer_block.py`**

Add this import at the top of `rbf_ffn/models/transformer_block.py` (after existing imports):

```python
from rbf_ffn.models.head_mixer import KromHCHeadMixer
```

Add this class at the end of the file:

```python
class KromHCWrapper(nn.Module):
    """
    Wraps any transformer block with KromHC head mixing.

    Applies head mixing as an additive residual after the inner block:

        x_block = inner_block(x)
        heads   = x_block reshaped to (B*N, n_heads, head_dim)
        mixed   = KromHCHeadMixer(heads)
        out     = x_block + mixer_proj(mixed reshaped back)

    Returns (out, H) where H: (B, N, n_heads, n_heads).
    """

    def __init__(self, inner_block: nn.Module, cfg: ModelConfig):
        super().__init__()
        self.inner_block = inner_block
        self.n_heads   = cfg.n_heads
        self.head_dim  = cfg.d_model // cfg.n_heads
        self.head_mixer = KromHCHeadMixer(
            n_heads=cfg.n_heads,
            head_dim=self.head_dim,
            d_context=self.head_dim,
        )
        self.mixer_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor):
        """
        x: (B, N, D)
        Returns: (out: (B, N, D), H: (B, N, n_heads, n_heads))
        """
        x_block = self.inner_block(x)                           # (B, N, D)
        B, N, D = x_block.shape
        heads = x_block.view(B * N, self.n_heads, self.head_dim)
        mixed, H = self.head_mixer(heads)                       # mixed: (B*N, n_heads, head_dim)
        correction = self.mixer_proj(mixed.view(B, N, D))
        H_4d = H.view(B, N, self.n_heads, self.n_heads)
        return x_block + correction, H_4d
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /home/harikrishnan-c/projects/machine-notes
python -m pytest rbf_ffn/tests/test_kromhc_wrapper.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Run full test suite to confirm nothing broken**

```bash
cd /home/harikrishnan-c/projects/machine-notes
python -m pytest rbf_ffn/tests/ -v --ignore=rbf_ffn/tests/test_train.py -q
```

Expected: all pre-existing tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /home/harikrishnan-c/projects/machine-notes
git add rbf_ffn/models/transformer_block.py rbf_ffn/tests/test_kromhc_wrapper.py
git commit -m "feat(models): add KromHCWrapper for composable head mixing"
```

---

## Task 3: Add `use_kromhc` to `config.py`

**Files:**
- Modify: `rbf_ffn/config.py`
- Modify: `rbf_ffn/tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

Open `rbf_ffn/tests/test_config.py` and add these tests (append to end of file):

```python
def test_use_kromhc_default_false():
    cfg = ModelConfig()
    assert cfg.use_kromhc is False


def test_kromhc_mixer_hidden_default():
    cfg = ModelConfig()
    assert cfg.kromhc_mixer_hidden == 32


def test_use_kromhc_loads_from_yaml(tmp_path):
    yaml_text = "use_kromhc: true\nkromhc_mixer_hidden: 64\n"
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml_text)
    cfg = load_config(path)
    assert cfg.use_kromhc is True
    assert cfg.kromhc_mixer_hidden == 64
```

- [ ] **Step 2: Run to confirm they fail**

```bash
cd /home/harikrishnan-c/projects/machine-notes
python -m pytest rbf_ffn/tests/test_config.py -v -k "kromhc" 2>&1 | head -20
```

Expected: `TypeError` — `ModelConfig` has no field `use_kromhc`.

- [ ] **Step 3: Add fields to `config.py`**

In `rbf_ffn/config.py`, add after the Kronecker MLP block (after line `kronecker_delta_rank: int = 16`):

```python
    # KromHC head mixing
    use_kromhc: bool = False           # wrap any block with KromHC head mixing
    kromhc_mixer_hidden: int = 32      # hidden dim of per-factor weight MLP
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /home/harikrishnan-c/projects/machine-notes
python -m pytest rbf_ffn/tests/test_config.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/harikrishnan-c/projects/machine-notes
git add rbf_ffn/config.py rbf_ffn/tests/test_config.py
git commit -m "feat(config): add use_kromhc and kromhc_mixer_hidden fields"
```

---

## Task 4: Update `model.py` — wrap blocks, change forward return

**Files:**
- Modify: `rbf_ffn/models/model.py`
- Modify: `rbf_ffn/tests/test_model.py`

- [ ] **Step 1: Update `test_model.py` call sites first**

The existing tests do `logits = model(tokens)` or `assert model(tokens).shape == ...`.
After this task `model.forward()` returns `(logits, hs)` — update all 7 affected tests.

Replace the body of `test_baseline_output_shape`:
```python
def test_baseline_output_shape():
    model = make_model("baseline")
    tokens = torch.randint(0, VOCAB, (B, N))
    logits, hs = model(tokens)
    assert logits.shape == (B, N, VOCAB)
    assert hs == []
```

Replace `test_gradient_flows_baseline`:
```python
def test_gradient_flows_baseline():
    model = make_model("baseline")
    tokens = torch.randint(0, VOCAB, (B, N))
    logits, _ = model(tokens)
    logits.sum().backward()
    assert model.token_embedding.weight.grad is not None
```

Replace `test_rationalglu_output_shape`:
```python
def test_rationalglu_output_shape():
    model = make_model("rationalglu")
    tokens = torch.randint(0, VOCAB, (B, N))
    logits, _ = model(tokens)
    assert logits.shape == (B, N, VOCAB)
```

Replace `test_first_order_pfd_rational_output_shape`:
```python
def test_first_order_pfd_rational_output_shape():
    model = make_model("first_order_pfd_rational")
    tokens = torch.randint(0, VOCAB, (B, N))
    logits, _ = model(tokens)
    assert logits.shape == (B, N, VOCAB)
```

Replace `test_polar_mlp_output_shape`:
```python
def test_polar_mlp_output_shape():
    model = make_model("polar_mlp")
    tokens = torch.randint(0, VOCAB, (B, N))
    logits, _ = model(tokens)
    assert logits.shape == (B, N, VOCAB)
```

Replace `test_polar_mlp_gradient_flows`:
```python
def test_polar_mlp_gradient_flows():
    model = make_model("polar_mlp")
    tokens = torch.randint(0, VOCAB, (B, N))
    logits, _ = model(tokens)
    loss = logits.sum()
    assert torch.isfinite(loss)
    loss.backward()
    assert model.token_embedding.weight.grad is not None
```

Replace `test_kronecker_delta_output_shape`:
```python
def test_kronecker_delta_output_shape():
    model = _make_delta_model()
    tokens = torch.randint(0, VOCAB, (B, N))
    logits, _ = model(tokens)
    assert logits.shape == (B, N, VOCAB)
```

Also add two new tests at the end of `test_model.py`:

```python
def _make_kromhc_model() -> CausalLM:
    cfg = ModelConfig(
        d_model=D, n_heads=H, n_layers=L,
        vocab_size=VOCAB, seq_len=N,
        model_type="baseline",
        ffn_hidden=86,
        dropout=0.0,
        use_kromhc=True,
    )
    return CausalLM(cfg)


def test_kromhc_output_shape():
    model = _make_kromhc_model()
    tokens = torch.randint(0, VOCAB, (B, N))
    logits, hs = model(tokens)
    assert logits.shape == (B, N, VOCAB)
    assert len(hs) == L


def test_kromhc_H_shape():
    model = _make_kromhc_model()
    tokens = torch.randint(0, VOCAB, (B, N))
    _, hs = model(tokens)
    for H in hs:
        assert H.shape == (B, N, H_var, H_var)  # H_var = n_heads
```

Wait, the variable `H` is already used as a constant in that file (`H = 4`). Let me fix that:

```python
def test_kromhc_H_shape():
    model = _make_kromhc_model()
    tokens = torch.randint(0, VOCAB, (B, N))
    _, hs = model(tokens)
    for H_mat in hs:
        assert H_mat.shape == (B, N, H, H)  # H=4 heads


def test_kromhc_no_duplicate_params():
    from rbf_ffn.models.model import build_optimizer_groups
    model = _make_kromhc_model()
    muon_params, adamw_params = build_optimizer_groups(model)
    all_ids = [id(p) for p in muon_params] + [id(p) for p in adamw_params]
    assert len(all_ids) == len(set(all_ids))
```

- [ ] **Step 2: Run to confirm failing tests**

```bash
cd /home/harikrishnan-c/projects/machine-notes
python -m pytest rbf_ffn/tests/test_model.py -v 2>&1 | tail -20
```

Expected: multiple failures — `tuple object has no attribute 'shape'` and `ImportError` on `_make_kromhc_model`.

- [ ] **Step 3: Update `model.py`**

Replace the full contents of `rbf_ffn/models/model.py`:

```python
# rbf_ffn/models/model.py
from __future__ import annotations
import torch
import torch.nn as nn
from rbf_ffn.config import ModelConfig
from rbf_ffn.models.transformer_block import (
    LlamaBlock, RationalBlock, RationalGLUBlock,
    PFDRationalBlock, PFDRationalGLUBlock, FirstOrderPFDRationalBlock,
    PolarMLPBlock, PolarAttnBlock, PolarFullBlock,
    KromHCWrapper,
)


def build_optimizer_groups(
    model: nn.Module,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Split model parameters into Muon and AdamW groups.

    Rules (first match wins, applied after deduplication by tensor id):
      1. "sigma_raw" in name           → AdamW
      2. param is token embedding weight → AdamW
      3. "delta_" in name              → AdamW  (KroneckerDeltaLinear delta_C/delta_D)
      4. param.ndim == 2               → Muon
      5. else                          → AdamW

    Returns (muon_params, adamw_params).
    """
    emb_id = id(model.token_embedding.weight)   # type: ignore[attr-defined]
    seen: set[int] = set()
    muon: list[torch.Tensor] = []
    adamw: list[torch.Tensor] = []

    for name, param in model.named_parameters():
        pid = id(param)
        if pid in seen:
            continue
        seen.add(pid)

        if "sigma_raw" in name:
            adamw.append(param)
        elif pid == emb_id:
            adamw.append(param)
        elif "delta_" in name:
            adamw.append(param)
        elif param.ndim == 2:
            muon.append(param)
        else:
            adamw.append(param)

    return muon, adamw


class CausalLM(nn.Module):
    """
    Causal language model.

        token_embedding → N × Block → RMSNorm → lm_head (weight-tied)

    Block type is selected by cfg.model_type:
        "baseline"       → LlamaBlock          (SwiGLU FFN)
        "rational"       → RationalBlock       (RationalFFN)
        "rationalglu"    → RationalGLUBlock    (RationalGatedFFN)
        "pfd_rational"   → PFDRationalBlock    (PFDRationalFFN)
        "pfd_rationalglu"→ PFDRationalGLUBlock (PFDRationalGatedFFN)
        "first_order_pfd_rational" → FirstOrderPFDRationalBlock (FirstOrderPFDRationalFFN)
        "polar_mlp"      → PolarMLPBlock       (AdaptivePolarMLP)
        "polar_attn"     → PolarAttnBlock      (PolarAttention + SwiGLU)
        "polar_full"     → PolarFullBlock      (PolarAttention + AdaptivePolarMLP)

    If cfg.use_kromhc=True, each block is wrapped in KromHCWrapper.

    forward() always returns (logits, hs):
        logits: (B, N, vocab_size)
        hs:     list of H tensors (B, N, n_heads, n_heads) per layer, or [] if not using KromHC
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        BlockClass = {
            "baseline":        LlamaBlock,
            "rational":        RationalBlock,
            "rationalglu":     RationalGLUBlock,
            "pfd_rational":    PFDRationalBlock,
            "pfd_rationalglu": PFDRationalGLUBlock,
            "first_order_pfd_rational": FirstOrderPFDRationalBlock,
            "polar_mlp":       PolarMLPBlock,
            "polar_attn":      PolarAttnBlock,
            "polar_full":      PolarFullBlock,
        }[cfg.model_type]

        def make_block():
            block = BlockClass(cfg)
            if cfg.use_kromhc:
                return KromHCWrapper(block, cfg)
            return block

        self.use_kromhc = cfg.use_kromhc
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([make_block() for _ in range(cfg.n_layers)])
        self.norm = nn.RMSNorm(cfg.d_model)
        self.pre_lm_head_silu = cfg.pre_lm_head_silu
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, list]:
        """
        tokens: (B, N) integer token ids
        returns: (logits: (B, N, vocab_size), hs: list of H per layer or [])
        """
        x = self.token_embedding(tokens)
        hs: list[torch.Tensor] = []
        if self.use_kromhc:
            for block in self.blocks:
                x, H = block(x)
                hs.append(H.detach())
        else:
            for block in self.blocks:
                x = block(x)
        x = self.norm(x)
        if self.pre_lm_head_silu:
            x = torch.nn.functional.silu(x)
        return self.lm_head(x), hs
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /home/harikrishnan-c/projects/machine-notes
python -m pytest rbf_ffn/tests/test_model.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Run full test suite (excluding train) to confirm no regressions**

```bash
cd /home/harikrishnan-c/projects/machine-notes
python -m pytest rbf_ffn/tests/ -v --ignore=rbf_ffn/tests/test_train.py -q
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /home/harikrishnan-c/projects/machine-notes
git add rbf_ffn/models/model.py rbf_ffn/tests/test_model.py
git commit -m "feat(models): wrap blocks with KromHCWrapper, return (logits, hs) from forward"
```

---

## Task 5: Update `train.py` — handle tuple return, log H stats

**Files:**
- Modify: `rbf_ffn/train.py`
- Modify: `rbf_ffn/tests/test_train.py`

- [ ] **Step 1: Write failing tests first**

Append to `rbf_ffn/tests/test_train.py`:

```python
def test_kromhc_training_completes(tmp_path):
    """KromHC training produces metrics.jsonl with kromhc stats."""
    cfg = _tiny_cfg(use_kromhc=True, n_heads=2)
    exp_dir = _run_train(cfg, tmp_path)
    assert (exp_dir / "metrics.jsonl").exists()
    rows = [json.loads(l) for l in (exp_dir / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) >= 1
    assert "kromhc/H_row_entropy_mean" in rows[0]
    assert "kromhc/H_offdiag_mass_mean" in rows[0]


def test_non_kromhc_metrics_have_no_H_keys(tmp_path):
    """Non-KromHC run must not emit kromhc/* keys in metrics."""
    cfg = _tiny_cfg()
    exp_dir = _run_train(cfg, tmp_path)
    rows = [json.loads(l) for l in (exp_dir / "metrics.jsonl").read_text().splitlines()]
    assert "kromhc/H_row_entropy_mean" not in rows[0]
```

- [ ] **Step 2: Run to confirm they fail**

```bash
cd /home/harikrishnan-c/projects/machine-notes
python -m pytest rbf_ffn/tests/test_train.py -v -k "kromhc" 2>&1 | head -30
```

Expected: failures — train.py still unpacks `logits = model(inputs)`.

- [ ] **Step 3: Update `train.py`**

**3a. Add `collect_kromhc_stats` function** (after `apply_activation_coeff_norm`, before `evaluate`):

```python
@torch.no_grad()
def collect_kromhc_stats(model: CausalLM, batch: torch.Tensor, device: torch.device) -> dict:
    """Compute summary stats for KromHC H matrices using a single batch.

    Returns:
        kromhc/H_row_entropy_mean  — mean Shannon entropy of H rows
        kromhc/H_offdiag_mass_mean — mean fraction of probability mass on off-diagonal
    """
    model.eval()
    inputs = batch[:, :-1].to(device)
    _, hs = model(inputs)
    if not hs:
        return {}
    # hs: list of (B, N, n_heads, n_heads) — one per layer
    all_H = torch.stack(hs)                          # (n_layers, B, N, n_heads, n_heads)
    n_heads = all_H.shape[-1]
    flat = all_H.reshape(-1, n_heads, n_heads)        # (n_layers*B*N, n_heads, n_heads)

    eps = 1e-8
    row_entropy = -(flat * (flat + eps).log()).sum(dim=-1).mean().item()

    diag_mask = torch.eye(n_heads, device=flat.device, dtype=torch.bool)
    offdiag = flat.masked_fill(diag_mask.unsqueeze(0), 0.0)
    offdiag_mass = (offdiag.sum(dim=(-2, -1)) / n_heads).mean().item()

    return {
        "kromhc/H_row_entropy_mean": row_entropy,
        "kromhc/H_offdiag_mass_mean": offdiag_mass,
    }
```

**3b. Update `evaluate()`** — change `logits = model(inputs)` to `logits, _ = model(inputs)`:

```python
@torch.no_grad()
def evaluate(model: CausalLM, loader, device: torch.device) -> tuple[float, float]:
    """Returns (val_loss, val_ppl) in nats/token."""
    model.eval()
    loss_sum, token_count = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        inputs, targets = batch[:, :-1], batch[:, 1:]
        n_tokens = inputs.numel()
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
            logits, _ = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="mean",
            )
        loss_sum    += loss.item() * n_tokens
        token_count += n_tokens
    val_loss = loss_sum / token_count
    return val_loss, math.exp(val_loss)
```

**3c. Update the training loop** — change `logits = model(inputs)` to `logits, _ = model(inputs)`:

```python
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
                logits, _ = model(inputs)
                loss   = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    reduction="mean",
                )
```

**3d. Collect H stats and add to metrics row** — add after the `val_loss, val_ppl = evaluate(...)` line:

```python
        kromhc_stats: dict = {}
        if cfg.use_kromhc:
            # Use a single val batch for stats (no full-epoch overhead)
            stats_batch = next(iter(val_loader))
            kromhc_stats = collect_kromhc_stats(model, stats_batch, device)
            model.train()
```

And update the `row` dict to include them:

```python
        row: dict = {
            "epoch":                epoch,
            "train_loss":           train_loss,
            "train_ppl":            train_ppl,
            "val_loss":             val_loss,
            "val_ppl":              val_ppl,
            "epoch_time_s":         epoch_time,
            "effective_batch_size": cfg.batch_size * cfg.grad_accum_steps,
            **kromhc_stats,
        }
```

**3e. Update `get_experiment_dir`** — add a tag for KromHC experiments:

```python
    if cfg.use_kromhc:
        norm_tags += "_kromhc"
```

(Add after the existing `if cfg.activation_norm:` block.)

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /home/harikrishnan-c/projects/machine-notes
python -m pytest rbf_ffn/tests/test_train.py -v
```

Expected: all tests PASS (including the two new KromHC tests).

- [ ] **Step 5: Commit**

```bash
cd /home/harikrishnan-c/projects/machine-notes
git add rbf_ffn/train.py rbf_ffn/tests/test_train.py
git commit -m "feat(train): handle (logits, hs) return, log KromHC H stats per epoch"
```

---

## Task 6: Add KromHC config files

**Files:**
- Create: `rbf_ffn/configs/baseline_kromhc.yaml`
- Create: `rbf_ffn/configs/baseline_qk_norm_kromhc.yaml`
- Create: `rbf_ffn/configs/pfd_rationalglu_qk_norm_weight_norm_kromhc.yaml`

- [ ] **Step 1: Create `rbf_ffn/configs/baseline_kromhc.yaml`**

```yaml
# Llama SwiGLU baseline with KromHC head mixing.
# Direct comparison to kromhc_transformer/configs/baseline.yaml.
model_type: baseline
d_model: 256
n_heads: 8
n_layers: 6
ffn_hidden: 688
dropout: 0.1
qk_norm: false
vocab_size: 50257
seq_len: 512
use_kromhc: true
seed: 42
n_epochs: 10
batch_size: 32
muon_lr: 0.02
adamw_lr: 0.0003
adamw_wd: 0.1
warmup_ratio: 0.02
grad_clip: 1.0
```

- [ ] **Step 2: Create `rbf_ffn/configs/baseline_qk_norm_kromhc.yaml`**

```yaml
# Llama SwiGLU baseline with QK norm and KromHC head mixing.
# Matches default settings in kromhc_transformer (qk_norm=true by default there).
model_type: baseline
d_model: 256
n_heads: 8
n_layers: 6
ffn_hidden: 688
dropout: 0.1
qk_norm: true
vocab_size: 50257
seq_len: 512
use_kromhc: true
seed: 42
n_epochs: 10
batch_size: 32
muon_lr: 0.02
adamw_lr: 0.0003
adamw_wd: 0.1
warmup_ratio: 0.02
grad_clip: 1.0
```

- [ ] **Step 3: Create `rbf_ffn/configs/pfd_rationalglu_qk_norm_weight_norm_kromhc.yaml`**

```yaml
# Best FFN variant (PFDRationalGLU + QK norm + weight norm) combined with KromHC head mixing.
model_type: pfd_rationalglu
d_model: 256
n_heads: 8
n_layers: 6
ffn_hidden: 688
pfd_n: 4
dropout: 0.1
qk_norm: true
vocab_size: 50257
seq_len: 512
linear_weight_norm: true
linear_weight_norm_value: 2.0
linear_weight_norm_max_only: false
use_kromhc: true
seed: 42
n_epochs: 10
batch_size: 32
muon_lr: 0.02
adamw_lr: 0.0003
adamw_wd: 0.1
warmup_ratio: 0.02
grad_clip: 1.0
```

- [ ] **Step 4: Smoke-test configs load cleanly**

```bash
cd /home/harikrishnan-c/projects/machine-notes
python -c "
from rbf_ffn.config import load_config
for f in ['baseline_kromhc', 'baseline_qk_norm_kromhc', 'pfd_rationalglu_qk_norm_weight_norm_kromhc']:
    cfg = load_config(f'rbf_ffn/configs/{f}.yaml')
    assert cfg.use_kromhc is True, f'{f}: use_kromhc not True'
    print(f'{f}: OK (model_type={cfg.model_type}, use_kromhc={cfg.use_kromhc})')
"
```

Expected:
```
baseline_kromhc: OK (model_type=baseline, use_kromhc=True)
baseline_qk_norm_kromhc: OK (model_type=baseline, use_kromhc=True)
pfd_rationalglu_qk_norm_weight_norm_kromhc: OK (model_type=pfd_rationalglu, use_kromhc=True)
```

- [ ] **Step 5: Commit**

```bash
cd /home/harikrishnan-c/projects/machine-notes
git add rbf_ffn/configs/baseline_kromhc.yaml \
        rbf_ffn/configs/baseline_qk_norm_kromhc.yaml \
        rbf_ffn/configs/pfd_rationalglu_qk_norm_weight_norm_kromhc.yaml
git commit -m "feat(configs): add three KromHC experiment configs"
```

---

## Task 7: Archive `kromhc_transformer/` and update `OVERVIEW.md`

**Files:**
- `git mv kromhc_transformer/ archive/kromhc_transformer/`
- Modify: `rbf_ffn/OVERVIEW.md`

- [ ] **Step 1: Create archive directory and move**

```bash
cd /home/harikrishnan-c/projects/machine-notes
mkdir -p archive
git mv kromhc_transformer archive/kromhc_transformer
```

- [ ] **Step 2: Confirm move**

```bash
ls /home/harikrishnan-c/projects/machine-notes/archive/
```

Expected: `kromhc_transformer`

- [ ] **Step 3: Confirm full test suite still passes**

```bash
cd /home/harikrishnan-c/projects/machine-notes
python -m pytest rbf_ffn/tests/ -q
```

Expected: all tests PASS (no imports from `kromhc_transformer`).

- [ ] **Step 4: Add consolidation note to `OVERVIEW.md`**

Add the following section at the top of `rbf_ffn/OVERVIEW.md`, immediately after the header lines (after the first blank line following the opening description paragraph):

```markdown
---

## Consolidation Note (2026-04-01)

`rbf_ffn/` is now the **single source of truth** for transformer experiments in this repo.
The original `kromhc_transformer/` implementation has been archived to `archive/kromhc_transformer/`.

KromHC head mixing is available in `rbf_ffn/` via `use_kromhc: true` in any config.
It wraps any `model_type` with a `KromHCWrapper` — see `models/head_mixer.py` and
`models/transformer_block.py`. Configs: `baseline_kromhc.yaml`, `baseline_qk_norm_kromhc.yaml`,
`pfd_rationalglu_qk_norm_weight_norm_kromhc.yaml`.

---
```

- [ ] **Step 5: Commit everything**

```bash
cd /home/harikrishnan-c/projects/machine-notes
git add archive/ rbf_ffn/OVERVIEW.md
git commit -m "chore: archive kromhc_transformer, update OVERVIEW with consolidation note"
```

---

## Self-Review

**Spec coverage:**
- ✅ head_mixer.py ported → Task 1
- ✅ KromHCWrapper (wrapper pattern) → Task 2
- ✅ use_kromhc / kromhc_mixer_hidden config fields → Task 3
- ✅ model.py wraps blocks + returns (logits, hs) → Task 4
- ✅ H logging (row entropy, off-diagonal mass) per epoch → Task 5
- ✅ New config files → Task 6
- ✅ Archive kromhc_transformer/ → Task 7
- ✅ OVERVIEW.md note → Task 7

**Placeholder scan:** No TBDs or vague steps. Every code step shows complete code.

**Type consistency:**
- `KromHCWrapper` defined in Task 2, imported in Task 4 model.py ✅
- `collect_kromhc_stats` signature matches its call site in train loop ✅
- `H_4d` in KromHCWrapper is `(B, N, n_heads, n_heads)` — matches test assertions in Tasks 2 and 4 ✅
- `hs` is `list[Tensor]` throughout — consistent between model.py, train.py, and test_model.py ✅
- Config field `use_kromhc` — consistent across config.py, model.py, train.py ✅
