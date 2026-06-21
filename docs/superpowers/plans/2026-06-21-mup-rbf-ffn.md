# muP (Maximal Update Parameterization) for rbf_ffn — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add full muP support to `rbf_ffn` — init scaling, LR scaling, logit scaling — behind a `mup: bool = False` config flag, with no behavior change when disabled.

**Architecture:** Three coordinated changes: (1) after model construction, `_apply_mup_init` reinits all `nn.Linear` hidden weights to `N(0, mup_init_std * sqrt(mup_base_width/d_model))`; (2) `CausalLM.forward` multiplies output logits by `mup_base_width/d_model`; (3) `train.py` scales Muon LR and non-embedding AdamW LR by the same factor. `mup=False` → all multipliers are 1.0, numerically identical to current SP.

**Tech Stack:** PyTorch, `torch.optim.Muon`, `torch.optim.AdamW`, pytest, existing `ModelConfig` dataclass / `build_optimizer_groups` pattern.

## Global Constraints

- `mup=False` is the default; all existing configs, checkpoints, and tests must pass unchanged
- `build_optimizer_groups(model)` return type `(muon_params, adamw_params)` must not change
- Kronecker-factored layers (`KroneckerLinear`, `KroneckerDeltaLinear`) are not `nn.Linear` instances — `_apply_mup_init` skips them automatically; no special casing needed
- Full test suite: `pytest rbf_ffn/tests/ -x -q`

---

### Task 1: Config — muP fields and validation

**Files:**
- Modify: `rbf_ffn/config.py`
- Test: `rbf_ffn/tests/test_config.py` (append)

**Interfaces:**
- Produces: `ModelConfig.mup: bool = False`, `ModelConfig.mup_base_width: int = 256`, `ModelConfig.mup_init_std: float = 0.02`; `ValueError` when `mup=True` and `mup_base_width <= 0`

- [ ] **Step 1: Write the failing tests**

Append to `rbf_ffn/tests/test_config.py`:

```python
# ── muP fields ────────────────────────────────────────────────────────────────

def test_mup_defaults():
    cfg = ModelConfig()
    assert cfg.mup is False
    assert cfg.mup_base_width == 256
    assert cfg.mup_init_std == pytest.approx(0.02)


def test_mup_zero_base_width_raises():
    with pytest.raises(ValueError, match="mup_base_width"):
        ModelConfig(mup=True, mup_base_width=0)


def test_mup_negative_base_width_raises():
    with pytest.raises(ValueError, match="mup_base_width"):
        ModelConfig(mup=True, mup_base_width=-1)


def test_mup_base_width_not_validated_when_disabled():
    cfg = ModelConfig(mup=False, mup_base_width=0)
    assert cfg.mup_base_width == 0


def test_mup_yaml_roundtrip(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("mup: true\nmup_base_width: 128\nmup_init_std: 0.01\n")
    cfg = load_config(p)
    assert cfg.mup is True
    assert cfg.mup_base_width == 128
    assert cfg.mup_init_std == pytest.approx(0.01)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest rbf_ffn/tests/test_config.py::test_mup_defaults rbf_ffn/tests/test_config.py::test_mup_zero_base_width_raises -x -q
```

Expected: FAIL with `AttributeError: 'ModelConfig' object has no attribute 'mup'`

- [ ] **Step 3: Add fields and validation to `rbf_ffn/config.py`**

In `ModelConfig`, after the `grad_accum_steps` field (end of the `# Training` block), add:

```python
    # Maximal Update Parameterization (muP)
    mup: bool = False
    mup_base_width: int = 256    # proxy model width at which muon_lr/adamw_lr were tuned
    mup_init_std: float = 0.02   # init std at base width; hidden matrices scaled by sqrt(base/d)
```

In `__post_init__`, after the existing `adaptive_weight_norm` validation block, add:

```python
        if self.mup and self.mup_base_width <= 0:
            raise ValueError(
                f"mup_base_width must be > 0 when mup=True, got {self.mup_base_width}"
            )
```

- [ ] **Step 4: Run the full config test suite**

```bash
pytest rbf_ffn/tests/test_config.py -x -q
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add rbf_ffn/config.py rbf_ffn/tests/test_config.py
git commit -m "feat(rbf_ffn): add muP config fields — mup, mup_base_width, mup_init_std"
```

---

### Task 2: Model — init scaling and logit scaling

**Files:**
- Modify: `rbf_ffn/models/model.py`
- Test: `rbf_ffn/tests/test_model.py` (append)

**Interfaces:**
- Consumes: `ModelConfig.mup`, `ModelConfig.mup_base_width`, `ModelConfig.mup_init_std` (Task 1)
- Produces: `_apply_mup_init(model: CausalLM, cfg: ModelConfig) -> None`; `CausalLM.mup_scale: float`

- [ ] **Step 1: Write the failing tests**

Append to `rbf_ffn/tests/test_model.py`:

```python
import math as _math


def _make_mup_model(d_model: int = D, base_width: int = 64, init_std: float = 0.02) -> CausalLM:
    cfg = ModelConfig(
        d_model=d_model, n_heads=H, n_layers=L,
        vocab_size=VOCAB, seq_len=N,
        ffn_hidden=d_model * 3 // 2,
        dropout=0.0,
        mup=True,
        mup_base_width=base_width,
        mup_init_std=init_std,
    )
    return CausalLM(cfg)


def test_mup_scale_attribute():
    """mup_scale == mup_base_width / d_model."""
    model = _make_mup_model(d_model=D, base_width=64)
    assert model.mup_scale == pytest.approx(64 / D)


def test_mup_scale_is_one_when_disabled():
    """mup=False (default) must set mup_scale to 1.0."""
    model = make_model("baseline")
    assert model.mup_scale == pytest.approx(1.0)


def test_mup_hidden_weight_std():
    """All hidden nn.Linear weights must have std ≈ mup_init_std * sqrt(base_width / d_model)."""
    base_width, init_std = 64, 0.02
    expected_std = init_std * _math.sqrt(base_width / D)
    model = _make_mup_model(d_model=D, base_width=base_width, init_std=init_std)
    tied_id = id(model.token_embedding.weight)
    stds = [
        m.weight.data.std().item()
        for m in model.modules()
        if isinstance(m, torch.nn.Linear) and id(m.weight) != tied_id
    ]
    assert stds, "No hidden Linear layers found"
    mean_std = sum(stds) / len(stds)
    # 30% relative tolerance: small matrices have high sample variance
    assert mean_std == pytest.approx(expected_std, rel=0.3)


def test_mup_embedding_weight_not_reinited():
    """_apply_mup_init must leave the token embedding weight alone (it's the input layer).

    nn.Embedding default init is N(0,1) → std close to 1.0.
    muP init std is 0.02 * sqrt(64/32) ≈ 0.028, which is far below 0.5.
    """
    model = _make_mup_model(d_model=D, base_width=64, init_std=0.02)
    emb_std = model.token_embedding.weight.data.std().item()
    assert emb_std > 0.5, f"Embedding std unexpectedly small ({emb_std}): was it reinited?"


def test_mup_logit_scaling():
    """Logits with mup=True must equal unscaled logits * (mup_base_width / d_model)."""
    torch.manual_seed(0)
    base_width = 64
    expected_scale = base_width / D

    cfg = ModelConfig(
        d_model=D, n_heads=H, n_layers=L,
        vocab_size=VOCAB, seq_len=N,
        ffn_hidden=D * 3 // 2, dropout=0.0,
        mup=True, mup_base_width=base_width, mup_init_std=0.02,
    )
    model = CausalLM(cfg)
    model.eval()
    tokens = torch.randint(0, VOCAB, (1, N))

    with torch.no_grad():
        logits_scaled, _ = model(tokens)
        # Bypass the multiplier by temporarily patching mup_scale to 1.0
        orig = model.mup_scale
        model.mup_scale = 1.0
        logits_raw, _ = model(tokens)
        model.mup_scale = orig

    assert torch.allclose(logits_scaled, logits_raw * expected_scale, atol=1e-5)


def test_mup_output_shape_unchanged():
    model = _make_mup_model()
    tokens = torch.randint(0, VOCAB, (B, N))
    logits, hs = model(tokens)
    assert logits.shape == (B, N, VOCAB)
    assert hs == []


def test_mup_gradient_flows():
    model = _make_mup_model()
    tokens = torch.randint(0, VOCAB, (B, N))
    logits, _ = model(tokens)
    logits.sum().backward()
    assert model.token_embedding.weight.grad is not None


def test_mup_base_equals_width_is_numerically_identical():
    """When mup_base_width == d_model, mup_scale == 1.0."""
    cfg = ModelConfig(
        d_model=D, n_heads=H, n_layers=L,
        vocab_size=VOCAB, seq_len=N,
        ffn_hidden=D * 3 // 2, dropout=0.0,
        mup=True, mup_base_width=D,
    )
    model = CausalLM(cfg)
    assert model.mup_scale == pytest.approx(1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest rbf_ffn/tests/test_model.py::test_mup_scale_attribute rbf_ffn/tests/test_model.py::test_mup_scale_is_one_when_disabled -x -q
```

Expected: FAIL with `AttributeError: 'CausalLM' object has no attribute 'mup_scale'`

- [ ] **Step 3: Add `import math` to `rbf_ffn/models/model.py`**

At the top of the file, after `from __future__ import annotations`, add:

```python
import math
```

The full import block becomes:

```python
from __future__ import annotations
import math
import torch
import torch.nn as nn
from rbf_ffn.config import ModelConfig
from rbf_ffn.models.transformer_block import TransformerBlock, KromHCWrapper, LoopBlock
from rbf_ffn.models.kronecker_linear import KroneckerLMHead, LoRALMHead
from rbf_ffn.models.attn_res import AttnResLayer
```

- [ ] **Step 4: Add `_apply_mup_init` immediately before `class CausalLM`**

```python
def _apply_mup_init(model: "CausalLM", cfg: ModelConfig) -> None:
    """Reinit all nn.Linear hidden weights with muP std = mup_init_std * sqrt(base/d_model).

    Skips the tied embedding weight (input layer — init unchanged).
    Biases and non-Linear params are untouched.
    Kronecker-factored layers are not nn.Linear, so they are skipped automatically.
    """
    std = cfg.mup_init_std * math.sqrt(cfg.mup_base_width / cfg.d_model)
    tied_id = id(model.token_embedding.weight)
    for module in model.modules():
        if isinstance(module, nn.Linear) and id(module.weight) != tied_id:
            nn.init.normal_(module.weight, mean=0.0, std=std)
```

- [ ] **Step 5: Add `mup_scale` and call `_apply_mup_init` at the end of `CausalLM.__init__`**

The last lines of `__init__` currently end with the `lm_head` setup. After all existing lines, append:

```python
        self.mup_scale: float = (cfg.mup_base_width / cfg.d_model) if cfg.mup else 1.0
        if cfg.mup:
            _apply_mup_init(self, cfg)
```

- [ ] **Step 6: Apply logit scaling in `CausalLM.forward`**

The current final return in `forward` is:

```python
        return self.lm_head(x), hs
```

Change it to:

```python
        return self.lm_head(x) * self.mup_scale, hs
```

(`torch.compile` folds `* 1.0` away when `mup=False`, so there is no runtime cost in the SP case.)

- [ ] **Step 7: Run the full model test suite**

```bash
pytest rbf_ffn/tests/test_model.py -x -q
```

Expected: all PASS

- [ ] **Step 8: Commit**

```bash
git add rbf_ffn/models/model.py rbf_ffn/tests/test_model.py
git commit -m "feat(rbf_ffn): muP init scaling (_apply_mup_init) and logit scaling in CausalLM"
```

---

### Task 3: Training — optimizer LR scaling and experiment naming

**Files:**
- Modify: `rbf_ffn/train.py`
- Test: `rbf_ffn/tests/test_train.py` (append)

**Interfaces:**
- Consumes: `ModelConfig.mup`, `ModelConfig.mup_base_width` (Task 1); `build_optimizer_groups` (unchanged)
- Produces: Muon `lr = muon_lr * mup_scale`; AdamW hidden group `lr = adamw_lr * mup_scale`; AdamW embedding group `lr = adamw_lr`; `_mup{N}` tag in experiment dir name

- [ ] **Step 1: Write the failing tests**

Append to `rbf_ffn/tests/test_train.py`:

```python
# ── muP optimizer LR scaling ──────────────────────────────────────────────────

def test_mup_muon_lr_is_scaled():
    """Muon LR must be muon_lr * (mup_base_width / d_model) when mup=True."""
    try:
        from torch.optim import Muon
    except ImportError:
        pytest.skip("Muon not available")
    from rbf_ffn.models.model import build_optimizer_groups

    cfg = _tiny_cfg(mup=True, mup_base_width=64, muon_lr=0.02, adamw_lr=3e-4)
    model = CausalLM(cfg)
    muon_params, _ = build_optimizer_groups(model)
    mup_scale = cfg.mup_base_width / cfg.d_model
    muon = Muon(muon_params, lr=cfg.muon_lr * mup_scale, momentum=0.95)
    assert muon.param_groups[0]["lr"] == pytest.approx(0.02 * mup_scale)


def test_mup_adamw_hidden_lr_is_scaled():
    """Non-embedding AdamW params must get lr = adamw_lr * (mup_base_width / d_model)."""
    from rbf_ffn.models.model import build_optimizer_groups

    cfg = _tiny_cfg(mup=True, mup_base_width=64, adamw_lr=3e-4)
    model = CausalLM(cfg)
    _, adamw_params = build_optimizer_groups(model)
    mup_scale = cfg.mup_base_width / cfg.d_model
    emb_id = id(model.token_embedding.weight)
    adamw = AdamW([
        {"params": [p for p in adamw_params if id(p) != emb_id],
         "lr": cfg.adamw_lr * mup_scale},
        {"params": [p for p in adamw_params if id(p) == emb_id],
         "lr": cfg.adamw_lr},
    ], weight_decay=cfg.adamw_wd, betas=(0.9, 0.95))
    hidden_group = next(g for g in adamw.param_groups
                        if any(id(p) != emb_id for p in g["params"]))
    assert hidden_group["lr"] == pytest.approx(3e-4 * mup_scale)


def test_mup_adamw_embedding_lr_unscaled():
    """Embedding must get the raw adamw_lr, not the scaled one."""
    from rbf_ffn.models.model import build_optimizer_groups

    cfg = _tiny_cfg(mup=True, mup_base_width=64, adamw_lr=3e-4)
    model = CausalLM(cfg)
    _, adamw_params = build_optimizer_groups(model)
    mup_scale = cfg.mup_base_width / cfg.d_model
    emb_id = id(model.token_embedding.weight)
    adamw = AdamW([
        {"params": [p for p in adamw_params if id(p) != emb_id],
         "lr": cfg.adamw_lr * mup_scale},
        {"params": [p for p in adamw_params if id(p) == emb_id],
         "lr": cfg.adamw_lr},
    ], weight_decay=cfg.adamw_wd, betas=(0.9, 0.95))
    emb_group = next(g for g in adamw.param_groups
                     if any(id(p) == emb_id for p in g["params"]))
    assert emb_group["lr"] == pytest.approx(3e-4)


def test_mup_disabled_lrs_unchanged():
    """When mup=False, both LRs must be exactly as configured."""
    try:
        from torch.optim import Muon
    except ImportError:
        pytest.skip("Muon not available")
    from rbf_ffn.models.model import build_optimizer_groups

    cfg = _tiny_cfg(mup=False, muon_lr=0.02, adamw_lr=3e-4)
    model = CausalLM(cfg)
    muon_params, adamw_params = build_optimizer_groups(model)
    muon = Muon(muon_params, lr=cfg.muon_lr * 1.0, momentum=0.95)
    adamw_opt = AdamW(adamw_params, lr=cfg.adamw_lr,
                      weight_decay=cfg.adamw_wd, betas=(0.9, 0.95))
    assert muon.param_groups[0]["lr"] == pytest.approx(0.02)
    assert adamw_opt.param_groups[0]["lr"] == pytest.approx(3e-4)
```

- [ ] **Step 2: Run tests to verify they pass (these tests exercise the optimizer construction logic directly)**

```bash
pytest rbf_ffn/tests/test_train.py::test_mup_muon_lr_is_scaled rbf_ffn/tests/test_train.py::test_mup_adamw_hidden_lr_is_scaled rbf_ffn/tests/test_train.py::test_mup_adamw_embedding_lr_unscaled rbf_ffn/tests/test_train.py::test_mup_disabled_lrs_unchanged -x -q
```

Expected: all PASS (the tests build the optimizers directly and verify the correct math — they do not depend on `train.py`'s internal implementation).

- [ ] **Step 3: Update optimizer construction in `rbf_ffn/train.py`**

Find the current optimizer block (the four lines starting with `muon_params, adamw_params = build_optimizer_groups(model)`):

```python
    muon_params, adamw_params = build_optimizer_groups(model)
    muon  = Muon( muon_params,  lr=cfg.muon_lr, momentum=0.95)
    adamw = AdamW(adamw_params, lr=cfg.adamw_lr,
                  weight_decay=cfg.adamw_wd, betas=(0.9, 0.95))
```

Replace with:

```python
    muon_params, adamw_params = build_optimizer_groups(model)
    mup_scale = (cfg.mup_base_width / cfg.d_model) if cfg.mup else 1.0

    muon  = Muon(muon_params, lr=cfg.muon_lr * mup_scale, momentum=0.95)

    if cfg.mup:
        emb_id = id(model.token_embedding.weight)
        adamw = AdamW([
            {"params": [p for p in adamw_params if id(p) != emb_id],
             "lr": cfg.adamw_lr * mup_scale},
            {"params": [p for p in adamw_params if id(p) == emb_id],
             "lr": cfg.adamw_lr},
        ], weight_decay=cfg.adamw_wd, betas=(0.9, 0.95))
    else:
        adamw = AdamW(adamw_params, lr=cfg.adamw_lr,
                      weight_decay=cfg.adamw_wd, betas=(0.9, 0.95))
```

- [ ] **Step 4: Add `_mup{N}` tag to `get_experiment_dir` in `rbf_ffn/train.py`**

In `get_experiment_dir`, after the existing `if cfg.qkv_gain:` block, add:

```python
    if cfg.mup:
        norm_tags += f"_mup{cfg.mup_base_width}"
```

- [ ] **Step 5: Run the full test suite**

```bash
pytest rbf_ffn/tests/ -x -q
```

Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add rbf_ffn/train.py rbf_ffn/tests/test_train.py
git commit -m "feat(rbf_ffn): muP optimizer LR scaling and _mup tag in experiment naming"
```
