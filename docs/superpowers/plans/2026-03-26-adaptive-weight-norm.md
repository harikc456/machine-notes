# Adaptive Weight Norm Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add depth-based adaptive weight normalization to `rbf_ffn` and `grokking`, where early layers get a higher target norm than late layers, with a phase-aware derivative correction that surgically tightens late layers when the generalization gap is actively growing.

**Architecture:** Two-component formula: `target_norm(l, t) = max(1.0, static(l) - correction(l, t))` where `static(l)` linearly interpolates from `norm_early` (layer 0) to `norm_late` (layer L-1), and `correction(l, t)` applies a `tanh`-bounded per-layer correction proportional to `Δema_log_gap(t)` (derivative of the smoothed generalization gap). Floor of 1.0 is always enforced to prevent flat-curvature dead zones. The existing `linear_weight_norm` path is unchanged for backward compatibility.

**Tech Stack:** PyTorch, Python dataclasses, pytest, YAML configs.

**Spec:** `docs/superpowers/specs/2026-03-26-adaptive-weight-norm-design.md`

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Modify | `rbf_ffn/config.py` | Add 6 new fields + `__post_init__` validation |
| Modify | `rbf_ffn/train.py` | Add `apply_adaptive_weight_norm`, EMA state, loop hooks, dir tag |
| Modify | `rbf_ffn/tests/test_config.py` | Config field and validation tests |
| Modify | `rbf_ffn/tests/test_train.py` | Function correctness + integration smoke tests |
| Create | `rbf_ffn/configs/baseline_adaptive_weight_norm.yaml` | New experiment config |
| Modify | `grokking/config.py` | Add 6 new fields + validation |
| Modify | `grokking/train.py` | Add `apply_adaptive_weight_norm`, EMA state, loop hooks |
| Modify | `grokking/tests/test_config.py` | Config field and validation tests |
| Modify | `grokking/tests/test_train.py` | Function correctness + integration smoke tests |

---

## Task 1: rbf_ffn config — add adaptive norm fields and validation

**Files:**
- Modify: `rbf_ffn/config.py`
- Modify: `rbf_ffn/tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

Append to `rbf_ffn/tests/test_config.py`:

```python
# ── Adaptive weight norm fields ───────────────────────────────────────────────

def test_adaptive_weight_norm_defaults():
    cfg = RBFFFNConfig()
    assert cfg.adaptive_weight_norm is False
    assert cfg.adaptive_norm_early == pytest.approx(2.5)
    assert cfg.adaptive_norm_late  == pytest.approx(1.2)
    assert cfg.adaptive_norm_gamma == pytest.approx(0.3)
    assert cfg.adaptive_norm_beta  == pytest.approx(5.0)
    assert cfg.adaptive_norm_alpha == pytest.approx(0.9)


def test_adaptive_norm_late_below_one_raises():
    with pytest.raises(ValueError, match="adaptive_norm_late"):
        RBFFFNConfig(adaptive_weight_norm=True, adaptive_norm_late=0.9)


def test_adaptive_norm_early_not_greater_than_late_raises():
    with pytest.raises(ValueError, match="adaptive_norm_early"):
        RBFFFNConfig(adaptive_weight_norm=True, adaptive_norm_early=1.2, adaptive_norm_late=1.2)


def test_adaptive_norm_validation_only_when_enabled():
    """Validation is skipped when adaptive_weight_norm=False (default)."""
    cfg = RBFFFNConfig(adaptive_weight_norm=False, adaptive_norm_late=0.5)
    assert cfg.adaptive_norm_late == pytest.approx(0.5)


def test_adaptive_norm_yaml_roundtrip(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "adaptive_weight_norm: true\n"
        "adaptive_norm_early: 3.0\n"
        "adaptive_norm_late: 1.5\n"
    )
    cfg = load_config(p)
    assert cfg.adaptive_weight_norm is True
    assert cfg.adaptive_norm_early == pytest.approx(3.0)
    assert cfg.adaptive_norm_late  == pytest.approx(1.5)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/harikrishnan-c/projects/machine-notes
python -m pytest rbf_ffn/tests/test_config.py::test_adaptive_weight_norm_defaults \
    rbf_ffn/tests/test_config.py::test_adaptive_norm_late_below_one_raises \
    rbf_ffn/tests/test_config.py::test_adaptive_norm_early_not_greater_than_late_raises \
    rbf_ffn/tests/test_config.py::test_adaptive_norm_validation_only_when_enabled \
    rbf_ffn/tests/test_config.py::test_adaptive_norm_yaml_roundtrip \
    -v
```

Expected: all 5 FAIL (fields don't exist yet).

- [ ] **Step 3: Add fields and `__post_init__` to `rbf_ffn/config.py`**

After the `activation_norm` block (around line 33), add:

```python
    # Adaptive weight normalization (depth-based)
    adaptive_weight_norm: bool = False
    adaptive_norm_early: float = 2.5   # target norm at layer 0
    adaptive_norm_late: float = 1.2    # target norm at layer L-1 (must be >= 1.0)
    adaptive_norm_gamma: float = 0.3   # max phase correction magnitude
    adaptive_norm_beta: float = 5.0    # tanh sensitivity to gap derivative
    adaptive_norm_alpha: float = 0.9   # EMA smoothing factor for log-gap
```

After the `grad_accum_steps` field, add the `__post_init__` method:

```python
    def __post_init__(self) -> None:
        if self.adaptive_weight_norm:
            if self.adaptive_norm_late < 1.0:
                raise ValueError(
                    f"adaptive_norm_late must be >= 1.0, got {self.adaptive_norm_late}"
                )
            if self.adaptive_norm_early <= self.adaptive_norm_late:
                raise ValueError(
                    f"adaptive_norm_early ({self.adaptive_norm_early}) must be > "
                    f"adaptive_norm_late ({self.adaptive_norm_late})"
                )
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest rbf_ffn/tests/test_config.py -v
```

Expected: all pass. The existing YAML tests (`test_existing_yamls_load_without_grad_accum_steps`) must still pass — existing configs don't have the new fields and that's fine since they default to `False`.

- [ ] **Step 5: Commit**

```bash
git add rbf_ffn/config.py rbf_ffn/tests/test_config.py
git commit -m "feat(rbf_ffn): add adaptive weight norm config fields and validation"
```

---

## Task 2: rbf_ffn — add `apply_adaptive_weight_norm` function

**Files:**
- Modify: `rbf_ffn/train.py`
- Modify: `rbf_ffn/tests/test_train.py`

- [ ] **Step 1: Write the failing tests**

At the top of `rbf_ffn/tests/test_train.py`, add these imports after the existing ones:

```python
import torch.nn as nn
from rbf_ffn.models.model import CausalLM
from rbf_ffn.train import apply_adaptive_weight_norm
```

Append these tests to `rbf_ffn/tests/test_train.py`:

```python
# ── apply_adaptive_weight_norm ────────────────────────────────────────────────

def _adaptive_cfg(n_layers: int = 4, **kwargs) -> RBFFFNConfig:
    defaults = dict(
        model_type="baseline",
        d_model=32,
        n_heads=2,
        n_layers=n_layers,
        ffn_hidden=64,
        seq_len=8,
        vocab_size=50,
        seed=0,
        adaptive_weight_norm=True,
        adaptive_norm_early=2.5,
        adaptive_norm_late=1.2,
        adaptive_norm_gamma=0.3,
        adaptive_norm_beta=5.0,
        adaptive_norm_alpha=0.9,
    )
    defaults.update(kwargs)
    return RBFFFNConfig(**defaults)


def test_adaptive_weight_norm_zero_delta_matches_static_schedule():
    """With delta_log_gap=0.0 the correction term is zero, so row norms equal
    the static depth schedule exactly."""
    cfg = _adaptive_cfg(n_layers=4)
    model = CausalLM(cfg)
    apply_adaptive_weight_norm(model, cfg, delta_log_gap=0.0)

    L = len(model.blocks)
    for layer_idx, block in enumerate(model.blocks):
        frac = layer_idx / max(L - 1, 1)
        expected = cfg.adaptive_norm_late + (cfg.adaptive_norm_early - cfg.adaptive_norm_late) * (1.0 - frac)
        for module in block.modules():
            if isinstance(module, nn.Linear):
                row_norms = module.weight.data.norm(dim=1)
                assert torch.allclose(
                    row_norms,
                    torch.full_like(row_norms, expected),
                    atol=1e-5,
                ), f"layer {layer_idx}: expected {expected:.4f}, got mean {row_norms.mean():.4f}"


def test_adaptive_weight_norm_floor_never_below_one():
    """Row norms never fall below 1.0 for any delta_log_gap, including
    extreme values that would push correction > static."""
    cfg = _adaptive_cfg(n_layers=4)
    model = CausalLM(cfg)

    for delta in [100.0, -100.0, 0.0, 0.5, -0.5]:
        apply_adaptive_weight_norm(model, cfg, delta_log_gap=delta)
        for block in model.blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    row_norms = module.weight.data.norm(dim=1)
                    assert (row_norms >= 1.0 - 1e-5).all(), \
                        f"norm < 1.0 with delta={delta}: {row_norms.min():.4f}"


def test_adaptive_weight_norm_gamma_zero_disables_phase_correction():
    """gamma=0 makes the correction term identically zero regardless of delta,
    so any delta_log_gap produces the same result as delta=0."""
    cfg = _adaptive_cfg(n_layers=3, adaptive_norm_gamma=0.0)
    model = CausalLM(cfg)
    apply_adaptive_weight_norm(model, cfg, delta_log_gap=50.0)

    L = len(model.blocks)
    for layer_idx, block in enumerate(model.blocks):
        frac = layer_idx / max(L - 1, 1)
        expected = cfg.adaptive_norm_late + (cfg.adaptive_norm_early - cfg.adaptive_norm_late) * (1.0 - frac)
        for module in block.modules():
            if isinstance(module, nn.Linear):
                row_norms = module.weight.data.norm(dim=1)
                assert torch.allclose(
                    row_norms,
                    torch.full_like(row_norms, expected),
                    atol=1e-5,
                )


def test_adaptive_weight_norm_early_greater_than_late():
    """Layer 0 has higher row norms than layer L-1 when delta=0."""
    cfg = _adaptive_cfg(n_layers=4)
    model = CausalLM(cfg)
    apply_adaptive_weight_norm(model, cfg, delta_log_gap=0.0)

    def mean_row_norm(block):
        norms = []
        for module in block.modules():
            if isinstance(module, nn.Linear):
                norms.append(module.weight.data.norm(dim=1).mean().item())
        return sum(norms) / len(norms)

    norm_first = mean_row_norm(model.blocks[0])
    norm_last  = mean_row_norm(model.blocks[-1])
    assert norm_first > norm_last, \
        f"expected layer 0 norm ({norm_first:.4f}) > layer L-1 norm ({norm_last:.4f})"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest rbf_ffn/tests/test_train.py::test_adaptive_weight_norm_zero_delta_matches_static_schedule \
    rbf_ffn/tests/test_train.py::test_adaptive_weight_norm_floor_never_below_one \
    rbf_ffn/tests/test_train.py::test_adaptive_weight_norm_gamma_zero_disables_phase_correction \
    rbf_ffn/tests/test_train.py::test_adaptive_weight_norm_early_greater_than_late \
    -v
```

Expected: all 4 FAIL with `ImportError: cannot import name 'apply_adaptive_weight_norm'`.

- [ ] **Step 3: Add `apply_adaptive_weight_norm` to `rbf_ffn/train.py`**

Insert this function immediately after `apply_linear_weight_norm` (after line 87):

```python
@torch.no_grad()
def apply_adaptive_weight_norm(
    model: CausalLM,
    cfg: RBFFFNConfig,
    delta_log_gap: float,
) -> None:
    """Apply per-layer adaptive weight norm.

    Target norm decreases linearly from cfg.adaptive_norm_early (layer 0)
    to cfg.adaptive_norm_late (layer L-1).  A phase-aware derivative correction
    proportional to tanh(beta * delta_log_gap) is applied most strongly to late
    layers (correction weight = l/(L-1)).  A hard floor of 1.0 is enforced on
    every target to prevent flat-curvature dead zones.

    Iterates model.blocks only — lm_head and embeddings are excluded by design.
    """
    L = len(model.blocks)
    for layer_idx, block in enumerate(model.blocks):
        frac = layer_idx / max(L - 1, 1)
        static = cfg.adaptive_norm_late + (cfg.adaptive_norm_early - cfg.adaptive_norm_late) * (1.0 - frac)
        correction = cfg.adaptive_norm_gamma * frac * math.tanh(cfg.adaptive_norm_beta * delta_log_gap)
        target = max(1.0, static - correction)

        for module in block.modules():
            if isinstance(module, nn.Linear):
                norms = module.weight.data.norm(dim=1, keepdim=True).clamp(min=1e-8)
                module.weight.data.mul_(target / norms)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest rbf_ffn/tests/test_train.py::test_adaptive_weight_norm_zero_delta_matches_static_schedule \
    rbf_ffn/tests/test_train.py::test_adaptive_weight_norm_floor_never_below_one \
    rbf_ffn/tests/test_train.py::test_adaptive_weight_norm_gamma_zero_disables_phase_correction \
    rbf_ffn/tests/test_train.py::test_adaptive_weight_norm_early_greater_than_late \
    -v
```

Expected: all 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add rbf_ffn/train.py rbf_ffn/tests/test_train.py
git commit -m "feat(rbf_ffn): add apply_adaptive_weight_norm with depth schedule and phase correction"
```

---

## Task 3: rbf_ffn — wire into training loop and experiment dir

**Files:**
- Modify: `rbf_ffn/train.py`
- Modify: `rbf_ffn/tests/test_train.py`

- [ ] **Step 1: Write the failing smoke test**

Append to `rbf_ffn/tests/test_train.py`:

```python
def test_train_adaptive_weight_norm_smoke(tmp_path):
    """Training completes without error with adaptive_weight_norm=True,
    and the experiment dir name contains '_adpwnorm'."""
    cfg = _tiny_cfg(
        n_layers=2,
        n_epochs=1,
        adaptive_weight_norm=True,
        adaptive_norm_early=2.5,
        adaptive_norm_late=1.2,
        adaptive_norm_gamma=0.3,
        adaptive_norm_beta=5.0,
        adaptive_norm_alpha=0.9,
    )
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("")

    with patch("rbf_ffn.train.get_dataloaders", return_value=_fake_loaders(cfg)):
        with patch("torch.optim.Muon", _MuonStub):
            exp_dir = train(cfg, config_path=config_path)

    assert "_adpwnorm" in exp_dir.name
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
python -m pytest rbf_ffn/tests/test_train.py::test_train_adaptive_weight_norm_smoke -v
```

Expected: FAIL — `_adpwnorm` not in exp_dir name (tag not added yet) or KeyError on unknown config field.

- [ ] **Step 3: Update `get_experiment_dir` to tag adaptive norm runs**

In `rbf_ffn/train.py`, in `get_experiment_dir`, after the `linear_weight_norm` check (line ~36), add:

```python
    if cfg.adaptive_weight_norm:
        norm_tags += "_adpwnorm"
```

The full updated block should read:

```python
    norm_tags = ""
    if cfg.qk_norm:
        norm_tags += "_qknorm"
    if cfg.linear_weight_norm:
        norm_tags += "_wnorm"
    if cfg.adaptive_weight_norm:
        norm_tags += "_adpwnorm"
    if cfg.activation_norm:
        norm_tags += "_actnorm"
```

- [ ] **Step 4: Initialize EMA state variables before the epoch loop**

In `rbf_ffn/train.py`, in the `train` function, after line `val_loss, val_ppl = float("inf"), float("inf")` (around line 174), add:

```python
    ema_log_gap: float = 0.0
    delta_log_gap: float = 0.0
```

- [ ] **Step 5: Apply adaptive norm in the grad-accum step block**

In `rbf_ffn/train.py`, inside the `if global_step % cfg.grad_accum_steps == 0:` block (around line 222), add the adaptive norm call after the existing `linear_weight_norm` call:

```python
                if cfg.linear_weight_norm:
                    apply_linear_weight_norm(model, cfg.linear_weight_norm_value)
                if cfg.adaptive_weight_norm:
                    apply_adaptive_weight_norm(model, cfg, delta_log_gap)
                if cfg.activation_norm:
                    apply_activation_coeff_norm(model)
```

- [ ] **Step 6: Apply adaptive norm in the epoch-end flush block**

In `rbf_ffn/train.py`, in the `if global_step % cfg.grad_accum_steps != 0:` block at the end of the epoch (around line 243), mirror the same change:

```python
            if cfg.linear_weight_norm:
                apply_linear_weight_norm(model, cfg.linear_weight_norm_value)
            if cfg.adaptive_weight_norm:
                apply_adaptive_weight_norm(model, cfg, delta_log_gap)
            if cfg.activation_norm:
                apply_activation_coeff_norm(model)
```

- [ ] **Step 7: Update EMA after val loss is computed each epoch**

In `rbf_ffn/train.py`, after `val_loss, val_ppl = evaluate(model, val_loader, device)` and `epoch_time = time.time() - t0` (around line 251), add:

```python
        if cfg.adaptive_weight_norm:
            log_gap = math.log(max(val_loss, 1e-8) / max(train_loss, 1e-8))
            new_ema = cfg.adaptive_norm_alpha * log_gap + (1.0 - cfg.adaptive_norm_alpha) * ema_log_gap
            delta_log_gap = new_ema - ema_log_gap
            ema_log_gap = new_ema
```

- [ ] **Step 8: Run all rbf_ffn tests**

```bash
python -m pytest rbf_ffn/tests/ -v
```

Expected: all pass. Critically, the existing weight norm tests and YAML-loading tests must still pass.

- [ ] **Step 9: Commit**

```bash
git add rbf_ffn/train.py rbf_ffn/tests/test_train.py
git commit -m "feat(rbf_ffn): wire adaptive weight norm into training loop with EMA phase tracking"
```

---

## Task 4: rbf_ffn — add experiment YAML config

**Files:**
- Create: `rbf_ffn/configs/baseline_adaptive_weight_norm.yaml`
- Modify: `rbf_ffn/tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Append to `rbf_ffn/tests/test_config.py`:

```python
def test_baseline_adaptive_weight_norm_yaml_loads():
    cfg = load_config(CONFIGS_DIR / "baseline_adaptive_weight_norm.yaml")
    assert cfg.adaptive_weight_norm is True
    assert cfg.adaptive_norm_early == pytest.approx(2.5)
    assert cfg.adaptive_norm_late  == pytest.approx(1.2)
    assert cfg.adaptive_norm_gamma == pytest.approx(0.3)
    assert cfg.adaptive_norm_beta  == pytest.approx(5.0)
    assert cfg.adaptive_norm_alpha == pytest.approx(0.9)
    assert cfg.model_type == "baseline"
    assert cfg.n_layers == 6
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
python -m pytest rbf_ffn/tests/test_config.py::test_baseline_adaptive_weight_norm_yaml_loads -v
```

Expected: FAIL — file does not exist.

- [ ] **Step 3: Create the YAML config**

Create `rbf_ffn/configs/baseline_adaptive_weight_norm.yaml`:

```yaml
# Llama SwiGLU baseline with adaptive depth-based weight normalisation.
# Target norm decreases linearly from adaptive_norm_early (layer 0) to
# adaptive_norm_late (layer L-1). A phase-aware derivative correction
# tightens late layers when the generalization gap is actively growing.
# Row norm floor of 1.0 is always enforced.
model_type: baseline
d_model: 256
n_heads: 8
n_layers: 6
ffn_hidden: 688
dropout: 0.1
qk_norm: true
vocab_size: 50257
seq_len: 512
adaptive_weight_norm: true
adaptive_norm_early: 2.5
adaptive_norm_late: 1.2
adaptive_norm_gamma: 0.3
adaptive_norm_beta: 5.0
adaptive_norm_alpha: 0.9
seed: 42
n_epochs: 3
batch_size: 16
muon_lr: 0.02
adamw_lr: 0.0003
adamw_wd: 0.1
warmup_ratio: 0.02
grad_clip: 1.0
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
python -m pytest rbf_ffn/tests/test_config.py::test_baseline_adaptive_weight_norm_yaml_loads -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rbf_ffn/configs/baseline_adaptive_weight_norm.yaml rbf_ffn/tests/test_config.py
git commit -m "feat(rbf_ffn): add baseline_adaptive_weight_norm.yaml experiment config"
```

---

## Task 5: grokking config — add adaptive norm fields and validation

**Files:**
- Modify: `grokking/config.py`
- Modify: `grokking/tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

Append to `grokking/tests/test_config.py`:

```python
# ── Adaptive weight norm fields ───────────────────────────────────────────────

def test_adaptive_weight_norm_defaults():
    from grokking.config import GrokConfig
    cfg = GrokConfig()
    assert cfg.adaptive_weight_norm is False
    assert cfg.adaptive_norm_early == pytest.approx(2.5)
    assert cfg.adaptive_norm_late  == pytest.approx(1.2)
    assert cfg.adaptive_norm_gamma == pytest.approx(0.3)
    assert cfg.adaptive_norm_beta  == pytest.approx(5.0)
    assert cfg.adaptive_norm_alpha == pytest.approx(0.9)


def test_adaptive_norm_late_below_one_raises():
    from grokking.config import GrokConfig
    with pytest.raises(ValueError, match="adaptive_norm_late"):
        GrokConfig(adaptive_weight_norm=True, adaptive_norm_late=0.8)


def test_adaptive_norm_early_not_greater_than_late_raises():
    from grokking.config import GrokConfig
    with pytest.raises(ValueError, match="adaptive_norm_early"):
        GrokConfig(adaptive_weight_norm=True, adaptive_norm_early=1.5, adaptive_norm_late=1.5)


def test_adaptive_norm_validation_skipped_when_disabled():
    from grokking.config import GrokConfig
    cfg = GrokConfig(adaptive_weight_norm=False, adaptive_norm_late=0.5)
    assert cfg.adaptive_norm_late == pytest.approx(0.5)


def test_adaptive_norm_yaml_roundtrip(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "adaptive_weight_norm: true\n"
        "adaptive_norm_early: 3.0\n"
        "adaptive_norm_late: 1.5\n"
    )
    cfg = load_config(path)
    assert cfg.adaptive_weight_norm is True
    assert cfg.adaptive_norm_early == pytest.approx(3.0)
    assert cfg.adaptive_norm_late  == pytest.approx(1.5)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest grokking/tests/test_config.py::test_adaptive_weight_norm_defaults \
    grokking/tests/test_config.py::test_adaptive_norm_late_below_one_raises \
    grokking/tests/test_config.py::test_adaptive_norm_early_not_greater_than_late_raises \
    grokking/tests/test_config.py::test_adaptive_norm_validation_skipped_when_disabled \
    grokking/tests/test_config.py::test_adaptive_norm_yaml_roundtrip \
    -v
```

Expected: all 5 FAIL.

- [ ] **Step 3: Add fields and extend `__post_init__` in `grokking/config.py`**

After the `log_every` field, add:

```python
    # Adaptive weight normalization (depth-based)
    adaptive_weight_norm: bool = False
    adaptive_norm_early: float = 2.5   # target norm at layer 0
    adaptive_norm_late: float = 1.2    # target norm at layer L-1 (must be >= 1.0)
    adaptive_norm_gamma: float = 0.3   # max phase correction magnitude
    adaptive_norm_beta: float = 5.0    # tanh sensitivity to gap derivative
    adaptive_norm_alpha: float = 0.9   # EMA smoothing factor for log-gap
```

Extend the existing `__post_init__` by appending to it:

```python
        if self.adaptive_weight_norm:
            if self.adaptive_norm_late < 1.0:
                raise ValueError(
                    f"adaptive_norm_late must be >= 1.0, got {self.adaptive_norm_late}"
                )
            if self.adaptive_norm_early <= self.adaptive_norm_late:
                raise ValueError(
                    f"adaptive_norm_early ({self.adaptive_norm_early}) must be > "
                    f"adaptive_norm_late ({self.adaptive_norm_late})"
                )
```

- [ ] **Step 4: Run all grokking config tests**

```bash
python -m pytest grokking/tests/test_config.py -v
```

Expected: all pass including the pre-existing tests.

- [ ] **Step 5: Commit**

```bash
git add grokking/config.py grokking/tests/test_config.py
git commit -m "feat(grokking): add adaptive weight norm config fields and validation"
```

---

## Task 6: grokking — add function and wire into training loop

**Files:**
- Modify: `grokking/train.py`
- Modify: `grokking/tests/test_train.py`

- [ ] **Step 1: Write the failing tests**

At the top of `grokking/tests/test_train.py`, add after existing imports:

```python
import torch.nn as nn
from grokking.model import GrokTransformer
from grokking.train import apply_adaptive_weight_norm as grok_apply_adaptive_weight_norm
```

Append to `grokking/tests/test_train.py`:

```python
# ── apply_adaptive_weight_norm ────────────────────────────────────────────────

def _adaptive_grok_cfg(**kwargs) -> GrokConfig:
    defaults = dict(
        p=7,
        operation="add",
        n_steps=1,
        log_every=1,
        batch_size=4,
        n_layers=2,
        d_model=32,
        n_heads=2,
        seed=0,
        warmup_ratio=0.0,
        adaptive_weight_norm=True,
        adaptive_norm_early=2.5,
        adaptive_norm_late=1.2,
        adaptive_norm_gamma=0.3,
        adaptive_norm_beta=5.0,
        adaptive_norm_alpha=0.9,
    )
    defaults.update(kwargs)
    return GrokConfig(**defaults)


def test_grok_adaptive_weight_norm_zero_delta_matches_static():
    """With delta_log_gap=0 row norms equal the static depth schedule."""
    cfg = _adaptive_grok_cfg(n_layers=2)
    model = GrokTransformer(cfg)
    grok_apply_adaptive_weight_norm(model, cfg, delta_log_gap=0.0)

    L = len(model.blocks)
    for layer_idx, block in enumerate(model.blocks):
        frac = layer_idx / max(L - 1, 1)
        expected = cfg.adaptive_norm_late + (cfg.adaptive_norm_early - cfg.adaptive_norm_late) * (1.0 - frac)
        for module in block.modules():
            if isinstance(module, nn.Linear):
                row_norms = module.weight.data.norm(dim=1)
                assert torch.allclose(
                    row_norms,
                    torch.full_like(row_norms, expected),
                    atol=1e-5,
                ), f"layer {layer_idx}: expected {expected:.4f}, got {row_norms.mean():.4f}"


def test_grok_adaptive_weight_norm_floor():
    """Row norms never fall below 1.0 regardless of delta_log_gap."""
    cfg = _adaptive_grok_cfg(n_layers=2)
    model = GrokTransformer(cfg)

    for delta in [100.0, -100.0, 0.0]:
        grok_apply_adaptive_weight_norm(model, cfg, delta_log_gap=delta)
        for block in model.blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    row_norms = module.weight.data.norm(dim=1)
                    assert (row_norms >= 1.0 - 1e-5).all(), \
                        f"norm < 1.0 with delta={delta}"


def test_grok_train_adaptive_weight_norm_smoke(tmp_path):
    """train() completes without error when adaptive_weight_norm=True."""
    cfg = _adaptive_grok_cfg(n_steps=20, log_every=10, n_layers=2)
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("")

    with patch("torch.optim.Muon", _MuonStub):
        exp_dir = train(cfg, config_path=config_path)

    assert (exp_dir / "metrics.jsonl").exists()
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest grokking/tests/test_train.py::test_grok_adaptive_weight_norm_zero_delta_matches_static \
    grokking/tests/test_train.py::test_grok_adaptive_weight_norm_floor \
    grokking/tests/test_train.py::test_grok_train_adaptive_weight_norm_smoke \
    -v
```

Expected: all 3 FAIL with `ImportError`.

- [ ] **Step 3: Add `apply_adaptive_weight_norm` to `grokking/train.py`**

Add `import math` to the imports at the top if not already present (it is already imported).

Insert this function after the `_eval_step` function (after line 76):

```python
@torch.no_grad()
def apply_adaptive_weight_norm(
    model: GrokTransformer,
    cfg: GrokConfig,
    delta_log_gap: float,
) -> None:
    """Apply per-layer adaptive weight norm.

    Target norm decreases linearly from cfg.adaptive_norm_early (layer 0)
    to cfg.adaptive_norm_late (layer L-1).  A phase-aware derivative correction
    proportional to tanh(beta * delta_log_gap) is applied most strongly to late
    layers.  Hard floor of 1.0 is enforced on every target.

    Iterates model.blocks only — lm_head, token_embedding, and pos_embedding
    are excluded by design.
    """
    import torch.nn as nn
    L = len(model.blocks)
    for layer_idx, block in enumerate(model.blocks):
        frac = layer_idx / max(L - 1, 1)
        static = cfg.adaptive_norm_late + (cfg.adaptive_norm_early - cfg.adaptive_norm_late) * (1.0 - frac)
        correction = cfg.adaptive_norm_gamma * frac * math.tanh(cfg.adaptive_norm_beta * delta_log_gap)
        target = max(1.0, static - correction)

        for module in block.modules():
            if isinstance(module, nn.Linear):
                norms = module.weight.data.norm(dim=1, keepdim=True).clamp(min=1e-8)
                module.weight.data.mul_(target / norms)
```

- [ ] **Step 4: Initialize EMA state and apply norm in training loop**

In `grokking/train.py`, in the `train` function, after `rows: list[dict] = []` (around line 172), add:

```python
    ema_log_gap: float = 0.0
    delta_log_gap: float = 0.0
```

After the optimizer step block (after `for sched in schedulers: sched.step()`), add:

```python
        if cfg.adaptive_weight_norm:
            apply_adaptive_weight_norm(model, cfg, delta_log_gap)
```

Inside the `if step % cfg.log_every == 0:` block, after `rows.append(row)`, add:

```python
            if cfg.adaptive_weight_norm and row["train_loss"] > 0:
                log_gap = math.log(max(row["val_loss"], 1e-8) / max(row["train_loss"], 1e-8))
                new_ema = cfg.adaptive_norm_alpha * log_gap + (1.0 - cfg.adaptive_norm_alpha) * ema_log_gap
                delta_log_gap = new_ema - ema_log_gap
                ema_log_gap = new_ema
```

- [ ] **Step 5: Run all grokking tests**

```bash
python -m pytest grokking/tests/ -v
```

Expected: all pass.

- [ ] **Step 6: Run full test suite**

```bash
python -m pytest rbf_ffn/tests/ grokking/tests/ -v
```

Expected: all pass. Note the total count — confirm no previously-passing tests regressed.

- [ ] **Step 7: Commit**

```bash
git add grokking/train.py grokking/tests/test_train.py
git commit -m "feat(grokking): add apply_adaptive_weight_norm and wire into training loop"
```
