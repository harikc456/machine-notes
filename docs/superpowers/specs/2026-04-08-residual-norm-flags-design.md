# Residual Connections and Norm Flags — Design Spec

**Date:** 2026-04-08
**Status:** Approved

## Motivation

The SIGReg transformer uses `PlainBlock` (`x = ffn(attn(x))`) with no residual connections and no normalisation layers. Without residuals, CE loss gradients vanish through 6 layers, preventing the language model from learning. Adding optional residual and norm flags lets experiments compare plain, residual-only, norm-only, and pre-norm+residual configurations without changing existing baselines.

## Config Changes (`sigreg/config.py`)

Two new fields on `SIGRegConfig`:

```python
use_residual: bool = False
norm_type: str = "none"   # "none" | "rmsnorm" | "layernorm"
```

`__post_init__` adds a validation assert:
```python
assert self.norm_type in ("none", "rmsnorm", "layernorm"), ...
```

Defaults are `False` / `"none"` so existing configs and tests are unaffected.

## Block Changes (`sigreg/models/block.py`)

`PlainBlock` is renamed to `TransformerBlock`. It reads `cfg.use_residual` and `cfg.norm_type` to instantiate optional norms and pick its forward path.

**Norm instantiation (in `__init__`):**
- `norm_type == "rmsnorm"` → two `nn.RMSNorm(d_model)` instances (`norm_attn`, `norm_ffn`)
- `norm_type == "layernorm"` → two `nn.LayerNorm(d_model)` instances
- `norm_type == "none"` → no norm modules

**Forward combinations:**

| `use_residual` | `norm_type` | forward |
|---|---|---|
| `False` | `"none"` | `x = ffn(attn(x))` ← unchanged baseline |
| `True` | `"none"` | `x = x + attn(x)` then `x = x + ffn(x)` |
| `False` | rmsnorm/ln | `x = ffn(attn(norm_attn(x)))` |
| `True` | rmsnorm/ln | `x = x + attn(norm_attn(x))` then `x = x + ffn(norm_ffn(x))` (pre-norm) |

Pre-norm ordering is used when both flags are active (modern standard: LLaMA, GPT-2).

## Model Changes (`sigreg/models/model.py`)

- Import `TransformerBlock` instead of `PlainBlock`
- Replace `PlainBlock(cfg)` with `TransformerBlock(cfg)` in `nn.ModuleList`
- No logic changes

## YAML Changes

Both `sigreg/configs/baseline.yaml` and `sigreg/configs/weak_loss.yaml` get explicit entries:

```yaml
use_residual: false
norm_type: "none"
```

This makes the flags visible to the reader without changing behaviour.

## What Is Not Changed

- `SIGRegCausalLM.forward` — hidden state collection is unchanged
- `train.py` — no changes
- `sigreg/losses.py` — no changes
- The SIGReg regularisation applies to block outputs regardless of residual/norm setting

## Testing

- Existing baseline configs (`use_residual: false`, `norm_type: "none"`) must produce identical forward-pass output to the current `PlainBlock`
- A smoke test with `use_residual: true, norm_type: "rmsnorm"` should run without error and produce a lower CE loss after a handful of steps than the plain baseline
