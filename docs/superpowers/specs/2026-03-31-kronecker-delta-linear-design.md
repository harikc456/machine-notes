# KroneckerDeltaLinear Design

**Date:** 2026-03-31
**Branch:** feat/kronecker-mlp-and-docs

## Summary

Add `KroneckerDeltaLinear` as a new drop-in linear variant alongside the existing `KroneckerLinear`. It combines a Kronecker-factored core (trained by Muon) with a low-rank delta pathway (trained by AdamW), controlled by a separate config flag.

## Architecture

### New class: `KroneckerDeltaLinear`

Added to `rbf_ffn/models/kronecker_linear.py` alongside `KroneckerLinear`.

Two parameter pathways:
- **Kronecker core** — `A` (out1 × in1) and `B` (out2 × in2), routed to Muon via the existing `ndim == 2` rule.
- **Low-rank delta** — `delta_C` (out_features × delta_rank) and `delta_D` (delta_rank × in_features), routed to AdamW via a new `"delta_"` name rule.

Forward pass:
```
kron_out  = einsum('...ij, mi, nj -> ...mn', x_reshaped, A, B).reshape(...)
delta_out = (x @ delta_D.T) @ delta_C.T
out       = kron_out + delta_out + bias
```

`delta_rank` is passed in at construction from `cfg.kronecker_delta_rank`.

### Config (`rbf_ffn/config.py`)

Two new fields:

```python
kronecker_delta_mlp: bool = False   # Replace up_proj/down_proj with KroneckerDeltaLinear
kronecker_delta_rank: int = 16      # Rank of the low-rank delta pathway
```

`kronecker_mlp` and `kronecker_delta_mlp` are independent flags. If both are `True`, `kronecker_delta_mlp` takes precedence (checked first in the FFN branch).

### FFN wiring (`rbf_ffn/models/llama_ffn.py`)

`SwiGLUFFN.__init__` uses a three-way branch:

- `kronecker_delta_mlp=True`: `gate_proj` → `nn.Linear`; `up_proj`, `down_proj` → `KroneckerDeltaLinear`
- `kronecker_mlp=True`: all three → `KroneckerLinear`
- else: all three → `nn.Linear`

`gate_proj` is always `nn.Linear` when `kronecker_delta_mlp` is active.

### Optimizer routing (`rbf_ffn/models/model.py`)

New rule in `build_optimizer_groups`, inserted before the `ndim == 2` check:

```python
elif "delta_" in name:
    adamw.append(param)
```

This ensures `delta_C` and `delta_D` go to AdamW regardless of their ndim, consistent with the existing `sigma_raw` pattern.

## Files Changed

| File | Change |
|------|--------|
| `rbf_ffn/models/kronecker_linear.py` | Add `KroneckerDeltaLinear` class |
| `rbf_ffn/config.py` | Add `kronecker_delta_mlp`, `kronecker_delta_rank` fields |
| `rbf_ffn/models/llama_ffn.py` | Three-way branch in `SwiGLUFFN.__init__` |
| `rbf_ffn/models/model.py` | Add `"delta_"` routing rule in `build_optimizer_groups` |

## Out of Scope

- No changes to attention layers
- No changes to training loop or checkpoint logic
- `kronecker_delta_mlp` only affects `SwiGLUFFN`; other FFN types are unchanged
