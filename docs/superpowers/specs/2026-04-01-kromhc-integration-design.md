# KromHC Integration into rbf_ffn — Design Spec

**Date**: 2026-04-01
**Status**: Approved

## Goal

Make `rbf_ffn/` the single source of truth for transformer experiments. Port KromHC head mixing from `kromhc_transformer/` into `rbf_ffn/` as an orthogonal, composable flag. Archive `kromhc_transformer/`.

---

## Architecture

KromHC head mixing is orthogonal to FFN choice. It is integrated via a **wrapper pattern**: existing block classes are untouched, and a thin `KromHCWrapper` optionally wraps any block at model construction time.

```
model.py
  └─ build inner block (LlamaBlock / RationalGLUBlock / PFDRationalGLUBlock / etc.)
  └─ if cfg.use_kromhc: wrap in KromHCWrapper
       KromHCWrapper
         ├─ inner_block    — any existing transformer block
         ├─ head_mixer     — KromHCHeadMixer (ported from kromhc_transformer)
         ├─ mixer_proj     — nn.Linear(d_model, d_model, bias=False)
         └─ forward(x) → (x, H)
```

`KromHCWrapper.forward(x)`:
1. Run `inner_block(x)` to get the full block output (post-attention + post-FFN residual).
   **Note**: the wrapper intercepts at the block output level, not mid-block. This keeps inner blocks unmodified and means head mixing is applied after the full block residual.
   *(Alternative: intercept between attention and FFN — defer to implementation decision.)*
2. Reshape block output `(B, N, D)` → `(B*N, n_heads, head_dim)`.
3. Pass through `KromHCHeadMixer` → `(mixed, H)`.
4. Reshape back and project through `mixer_proj` → `(B, N, D)`.
5. Return `(x + mixed_proj_out, H)`.
   *(Residual formulation: mixing is an additive correction, not a replacement.)*

---

## Components

### `rbf_ffn/models/head_mixer.py` (new)

Ported directly from `kromhc_transformer/models/head_mixer.py`. No changes required.

**`KromHCHeadMixer(n_heads, head_dim, mixer_hidden=32)`**:
- Decomposes `n_heads` as `2^K` (must be power of 2; 8 heads → K=3).
- Registers K fixed permutation bases `(2, 2, 2)` as buffers (non-trainable).
- K small MLPs: `head_dim → mixer_hidden → 2` with softmax output → convex weights over the 2 permutation matrices.
- Per-token Kronecker chain: `U_0 ⊗ U_1 ⊗ ... ⊗ U_{K-1}` → `(B*N, n_heads, n_heads)` doubly-stochastic matrix H.
- Forward: `mixed = H @ x`, returns `(mixed, H)`.

### `rbf_ffn/models/transformer_block.py` (modified)

Add `KromHCWrapper` class. All existing block classes unchanged.

```python
class KromHCWrapper(nn.Module):
    def __init__(self, inner_block, cfg):
        self.inner_block = inner_block
        self.head_mixer = KromHCHeadMixer(cfg.n_heads, cfg.d_model // cfg.n_heads)
        self.mixer_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        x = self.inner_block(x)           # (B, N, D)
        B, N, D = x.shape
        heads = x.view(B*N, self.n_heads, self.head_dim)
        mixed, H = self.head_mixer(heads)  # H: (B*N, n_heads, n_heads)
        out = self.mixer_proj(mixed.view(B, N, D))
        H = H.view(B, N, self.n_heads, self.n_heads)
        return x + out, H
```

### `rbf_ffn/models/model.py` (modified)

- `build_block()` constructs the inner block as before, then wraps if `cfg.use_kromhc`.
- `CausalLM.forward()` collects H from each wrapped block; returns `(logits, hs)` where `hs` is a list of per-layer H tensors (or `None` if not using KromHC).
- Optimizer routing: no changes needed — head mixer params are scalars/1D → already route to AdamW.

### `rbf_ffn/config.py` (modified)

```python
# KromHC head mixing
use_kromhc: bool = False       # wrap any block with KromHC head mixing
kromhc_mixer_hidden: int = 32  # hidden dim of per-factor weight MLP
```

### `rbf_ffn/train.py` (modified)

When `cfg.use_kromhc=True`, log per-step summary stats for H:
- `kromhc/H_row_entropy_mean` — mean entropy of H rows (measures how diffuse mixing is)
- `kromhc/H_offdiag_mass_mean` — mean off-diagonal mass (measures cross-head mixing strength)

Computed from the list of H tensors returned by `model.forward()`, averaged across layers and batch. Only computed every N steps (same cadence as existing metric logging) to avoid overhead.

---

## New Config Files

| File | model_type | use_kromhc | Notes |
|------|-----------|-----------|-------|
| `baseline_kromhc.yaml` | baseline | true | Direct comparison to kromhc_transformer baseline |
| `baseline_qk_norm_kromhc.yaml` | baseline | true | + QK norm (matches default in kromhc_transformer) |
| `pfd_rationalglu_qk_norm_weight_norm_kromhc.yaml` | pfd_rationalglu | true | Best FFN + head mixing |

---

## Migration & Cleanup

- Move `kromhc_transformer/` → `archive/kromhc_transformer/` at repo root.
- Add a note to `rbf_ffn/OVERVIEW.md` recording the consolidation and pointing to the archive.
- No code deleted.

---

## What Is NOT Changed

- All existing block classes (`LlamaBlock`, `RationalGLUBlock`, etc.) — untouched.
- All existing configs — untouched (default `use_kromhc=False`).
- Optimizer routing logic — untouched.
- Training loop structure — only additive logging when `use_kromhc=True`.
