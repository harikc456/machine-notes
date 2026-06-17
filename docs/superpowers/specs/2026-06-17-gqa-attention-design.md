# GQA Attention Variant Design

**Date:** 2026-06-17  
**Status:** Approved

## Summary

Add Grouped Query Attention (GQA) support to all five existing attention variants
(`standard`, `polar`, `xsa`, `kv_shared`, `xsa_kv_shared`) via a single new config
field `n_kv_heads`. Setting `n_kv_heads=1` gives MQA; any value between 1 and
`n_heads` gives GQA; the default of 0 resolves to `n_heads` (full MHA, no change
in behaviour).

## Config

**File:** `rbf_ffn/config.py`

Add to `ModelConfig`:

```python
n_kv_heads: int = 0  # 0 = match n_heads (MHA). 1 = MQA. 1 < n < n_heads = GQA.
```

In `__post_init__`:

```python
if self.n_kv_heads == 0:
    self.n_kv_heads = self.n_heads
if self.n_heads % self.n_kv_heads != 0:
    raise ValueError(
        f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
    )
```

## Attention Classes

**File:** `rbf_ffn/models/attention.py`

Same pattern applied to all five classes. Changes per class:

### Classes with separate K/V projections: `CausalSelfAttention`, `ExclusiveSelfAttention`, `PolarAttention`

**`__init__` changes:**
- Store `self.n_kv_heads = cfg.n_kv_heads` and `self.n_groups = n_heads // n_kv_heads`.
- `k_proj = nn.Linear(D, n_kv_heads * head_dim, bias=False)`
- `v_proj = nn.Linear(D, n_kv_heads * head_dim, bias=False)`
- `qkv_gain` k/v parameters sized `n_kv_heads` instead of `n_heads`.
- `k_norm = nn.RMSNorm(head_dim)` — unchanged (normalises last dim regardless of head count).

**`forward` changes:**
- Split K/V to `(B, n_kv_heads, N, head_dim)` instead of `(B, n_heads, N, head_dim)`.
- Apply RoPE, gain, QK-norm to K at `n_kv_heads` size.
- Expand before attention: `k = k.repeat_interleave(self.n_groups, dim=1)`, same for `v`.
- When `n_groups == 1`, `repeat_interleave` is a no-op.

### KV-shared classes: `KVSharedAttention`, `KVSharedExclusiveSelfAttention`

**`__init__` changes:**
- `kv_proj = nn.Linear(D, n_kv_heads * head_dim, bias=False)`
- Same gain/norm sizing as above.

**`forward` changes:**
- Split to `(B, n_kv_heads, N, head_dim)`, expand K and V via `repeat_interleave`.

### `PolarAttention` specifics

Polar builds its attention matrix manually (no SDPA). The expand step occurs after
the transpose to `(B, n_kv_heads, N, .)` and before the `q_dir @ k_dir.T` matmul.
Both `r_k` and `v` are expanded the same way so all shapes align.

## Tests

**File:** `rbf_ffn/tests/test_transformer_block.py`

1. **Shape test with GQA:** parametrize all five `attn_type` values with `n_kv_heads=2`
   (given `n_heads=4` in the fixture → 2 groups of 2). Assert output shape `(B, N, D)`.
2. **Gradient test:** one variant with `n_kv_heads=2`, assert `x.grad is not None`.
3. **Config validation:** `ModelConfig(n_heads=8, n_kv_heads=3)` raises `ValueError`.
4. **MHA no-op:** `n_kv_heads=0` (or `n_kv_heads=n_heads`) produces identical output
   to `n_kv_heads` unset.

## Non-Goals

- No new `ATTN_REGISTRY` keys.
- No cross-layer KV sharing or sliding-window attention.
- No KV cache for inference.
