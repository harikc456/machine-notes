# Spec: FirstOrderPFDRationalFFN

**Date:** 2026-03-17
**model_type:** `first_order_pfd_rational`

---

## Overview

Add a `FirstOrderPFDRationalFFN` variant to the existing rational FFN family. It uses only two large projection matrices (`up_proj`, `down_proj`) instead of three, achieving ~33% fewer FFN parameters versus SwiGLU. A learnable phase-shift vector `phi` decouples the gate signal from the value signal despite sharing the same projection `u`.

The gate activation uses `PFDRationalActivation` (partial fraction decomposition), consistent with the `pfd_rational` and `pfd_rationalglu` variants.

---

## Architecture

```
u    = up_proj(x)                           # (B, N, ffn_hidden)
gate = PFDRationalActivation(sin(u + phi))  # (B, N, ffn_hidden)
out  = down_proj(gate * u)                  # (B, N, d_model)
```

- `up_proj`: `nn.Linear(d_model, ffn_hidden, bias=False)`
- `down_proj`: `nn.Linear(ffn_hidden, d_model, bias=False)`
- `phi`: `nn.Parameter(torch.randn(ffn_hidden) * 0.02)` — phase shift vector; small init (0.02) avoids large initial offsets that would saturate `sin` and produce near-zero gradients early in training
- `act`: `PFDRationalActivation(n)` — shared across positions and channels

The sine wrapping of `(u + phi)` provides a structured nonlinearity: the phase shift `phi` allows the gate to explore a different region of the activation landscape than the raw value `u`, without a second large matrix.

**Numerical safety note:** `sin(u + phi)` is bounded in `[-1, 1]`. The `PFDRationalActivation` denominator is `x^2 + c_i^2`. `c` is initialized as `arange(1, n+1)` so the smallest denominator at init is `(-1)^2 + 1^2 = 2`, which is safe. If `c_i` drifts to zero during training and `x = 0`, the denominator collapses to zero causing NaN. This is an inherited limitation shared by `PFDRationalFFN` and `PFDRationalGatedFFN` — no existing guard exists for this in the codebase and none is added here.

---

## Parameter Count

| Component                  | Shape                | Count (d=256, h=688)     |
|----------------------------|----------------------|--------------------------|
| up_proj                    | d_model × ffn_hidden | 256 × 688 = 176,128      |
| down_proj                  | ffn_hidden × d_model | 688 × 256 = 176,128      |
| phi                        | ffn_hidden           | 688                      |
| PFD act: a, b, c (n each) + gamma (scalar) | — | 3×4 + 1 = 13  |
| **Total FFN**              |                      | **353,957**              |

Compare to `PFDRationalGatedFFN` (3 matrices): `176,128 × 3 + 13 = 528,397` — a ~33% reduction.

---

## Components Changed

### `rbf_ffn/models/rational_ffn.py`

Add `FirstOrderPFDRationalFFN` class after `PFDRationalGatedFFN`. Signature:

```python
class FirstOrderPFDRationalFFN(nn.Module):
    def __init__(self, cfg: RBFFFNConfig, n: int = 4):
        ...
        self.phi = nn.Parameter(torch.randn(cfg.ffn_hidden) * 0.02)
        self.act = PFDRationalActivation(n)
        ...
```

Uses the existing `PFDRationalActivation` (no new imports needed in this file).

### `rbf_ffn/models/transformer_block.py`

1. Append `FirstOrderPFDRationalFFN` to the **end** of the existing import line (do not replace it):
   ```python
   from rbf_ffn.models.rational_ffn import RationalFFN, RationalGatedFFN, PFDRationalFFN, PFDRationalGatedFFN, FirstOrderPFDRationalFFN
   ```

2. Add `FirstOrderPFDRationalBlock` — identical to `PFDRationalGLUBlock` except `self.ffn`:
   ```python
   class FirstOrderPFDRationalBlock(nn.Module):
       def __init__(self, cfg: RBFFFNConfig):
           super().__init__()
           self.norm1 = nn.RMSNorm(cfg.d_model)
           self.attn  = CausalSelfAttention(cfg)
           self.norm2 = nn.RMSNorm(cfg.d_model)
           self.ffn   = FirstOrderPFDRationalFFN(cfg, n=cfg.pfd_n)

       def forward(self, x: torch.Tensor) -> torch.Tensor:
           x = x + self.attn(self.norm1(x))
           x = x + self.ffn(self.norm2(x))
           return x
   ```

### `rbf_ffn/config.py`

Update the `model_type` field comment:
```python
model_type: str = "rbf"  # "baseline" | "rbf" | "rational" | "rationalglu" | "pfd_rational" | "pfd_rationalglu" | "first_order_pfd_rational"
```

### `rbf_ffn/models/model.py`

1. Append `FirstOrderPFDRationalBlock` to the **end** of the existing import line:
   ```python
   from rbf_ffn.models.transformer_block import LlamaBlock, RBFBlock, RationalBlock, RationalGLUBlock, PFDRationalBlock, PFDRationalGLUBlock, FirstOrderPFDRationalBlock
   ```

2. Add to `BlockClass` dict in `CausalLM.__init__`:
   ```python
   "first_order_pfd_rational": FirstOrderPFDRationalBlock,
   ```

3. Update the `CausalLM` docstring to add:
   ```
   "first_order_pfd_rational" → FirstOrderPFDRationalBlock (FirstOrderPFDRationalFFN)
   ```

### `rbf_ffn/configs/first_order_pfd_rational_ffn.yaml`

New file. Copy `pfd_rationalglu_ffn.yaml` verbatim, change only `model_type`:
```yaml
model_type: first_order_pfd_rational
n_epochs: 3
```
All other fields remain identical to `pfd_rationalglu_ffn.yaml`.

---

## Tests

### `rbf_ffn/tests/test_first_order_pfd_rational_ffn.py` (new file)

```python
B, N, D = 2, 16, 32

def make_cfg():
    return RBFFFNConfig(
        d_model=D, n_heads=4, n_layers=2, seq_len=N,
        ffn_hidden=86, dropout=0.0, model_type="first_order_pfd_rational", pfd_n=4,
    )
```

Tests:
1. `test_ffn_output_shape` — `ffn(x).shape == (B, N, D)`
2. `test_ffn_no_bias` — `up_proj.bias is None` and `down_proj.bias is None`
3. `test_phi_receives_gradient` — `ffn.phi.grad is not None` after `ffn(x).sum().backward()`
4. `test_input_gradient_flows` — `x.grad is not None` after backward with `requires_grad=True`
5. `test_pfd_act_receives_gradient` — `ffn.act.a.grad is not None` after backward

### `rbf_ffn/tests/test_transformer_block.py` (extend existing file)

Append `FirstOrderPFDRationalBlock` to the existing import line.

Config fixture for the new block tests (use the module-level constants `D`, `H`, `B`, `N` already defined in the file):
```python
def make_first_order_pfd_cfg():
    return RBFFFNConfig(
        d_model=D, n_heads=H, dropout=0.0,
        model_type="first_order_pfd_rational", pfd_n=4,
    )
```

Add three tests:

1. `test_first_order_pfd_rational_block_shape`
   ```python
   block = FirstOrderPFDRationalBlock(make_first_order_pfd_cfg())
   x = torch.randn(B, N, D)
   assert block(x).shape == x.shape
   ```

2. `test_first_order_pfd_rational_block_gradient_flow`
   ```python
   block = FirstOrderPFDRationalBlock(make_first_order_pfd_cfg())
   x = torch.randn(B, N, D)
   block(x).sum().backward()
   assert block.ffn.phi.grad is not None
   ```

3. `test_first_order_pfd_rational_block_residual_connection`
   — Zero `block.ffn.down_proj.weight` and `block.attn.o_proj.weight` with `torch.no_grad()`, assert `torch.allclose(block(x), x, atol=1e-5)`. Mirrors the existing `test_rational_block_residual_connection` pattern exactly.

### `rbf_ffn/tests/test_model.py` (extend existing file)

Add two tests:

1. `test_first_order_pfd_rational_output_shape` — construct `CausalLM` with `model_type="first_order_pfd_rational"`, assert logits shape `(B, N, vocab_size)`

2. `test_first_order_pfd_rational_params_in_adamw` — mirrors the `test_rationalglu_params_in_adamw` pattern. Construct `CausalLM`, call `build_optimizer_groups(model)`, collect `adamw_ids = {id(p) for p in adamw_params}` and `muon_ids = {id(p) for p in muon_params}`. For every block assert:
   ```python
   assert id(block.ffn.phi)       in adamw_ids and id(block.ffn.phi)       not in muon_ids
   assert id(block.ffn.act.a)     in adamw_ids and id(block.ffn.act.a)     not in muon_ids
   assert id(block.ffn.act.b)     in adamw_ids and id(block.ffn.act.b)     not in muon_ids
   assert id(block.ffn.act.c)     in adamw_ids and id(block.ffn.act.c)     not in muon_ids
   assert id(block.ffn.act.gamma) in adamw_ids and id(block.ffn.act.gamma) not in muon_ids
   ```

---

## Optimizer Routing

`phi`, `a`, `b`, `c` are 1D vectors (`ndim == 1`); `gamma` is a 0-D scalar (`ndim == 0`). All route to AdamW via the `else` branch of the existing rule in `build_optimizer_groups`:
- `"sigma_raw" in name` → AdamW
- `pid == emb_id` → AdamW
- `param.ndim == 2` → Muon
- else → **AdamW**

All new parameters route to AdamW automatically. No changes needed to optimizer logic.

---

## Constraints

- No bias on projections (Llama convention).
- `pfd_n` from config controls the number of PFD terms (default 4).
- No changes to training, data, or attention code.
