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
u    = up_proj(x)                        # (B, N, ffn_hidden)
gate = PFDRationalActivation(sin(u + phi))  # (B, N, ffn_hidden)
out  = down_proj(gate * u)               # (B, N, d_model)
```

- `up_proj`: `nn.Linear(d_model, ffn_hidden, bias=False)`
- `down_proj`: `nn.Linear(ffn_hidden, d_model, bias=False)`
- `phi`: `nn.Parameter(torch.randn(ffn_hidden) * 0.02)` — phase shift vector
- `act`: `PFDRationalActivation(n)` — shared across positions and channels

The sine wrapping of `(u + phi)` provides a structured nonlinearity: the phase shift `phi` allows the gate to explore a different region of the activation landscape than the raw value `u`, without a second large matrix.

---

## Parameter Count

| Component     | Shape               | Count (d=256, h=688) |
|---------------|---------------------|----------------------|
| up_proj       | d_model × ffn_hidden | 256 × 688 = 176,128  |
| down_proj     | ffn_hidden × d_model | 688 × 256 = 176,128  |
| phi           | ffn_hidden           | 688                  |
| PFD act (a,b,c,gamma) | 3n + 1 (n=4) | 13                  |
| **Total FFN** |                     | **~352,957**         |

Compare to `PFDRationalGatedFFN` (3 matrices): ~529,085 — a ~33% reduction.

---

## Components Changed

### `rbf_ffn/models/rational_ffn.py`

Add `FirstOrderPFDRationalFFN` class after `PFDRationalGatedFFN`. Uses the existing `PFDRationalActivation`.

### `rbf_ffn/models/transformer_block.py`

Add `FirstOrderPFDRationalBlock`:
- Import `FirstOrderPFDRationalFFN`
- Pre-norm with `RMSNorm`, same pattern as all other blocks
- `ffn = FirstOrderPFDRationalFFN(cfg, n=cfg.pfd_n)`

### `rbf_ffn/config.py`

Add `"first_order_pfd_rational"` to the `model_type` field comment.

### `rbf_ffn/models/model.py`

Add to `BlockClass` dict in `CausalLM`:
```python
"first_order_pfd_rational": FirstOrderPFDRationalBlock,
```

### `rbf_ffn/configs/first_order_pfd_rational_ffn.yaml`

New YAML config copied from `pfd_rationalglu_ffn.yaml` with:
```yaml
model_type: first_order_pfd_rational
```

### `rbf_ffn/tests/test_first_order_pfd_rational_ffn.py`

New test file covering:
1. `test_ffn_output_shape` — output is `(B, N, d_model)`
2. `test_ffn_no_bias` — `up_proj.bias` and `down_proj.bias` are `None`
3. `test_phi_receives_gradient` — `ffn.phi.grad` is not `None` after backward
4. `test_input_gradient_flows` — `x.grad` is not `None` after backward
5. `test_pfd_act_receives_gradient` — `ffn.act.a.grad` is not `None` after backward

---

## Optimizer Routing

`phi` is a 1D parameter (vector). The existing rule in `build_optimizer_groups`:
- `param.ndim == 2` → Muon
- else → AdamW

`phi` routes to **AdamW** automatically — no changes needed to optimizer logic.

---

## Constraints

- No bias on projections (Llama convention).
- `pfd_n` from config controls the number of PFD terms (default 4).
- No changes to training, data, or attention code.
