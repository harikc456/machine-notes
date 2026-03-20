# RationalGLU: Gated Rational Activation Ablation

**Date:** 2026-03-16
**Status:** Design approved
**Context:** Follows the non-gated RationalFFN experiment. This experiment tests whether adding a gating structure — replacing SiLU in SwiGLU with a learnable RationalActivation — improves over both the SiLU baseline and the non-gated rational variant.

---

## Goal

Add a fourth FFN variant to the WikiText-103 ablation suite:

| Config | model_type | FFN |
|---|---|---|
| `baseline.yaml` | `baseline` | SwiGLU: `down(SiLU(gate) * up)` |
| `rational_ffn.yaml` | `rational` | RationalFFN: `down(RationalAct(up))` — no gate |
| `rationalglu_ffn.yaml` | `rationalglu` | RationalGatedFFN: `down(RationalAct(gate) * up)` |

The only variable between `baseline` and `rationalglu` is the gate activation: fixed `SiLU` vs learnable `RationalActivation`.

---

## Architecture

### RationalActivation (reused, unchanged)

Defined in `rbf_ffn/models/rational_ffn.py`. No changes.

```
f(x) = P(x) / Q(x)

P(x) = a0 + a1·x + a2·x² + a3·x³     (Horner: a0 + x·(a1 + x·(a2 + x·a3)))
Q(x) = 1 + |x·(b0 + x·b1)|

Parameters:
  a: nn.Parameter, shape (4,), init [0.4401, 0.5, 0.507, 0.05]
  b: nn.Parameter, shape (2,), init [0.0, 0.01]
```

- `Q(x) >= 1` always — numerically safe.
- Applied element-wise; `a` and `b` shared across all positions and channels.
- One independent `RationalActivation` per block (6 independent sets of `a`, `b`).

### RationalGatedFFN

Appended to `rbf_ffn/models/rational_ffn.py`:

```python
class RationalGatedFFN(nn.Module):
    """
    Gated FFN with learnable rational activation replacing SiLU.

        gate = RationalActivation(gate_proj(x))
        out  = down_proj(gate * up_proj(x))

    Matches SwiGLU parameter count at ffn_hidden=688.
    No bias (Llama convention). Input/output: (B, N, d_model).
    """
    def __init__(self, cfg: RBFFFNConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up_proj   = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.act       = RationalActivation()
        self.down_proj = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))
```

- `RationalActivation` is applied to the gate path only; `up_proj` has no activation.
- `cfg.ffn_hidden` drives the hidden dimension — never hardcoded.
- No bias on any projection (Llama convention).

### Parameter count

At `d_model=256`, `ffn_hidden=688`, `n_layers=6`:

| Variant | Projections | Act params | FFN params/layer | Total FFN |
|---|---|---|---|---|
| SwiGLU (baseline) | 3 × (256×688) | 0 | 528,384 | 3,170,304 |
| RationalFFN (non-gated) | 2 × (256×688) | 6 | 352,262 | 2,113,572 |
| RationalGatedFFN | 3 × (256×688) | 6 | 528,390 | 3,170,340 |

RationalGatedFFN is parameter-matched to SwiGLU (difference of 36 params = 6 blocks × 6 rational params).

### RationalGLUBlock

Appended to `rbf_ffn/models/transformer_block.py`:

```python
class RationalGLUBlock(nn.Module):
    """
    Transformer block with RationalGatedFFN replacing the MLP.

        x = x + attn(norm1(x))
        x = x + ffn(norm2(x))    ← ffn is RationalGatedFFN

    Pre-norm with RMSNorm. No bias anywhere.
    """
    def __init__(self, cfg: RBFFFNConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.attn  = CausalSelfAttention(cfg)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.ffn   = RationalGatedFFN(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
```

### Model Selection

Extend import in `model.py`:

```python
from rbf_ffn.models.transformer_block import LlamaBlock, RBFBlock, RationalBlock, RationalGLUBlock
```

Add entry to dispatch dict:

```python
BlockClass = {
    "baseline":    LlamaBlock,
    "rbf":         RBFBlock,
    "rational":    RationalBlock,
    "rationalglu": RationalGLUBlock,
}[cfg.model_type]
```

Update `CausalLM` docstring to include:

```
"rationalglu" → RationalGLUBlock (RationalGatedFFN)
```

---

## Config

### `config.py` comment update

```python
model_type: str = "rbf"  # "baseline" | "rbf" | "rational" | "rationalglu"
```

### New config file: `rbf_ffn/configs/rationalglu_ffn.yaml`

Identical to `rational_ffn.yaml` with `model_type: rationalglu`:

```yaml
# RationalGLU ablation — gate_variant, sigma_variant, K, centers,
# sigma_init, and sinkhorn_iters are inert for model_type: rationalglu
# but present for a uniform config schema across all runs.
model_type: rationalglu
gate_variant: G0
sigma_variant: global
d_model: 256
n_heads: 8
n_layers: 6
ffn_hidden: 688
dropout: 0.1
K: 5
centers: [-1.0, -0.5, 0.0, 0.5, 1.0]
sigma_init: 0.5
sinkhorn_iters: 20
vocab_size: 50257
seq_len: 512
seed: 42
n_epochs: 3
batch_size: 16
muon_lr: 0.02
adamw_lr: 0.0003
adamw_wd: 0.1
warmup_ratio: 0.02
grad_clip: 1.0
grad_accum_steps: 1
```

---

## Optimizer Groups

`RationalActivation.a` (shape `(4,)`) and `.b` (shape `(2,)`) are 1-D → fall into AdamW under existing rule 4. No changes to `build_optimizer_groups`.

| Parameter | Shape | Group |
|---|---|---|
| `gate_proj.weight` | `(ffn_hidden, d_model)` | Muon |
| `up_proj.weight` | `(ffn_hidden, d_model)` | Muon |
| `down_proj.weight` | `(d_model, ffn_hidden)` | Muon |
| `act.a` | `(4,)` | AdamW |
| `act.b` | `(2,)` | AdamW |

---

## Metrics

No changes to `train.py`. The `if cfg.model_type == "rbf"` guard already excludes `"rationalglu"` from sigma stats.

---

## File Changes

### Modified Files

```
rbf_ffn/
  models/
    rational_ffn.py        # append RationalGatedFFN class
    transformer_block.py   # append RationalGLUBlock class;
                           # update import to:
                           #   from rbf_ffn.models.rational_ffn import RationalFFN, RationalGatedFFN
    model.py               # extend import to include RationalGLUBlock;
                           # add "rationalglu" entry to dispatch dict;
                           # update CausalLM docstring
  config.py                # update model_type comment to include "rationalglu"
```

### New Files

```
rbf_ffn/
  configs/
    rationalglu_ffn.yaml   # model_type: rationalglu (full contents above)
```

---

## Tests

### `rbf_ffn/tests/test_rational_ffn.py` (extend existing)

Update the import line to:
```python
from rbf_ffn.models.rational_ffn import RationalActivation, RationalFFN, RationalGatedFFN
```

Config helper:

```python
def make_gated_cfg():
    return RBFFFNConfig(
        d_model=D, n_heads=4, n_layers=2, seq_len=N,
        ffn_hidden=86, dropout=0.0, model_type="rationalglu",
    )
```

| Test | Assertion |
|---|---|
| `test_rational_gated_ffn_shape` | `RationalGatedFFN(make_gated_cfg())(x)` → shape `(B, N, D)` |
| `test_rational_gated_ffn_no_bias` | `gate_proj.bias`, `up_proj.bias`, `down_proj.bias` are all `None` |
| `test_rational_gated_ffn_gate_gradient` | `act.a.grad` and `act.b.grad` non-None after `output.sum().backward()` |

### `rbf_ffn/tests/test_transformer_block.py` (extend existing)

Update the import line to:
```python
from rbf_ffn.models.transformer_block import LlamaBlock, RBFBlock, RationalBlock, RationalGLUBlock
```

| Test | Assertion |
|---|---|
| `test_rationalglu_block_shape` | `RationalGLUBlock(cfg)(x)` → shape `(B, N, D)` |
| `test_rationalglu_block_gradient_flow` | `x.grad` non-None after backward |
| `test_rationalglu_block_residual_connection` | Zero `ffn.down_proj.weight` + `attn.o_proj.weight` → `block(x) ≈ x` (atol=1e-5) |

### `rbf_ffn/tests/test_model.py` (extend existing)

| Test | Assertion |
|---|---|
| `test_rationalglu_output_shape` | `make_model("rationalglu")(tokens)` → shape `(B, N, vocab_size)` |
| `test_rationalglu_params_in_adamw` | `id(block.ffn.act.a)` in AdamW AND not in Muon; same for `act.b`; checked for all blocks |
