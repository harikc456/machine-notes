# Rational Activation: SiLU vs Rational Function Ablation

**Date:** 2026-03-16
**Status:** Design approved
**Context:** RBF-FFN was inferior to the Llama baseline on WikiText-103. This experiment tests whether replacing the SwiGLU FFN with a simpler non-gated FFN using a learnable Rational Activation improves over the SiLU baseline.

---

## Goal

Compare two Llama transformer variants on WikiText-103:

1. **Baseline** — SwiGLU FFN: `gate_proj → SiLU(·) * up_proj → down_proj`
2. **Rational** — RationalFFN: `up_proj → RationalActivation → down_proj`

All other architecture components (attention, norms, embeddings, training loop) are identical. The only variable is the FFN module.

---

## Architecture

### Shared Stack

Identical to the existing Llama baseline: RMSNorm + RoPE + CausalSelfAttention + pre-norm residual blocks. See `2026-03-12-rbf-ffn-wikitext-design.md` for full attention spec.

### RationalActivation

```
f(x) = P(x) / Q(x)

P(x) = a0 + a1·x + a2·x² + a3·x³     (Horner: a0 + x·(a1 + x·(a2 + x·a3)))
Q(x) = 1 + |x·(b0 + x·b1)|

Python: q_x = 1.0 + torch.abs(x * (b[0] + x * b[1]))

Parameters:
  a: nn.Parameter, shape (4,), init [0.4401, 0.5, 0.507, 0.05]
  b: nn.Parameter, shape (2,), init [0.0, 0.01]
```

- Applied **element-wise**; `a` and `b` are shared across all positions and channels.
- `Q(x) >= 1` always because `|·| >= 0` — the absolute value wraps the entire product `x·(b0 + x·b1)`. Division is always numerically safe.
- Init values from the sample code are reasonable starting values for the learnable rational function; they do not constitute a tight pointwise SiLU approximation (e.g. `f(0) = a0 = 0.4401 ≠ SiLU(0) = 0`). The parameters learn away from this init during training.
- One `RationalActivation` instance per block (6 independent sets of `a`, `b`).
- Total extra parameters: 6 × 6 = 36 (negligible).

### RationalFFN

```python
class RationalFFN(nn.Module):
    def __init__(self, cfg: RBFFFNConfig):
        super().__init__()
        self.up_proj   = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.act       = RationalActivation()
        self.down_proj = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down_proj(self.act(self.up_proj(x)))
```

- `cfg.ffn_hidden` drives the hidden dimension — never hardcoded.
- No bias on projections (Llama convention).
- Input/output: `(B, N, d_model)`.

### RationalBlock

Identical structure to `LlamaBlock`:

```python
class RationalBlock(nn.Module):
    def __init__(self, cfg: RBFFFNConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.attn  = CausalSelfAttention(cfg)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.ffn   = RationalFFN(cfg)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
```

### Model Selection

`CausalLM.__init__` in `model.py` currently has (line 59):

```python
BlockClass = LlamaBlock if cfg.model_type == "baseline" else RBFBlock
```

Replace this line in full with a dict dispatch:

```python
BlockClass = {
    "baseline": LlamaBlock,
    "rbf":      RBFBlock,
    "rational": RationalBlock,
}[cfg.model_type]
```

Note: the dict raises `KeyError` for unrecognized `model_type` values. This is intentional — it fails loudly rather than silently routing to the wrong block class. The prior `if/else` fell through to `RBFBlock` for unknown values; the dict is a deliberate improvement.

Also extend the import on line 6 of `model.py`:

```python
from rbf_ffn.models.transformer_block import LlamaBlock, RBFBlock, RationalBlock
```

Update the `CausalLM` class docstring to include:

```
"rational" → RationalBlock (RationalFFN)
```

---

## Config

### New field value

`model_type: "rational"` — no new config fields needed. `ffn_hidden` is already present and shared.

### New config file: `rbf_ffn/configs/rational_ffn.yaml`

Full contents (mirrors `baseline.yaml` with `model_type: rational` and adds the explicit `grad_accum_steps` field that `baseline.yaml` currently omits):

```yaml
model_type: rational
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

Note: `gate_variant`, `sigma_variant`, `K`, `centers`, `sigma_init`, and `sinkhorn_iters` are inert for `model_type: rational` but are present for a uniform config schema.

---

## Optimizer Groups

`RationalActivation.a` (shape `(4,)`) and `.b` (shape `(2,)`) are 1D → fall into AdamW under existing rule 4. No changes to `build_optimizer_groups`.

| Parameter | Shape | Group |
|---|---|---|
| `up_proj.weight` | `(ffn_hidden, d_model)` | Muon |
| `down_proj.weight` | `(d_model, ffn_hidden)` | Muon |
| `act.a` | `(4,)` | AdamW |
| `act.b` | `(2,)` | AdamW |

---

## Metrics & Experiment Dir

- `metrics.jsonl` rows contain: `epoch`, `train_loss`, `train_ppl`, `val_loss`, `val_ppl`, `epoch_time_s`, `effective_batch_size`. No sigma stats — `train.py` guards sigma collection with `if cfg.model_type == "rbf"`, which already excludes `"rational"` runs. No changes to `train.py` needed.
- Experiment dir: `get_experiment_dir` embeds `cfg.model_type`, producing e.g. `20260316_..._rational_G0_global_d256_K5`. The `gate_variant`, `sigma_variant`, and `K` segments are inert noise for `rational` runs (same accepted limitation as `baseline` runs). No code changes needed.

---

## File Changes

### New Files

```
rbf_ffn/
  models/
    rational_ffn.py          # both RationalActivation and RationalFFN defined here
  configs/
    rational_ffn.yaml        # model_type: rational (full contents above)
```

### Modified Files

```
rbf_ffn/
  models/
    transformer_block.py     # add RationalBlock class; add import:
                             #   from rbf_ffn.models.rational_ffn import RationalFFN
    model.py                 # extend import to include RationalBlock;
                             # replace if/else with dict dispatch;
                             # update CausalLM docstring
  config.py                  # update model_type comment: "baseline" | "rbf" | "rational"
                             # update ffn_hidden comment: used by SwiGLU and RationalFFN; ignored by RBF model
```

### New Tests

```
rbf_ffn/
  tests/
    test_rational_ffn.py     # RationalActivation, RationalFFN, RationalBlock unit tests
    test_model.py            # extend with rational optimizer-group test
```

---

## Tests

**`rbf_ffn/tests/test_rational_ffn.py`**

Imports:
```python
from rbf_ffn.models.rational_ffn import RationalActivation, RationalFFN
from rbf_ffn.models.transformer_block import RationalBlock
```

Uses constants consistent with the existing test suite: `B, N, D = 2, 16, 32`.

Config helper used throughout:

```python
def make_cfg():
    return RBFFFNConfig(
        d_model=D, n_heads=4, n_layers=2, seq_len=N,
        ffn_hidden=86, dropout=0.0, model_type="rational",
    )
```

(`n_layers=2`, `ffn_hidden=86` ≈ 8/3 × 32 — mirrors `test_model.py`'s convention.)

`RationalActivation` takes no constructor arguments; `a` and `b` init values are fixed constants in the implementation.

| Test | Assertion |
|---|---|
| `test_rational_activation_shape` | Input `(2, 16, 32)` → output shape `(2, 16, 32)` |
| `test_rational_activation_gradients` | `a.grad` and `b.grad` are non-None after `output.sum().backward()` |
| `test_rational_ffn_shape` | `RationalFFN(make_cfg()).forward(x)` output shape `(B, N, d_model)` |
| `test_rational_ffn_no_bias` | `up_proj.bias` and `down_proj.bias` are `None` |
| `test_rational_block_shape` | `RationalBlock(make_cfg()).forward(x)` returns `(B, N, d_model)` |

**`rbf_ffn/tests/test_model.py`** (extend existing file):

| Test | Construction | Assertion |
|---|---|---|
| `test_rational_params_in_adamw` | `make_model("rational")` (uses existing helper) | `id(act.a)` and `id(act.b)` appear in AdamW id-set AND are absent from Muon id-set |

`make_model("rational")` works with the existing helper once the dict dispatch replaces the `if/else` in `model.py`. Reach the activation params via `model.blocks[i].ffn.act.a` / `.b`; collect across all blocks.

---

## Ablation Matrix

| Config file | model_type | FFN |
|---|---|---|
| `baseline.yaml` | `baseline` | SwiGLU (SiLU gate) |
| `rational_ffn.yaml` | `rational` | RationalFFN (no gating) |
