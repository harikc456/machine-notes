# Grokking — Modular Arithmetic Training Script

**Date:** 2026-03-20
**Status:** Approved

## Goal

Reproduce the grokking phenomenon (Power et al., 2022) where a transformer first memorises modular arithmetic then suddenly generalises. Support multiple operations and both AdamW and Muon optimisers to study how the choice of optimiser affects the onset of grokking.

---

## Module Structure

```
grokking/
├── __init__.py
├── config.py             # GrokConfig dataclass + load_config
├── data.py               # Dataset + DataLoader for modular arithmetic
├── model.py              # Transformer (embeddings + blocks + head)
├── train.py              # Training loop, metric logging, plot generation
└── configs/
    ├── adamw_add.yaml    # AdamW baseline, addition mod p
    └── muon_add.yaml     # Muon variant, addition mod p
```

Experiment outputs are saved to `grokking/experiments/<timestamp>_<op>_<optimizer>/`:
- `config.yaml` — copy of the run config
- `metrics.jsonl` — one JSON line per log step with train/val loss and accuracy
- `plot.png` — dual-panel loss + accuracy curve

---

## Data

- **Universe:** all `p²` pairs `(a, b)` for `a, b ∈ {0, …, p-1}`
- **Label:** result of the chosen operation mod p
- **Train fraction:** configurable, default `0.4` (40 % forces memorisation before generalisation, following the paper)
- **Input format:** token sequence `[a, op_token, b, =]`; the model predicts the token after `=` (classification over p classes)
- **Supported operations:**
  - `add`: `(a + b) % p`
  - `sub`: `(a - b) % p`
  - `mul`: `(a * b) % p`
  - `div`: `(a * b⁻¹) % p` (modular inverse, only defined where b ≠ 0)
  - `x2_plus_xy_plus_y2`: `(a² + ab + b²) % p`
- **Seeded split:** reproducible train/val split via `seed`

---

## Model

A standard transformer encoder (no causal masking needed; the input is a fixed-length sequence):

| Hyperparameter | Default | Notes |
|---|---|---|
| `d_model` | 128 | embedding dimension |
| `n_heads` | 4 | attention heads |
| `n_layers` | 2 | transformer blocks |
| `dropout` | 0.0 | paper uses 0 |
| `p` | 97 | modulus (prime) |

**Vocab:** `p` digit tokens + 1 op token + 1 equals token → size `p + 2`.

Only the logit at the `=` position (last input position) is used for cross-entropy loss. Loss is classification over `p` classes.

---

## Training

### Optimisers

| Config key | Optimiser | Notes |
|---|---|---|
| `optimizer: adamw` | AdamW | `lr=1e-3`, `weight_decay=1.0`, `betas=(0.9, 0.98)` |
| `optimizer: muon` | Muon (matrix params) + AdamW (scalars/biases/embeddings) | same split strategy as `rbf_ffn` |

Weight decay on AdamW is the critical hyperparameter for inducing grokking. Muon runs use AdamW for embedding and head parameters (non-matrix tensors).

### Schedule

Cosine decay with linear warmup (warmup ratio configurable, default `0.01`).

### Key hyperparameters

| Param | Default |
|---|---|
| `n_steps` | 50 000 |
| `batch_size` | 512 |
| `log_every` | 10 |
| `grad_clip` | 1.0 |
| `seed` | 42 |

### Logging (per log step)

- `step`, `train_loss`, `val_loss`, `train_acc`, `val_acc`
- Written as JSONL to `metrics.jsonl`

### Plot

Single PNG saved at end of training with two vertically stacked panels:
1. **Loss** — train and val curves vs step (log-scale y recommended)
2. **Accuracy** — train and val curves vs step (linear 0–1)

Both panels share the x-axis (step).

---

## Config Schema (`GrokConfig`)

```python
@dataclass
class GrokConfig:
    # Data
    p: int = 97
    operation: str = "add"          # add | sub | mul | div | x2_plus_xy_plus_y2
    train_fraction: float = 0.4
    seed: int = 42

    # Model
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.0

    # Training
    n_steps: int = 50_000
    batch_size: int = 512
    optimizer: str = "adamw"        # adamw | muon
    lr: float = 1e-3
    weight_decay: float = 1.0
    betas: tuple = (0.9, 0.98)
    warmup_ratio: float = 0.01
    grad_clip: float = 1.0
    log_every: int = 10
```

---

## Success Criteria

1. With AdamW + `weight_decay=1.0`, training accuracy reaches ~100% within a few thousand steps while validation accuracy stays low, then validation accuracy jumps sharply to ~100% much later — the classic grokking signature.
2. Muon run produces a comparable or different grokking curve, enabling comparison.
3. Plots clearly show the two-phase behaviour (memorisation → generalisation).
4. All supported operations run without error.
5. Module integrates cleanly into the existing project structure (importable, consistent style).
