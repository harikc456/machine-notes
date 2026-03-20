# Grokking — Modular Arithmetic Training Script

**Date:** 2026-03-20
**Status:** Draft

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

Experiment outputs are saved to `grokking/experiments/<timestamp>_<op>_<optimizer>/`, where the path is anchored to `Path(__file__).parent / "experiments"` (i.e. relative to `train.py`, matching `rbf_ffn` convention):
- `config.yaml` — copy of the run config
- `metrics.jsonl` — one JSON line per log step with train/val loss and accuracy
- `plot.png` — dual-panel loss + accuracy curve

**CLI invocation:**
```
python -m grokking.train --config grokking/configs/adamw_add.yaml
python -m grokking.train --config grokking/configs/muon_add.yaml --n_steps 100000
```

`train.py` exposes two CLI arguments: `--config` (required, path to YAML) and `--n_steps` (optional int, overrides the config value). No other config fields are CLI-overridable; all other changes must be made in the YAML.

---

## Data

- **Universe:** all `p²` pairs `(a, b)` for `a, b ∈ {0, …, p-1}`
- **Label:** result of the chosen operation mod p
- **Train fraction:** configurable, default `0.4` (40 % forces memorisation before generalisation, following the paper)
- **Input format:** token sequence `[a, op_token, b, =]` — always 4 tokens; sequence length is a **fixed constant (`seq_len = 4`)** in `model.py`, not user-configurable
- **Supported operations:**
  - `add`: `(a + b) % p`
  - `sub`: `(a - b) % p`
  - `mul`: `(a * b) % p`
  - `div`: `(a * pow(b, -1, p)) % p` (modular inverse; only valid pairs where `b ≠ 0`)
  - `x2_plus_xy_plus_y2`: `(a**2 + a*b + b**2) % p`
- **Seeded split:** reproducible train/val split via `seed`

---

## Model

A standard transformer encoder (no causal masking; input is fixed-length):

| Hyperparameter | Default | Notes |
|---|---|---|
| `d_model` | 128 | embedding dimension |
| `n_heads` | 4 | attention heads |
| `n_layers` | 2 | transformer blocks |
| `dropout` | 0.0 | paper uses 0 |
| `p` | 97 | modulus (prime) |

**Vocab size:** `p + 2` (p digit tokens + 1 op token + 1 equals token).

**Weight tying:** the output head (`lm_head`) is **not weight-tied** to the input embedding. It is a separate `nn.Linear(d_model, p, bias=False)` layer.

Only the logit at the `=` position (index 3, the last input token) is used for cross-entropy loss. Loss is classification over `p` classes.

---

## Training

### Optimisers

The optimiser is selected via `optimizer: adamw | muon` in config.

**AdamW mode:**
- Single AdamW over all parameters
- `lr = adamw_lr`, `weight_decay = weight_decay`, `betas = (0.9, 0.98)` (betas hard-coded in `train.py`, not in config)
- Note: `betas` second value is `0.98`, matching Power et al. 2022 — this intentionally deviates from the `rbf_ffn` default of `0.95`

**Muon mode:**
- Muon for all 2-D weight matrices (criterion: `param.ndim == 2`), **excluding** the token embedding matrix (identified by tensor identity, same as `rbf_ffn.build_optimizer_groups`)
- AdamW for all remaining parameters (embeddings, biases, scalars), with `weight_decay = weight_decay` and `betas = (0.9, 0.98)` — `weight_decay` applies to this AdamW sub-group in the same way as AdamW-only mode
- Since the head is not weight-tied, `lm_head.weight` (2-D) goes to Muon
- Separate learning rates: `muon_lr` for Muon, `adamw_lr` for AdamW sub-group (see config schema)

Weight decay on AdamW (both modes) is the critical hyperparameter for inducing grokking.

### Schedule

Cosine decay with linear warmup (warmup ratio configurable, default `0.01`), applied to both optimisers in Muon mode via a shared `LambdaLR`.

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
- Validation runs over the full val set at each log step

### Plot

Single PNG saved at end of training with two vertically stacked panels sharing the x-axis (step):
1. **Loss** — train and val curves vs step (log-scale y)
2. **Accuracy** — train and val curves vs step (linear 0–1)

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
    adamw_lr: float = 1e-3
    muon_lr: float = 0.02           # only used when optimizer=muon
    weight_decay: float = 1.0
    warmup_ratio: float = 0.01
    grad_clip: float = 1.0
    log_every: int = 10
```

`betas` are hard-coded to `(0.9, 0.98)` in `train.py` and are not YAML-serialised.

---

## Success Criteria

1. With `optimizer: adamw` and `weight_decay=1.0`, training accuracy reaches ~100% before validation accuracy does (memorisation phase), and validation accuracy then jumps sharply toward ~100% in a later phase — the classic grokking signature visible in the plot.
2. A Muon run (`optimizer: muon`) completes all `n_steps` without error, producing a valid `metrics.jsonl` with one entry per `log_every` steps and a `plot.png` in the experiment directory.
3. All five supported operations (`add`, `sub`, `mul`, `div`, `x2_plus_xy_plus_y2`) complete a short smoke run (e.g. 100 steps) without error.
4. Module is importable as `grokking` and follows `rbf_ffn` style conventions (config dataclass, `load_config`, experiment dir anchored to `__file__`).
