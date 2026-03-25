# Grokking: Modular Arithmetic with Transformers

Reproduces the "grokking" phenomenon from [Power et al. 2022](https://arxiv.org/abs/2201.02177) — where a small transformer first memorizes a modular arithmetic task, then suddenly generalizes long after training loss plateaus.

## Setup

```bash
pip install torch pyyaml matplotlib pytest
```

## Run

```bash
# AdamW baseline
python -m grokking.train --config grokking/configs/adamw_add.yaml

# Muon optimizer variant
python -m grokking.train --config grokking/configs/muon_add.yaml

# Override steps
python -m grokking.train --config grokking/configs/adamw_add.yaml --n_steps 100000
```

Outputs land in `grokking/experiments/<timestamp>_<operation>_<optimizer>/`:
- `metrics.jsonl` — per-step train/val loss and accuracy
- `plot.png` — dual-panel loss + accuracy curves
- `config.yaml` — copy of the run config

## Configuration

Edit the YAML files or create your own. Key fields:

| Field | Default | Description |
|---|---|---|
| `p` | 97 | Prime modulus |
| `operation` | `add` | One of: `add`, `sub`, `mul`, `div`, `x2_plus_xy_plus_y2` |
| `train_fraction` | 0.4 | Fraction of pairs used for training |
| `n_steps` | 50000 | Total training steps |
| `batch_size` | 512 | Batch size |
| `optimizer` | `adamw` | `adamw` or `muon` |
| `adamw_lr` | 1e-3 | AdamW learning rate |
| `muon_lr` | 0.02 | Muon learning rate (matrix params only) |
| `weight_decay` | 1.0 | L2 penalty — critical for grokking to occur |
| `d_model` | 128 | Embedding dimension |
| `n_heads` | 4 | Attention heads |
| `n_layers` | 2 | Transformer blocks |

## Model

A 2-layer encoder-only transformer trained to predict `(a ○ b) mod p`. Input is the fixed 4-token sequence `[a, op, b, =]`; the logits at the `=` position are used for prediction. Vocab size is `p + 2`.

## Tests

```bash
python -m pytest grokking/tests/ -v
```

## What to expect

With `train_fraction=0.4` and `weight_decay=1.0`, the model will:
1. Rapidly overfit to training data (high train acc, near-zero val acc)
2. Much later — sometimes 10-50k steps after train loss saturates — val acc jumps sharply toward 100%

This delayed generalization is grokking.
