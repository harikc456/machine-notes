# RBF-FFN

A drop-in transformer FFN replacement using Radial Basis Function kernel expansion, with four gate variants for ablation study.

**Architecture:** `LayerNorm → RBF → Gate → Down Projection`

The RBF layer expands each scalar feature across K static Gaussian centers, replacing the standard up-projection entirely.

---

## Setup

```bash
uv pip install -e .
```

---

## Running Experiments

Each experiment is driven by a YAML config file. Results are saved automatically to `rbf_ffn/experiments/<timestamp>_<variant>/`.

### Gate Ablations (G0–G2)

```bash
# G0 — element-wise self-gate (baseline)
python -m rbf_ffn.train --config rbf_ffn/configs/g0_baseline.yaml

# G1-A — cross-kernel mixing gate (RBF output → Linear → sigmoid)
python -m rbf_ffn.train --config rbf_ffn/configs/g1a_cross_kernel.yaml

# G1-B — input-driven gate (pre-RBF x → Linear → sigmoid)
python -m rbf_ffn.train --config rbf_ffn/configs/g1b_input_driven.yaml

# G2 — Sinkhorn aggregation (no gate, collapses K via doubly-stochastic weights)
python -m rbf_ffn.train --config rbf_ffn/configs/g2_sinkhorn.yaml
```

### σ Ablations (bandwidth)

```bash
# σ-B — per-center bandwidth (K learnable scalars)
python -m rbf_ffn.train --config rbf_ffn/configs/sigma_b_per_center.yaml

# σ-C — per-dim-per-center bandwidth (d_model × K learnable scalars)
python -m rbf_ffn.train --config rbf_ffn/configs/sigma_c_per_dim.yaml
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | required | Path to YAML config |
| `--n_epochs` | 5 | Number of training epochs |
| `--batch_size` | 32 | Batch size |

Example with custom epochs:
```bash
python -m rbf_ffn.train --config rbf_ffn/configs/g0_baseline.yaml --n_epochs 20
```

---

## Experiment Output

Each run creates a timestamped directory:

```
rbf_ffn/experiments/20260312_133237_G0_d64_K5/
  config.yaml      # exact config used (for reproducibility)
  metrics.jsonl    # one JSON line per epoch: {"epoch": 0, "loss": 1.23, "acc": 0.12}
```

---

## Config Fields

| Field | Default | Description |
|-------|---------|-------------|
| `gate_variant` | `G0` | One of `G0`, `G1A`, `G1B`, `G2` |
| `sigma_variant` | `global` | One of `global`, `per_center`, `per_dim` |
| `d_model` | 256 | Model dimension |
| `n_heads` | 8 | Attention heads |
| `n_layers` | 6 | Number of transformer blocks |
| `K` | 5 | Number of RBF centers |
| `centers` | `[-1,-0.5,0,0.5,1]` | Static RBF center positions |
| `sigma_init` | 0.5 | Initial bandwidth (= grid spacing) |
| `sinkhorn_iters` | 20 | Sinkhorn iterations (G2 only) |
| `dropout` | 0.1 | Attention dropout |

Custom configs can specify any subset of fields; unspecified fields use defaults.

---

## Tests

```bash
pytest rbf_ffn/tests/ -v
```

---

## Gate Variants

| ID | Gate input | Mechanism | Down proj input |
|----|-----------|-----------|-----------------|
| G0 | RBF output | `sigmoid(w ⊙ rbf + b) ⊙ rbf` | `d_model·K` |
| G1-A | RBF output | `Linear(d_model·K → d_model·K) → sigmoid → ⊙ rbf` | `d_model·K` |
| G1-B | Pre-RBF input `x` | `Linear(d_model → d_model·K) → sigmoid → ⊙ rbf` | `d_model·K` |
| G2 | RBF output | Sinkhorn over K → weighted sum | `d_model` |

G0 is the baseline. G2 is the most parameter-efficient (no gate weights, smaller down projection).
