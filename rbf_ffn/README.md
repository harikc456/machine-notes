# RBF-FFN

A transformer FFN architecture exploration comparing RBF kernels, rational activations, gating mechanisms, and normalization strategies on WikiText-103.

**Current best result:** SwiGLU + QK-norm + weight-norm → **58.16 val PPL** (−23.1% vs vanilla SwiGLU baseline)

---

## Setup

```bash
uv pip install -e .
```

---

## Running Experiments

Each experiment is driven by a YAML config file. Results are saved automatically to `rbf_ffn/experiments/<timestamp>_<name>/`.

### Normalization Ablations (recommended starting point)

```bash
# Best overall: SwiGLU + QK-norm + weight-norm
python -m rbf_ffn.train --config rbf_ffn/configs/baseline_qk_norm.yaml  # + edit to add weight_norm

# Baseline + weight-norm only
python -m rbf_ffn.train --config rbf_ffn/configs/baseline_weight_norm.yaml

# Baseline + adaptive weight-norm (depth-based, phase-aware)
python -m rbf_ffn.train --config rbf_ffn/configs/baseline_adaptive_weight_norm.yaml

# PFDRationalGLU + QK-norm + weight-norm
python -m rbf_ffn.train --config rbf_ffn/configs/pfd_rationalglu_qk_norm_weight_norm.yaml
```

### Rational Activation Variants

```bash
# Vanilla baseline (SwiGLU reference)
python -m rbf_ffn.train --config rbf_ffn/configs/baseline.yaml

# RationalGLU — best FFN activation quality/cost trade-off
python -m rbf_ffn.train --config rbf_ffn/configs/rationalglu_ffn.yaml

# PFDRationalGLU — best FFN activation overall (at 3 epochs, no weight norm)
python -m rbf_ffn.train --config rbf_ffn/configs/pfd_rationalglu_ffn.yaml

# Non-gated rational (ablation)
python -m rbf_ffn.train --config rbf_ffn/configs/rational_ffn.yaml

# FirstOrderPFDRational — 33% fewer FFN params, near-SwiGLU accuracy
python -m rbf_ffn.train --config rbf_ffn/configs/first_order_pfd_rational_ffn.yaml
```

### With QK-norm

```bash
python -m rbf_ffn.train --config rbf_ffn/configs/baseline_qk_norm.yaml
python -m rbf_ffn.train --config rbf_ffn/configs/rationalglu_qk_norm.yaml
python -m rbf_ffn.train --config rbf_ffn/configs/pfd_rationalglu_qk_norm.yaml
```

### Override epochs at runtime

```bash
python -m rbf_ffn.train --config rbf_ffn/configs/pfd_rationalglu_ffn.yaml --n_epochs 10
```

---

## Experiment Output

Each run creates a timestamped directory:

```
rbf_ffn/experiments/20260324_164546_baseline_qknorm_wnorm_d256/
  config.yaml      # exact config used (reproducibility)
  metrics.jsonl    # one JSON line per epoch: {"epoch": 0, "train_loss": ..., "val_ppl": ...}
  checkpoint_best.pt
  checkpoint_final.pt
```

---

## Config Fields

### Model

| Field | Default | Description |
|-------|---------|-------------|
| `model_type` | `"baseline"` | `"baseline"` \| `"rational"` \| `"rationalglu"` \| `"pfd_rational"` \| `"pfd_rationalglu"` \| `"first_order_pfd_rational"` |
| `d_model` | 256 | Model dimension |
| `n_heads` | 8 | Attention heads |
| `n_layers` | 6 | Number of transformer blocks |
| `ffn_hidden` | 688 | FFN hidden dim (SwiGLU / rational variants) |
| `pfd_n` | 4 | Number of partial fraction terms (PFD variants only) |
| `dropout` | 0.1 | Attention dropout |
| `seq_len` | 512 | Sequence length |
| `vocab_size` | 50257 | Vocabulary size |

### Normalization

| Field | Default | Description |
|-------|---------|-------------|
| `qk_norm` | `false` | Enable QK normalization in attention (consistent −0.5–0.9 ppl gain) |
| `linear_weight_norm` | `false` | Normalize linear weight rows to L2 norm after each optimizer step |
| `linear_weight_norm_value` | `2.0` | Target L2 norm per output neuron |
| `activation_norm` | `false` | Normalize rational/PFD activation coefficients (slightly harmful with weight_norm) |
| `adaptive_weight_norm` | `false` | Depth-based weight norm: linearly decreasing target from early to late layers |
| `adaptive_norm_early` | `2.5` | Target norm at layer 0 |
| `adaptive_norm_late` | `1.2` | Target norm at layer L-1 (must be ≥ 1.0) |
| `adaptive_norm_gamma` | `0.3` | Max phase-correction magnitude (EMA-based) |
| `adaptive_norm_beta` | `5.0` | Tanh sensitivity to gap derivative |
| `adaptive_norm_alpha` | `0.9` | EMA smoothing factor for log-gap |

### Training

| Field | Default | Description |
|-------|---------|-------------|
| `seed` | 42 | Random seed |
| `n_epochs` | 10 | Number of training epochs |
| `batch_size` | 32 | Batch size |
| `muon_lr` | 0.02 | Learning rate for Muon optimizer (2D weight matrices) |
| `adamw_lr` | 3e-4 | Learning rate for AdamW (biases, embeddings, 1D params) |
| `adamw_wd` | 0.1 | AdamW weight decay |
| `warmup_ratio` | 0.02 | Fraction of steps for linear LR warmup |
| `grad_clip` | 1.0 | Gradient norm clipping |
| `grad_accum_steps` | 1 | Mini-batches per optimizer step (1 = no accumulation) |

Custom configs can specify any subset of fields; unspecified fields use defaults.

---

## Results Summary (3-epoch runs, WikiText-103, d_model=256)

### Normalization ablations

| Variant | Val PPL (ep 2) | Δ vs SwiGLU |
|---------|---------------|-------------|
| **SwiGLU + QK-norm + weight-norm** | **58.16** | **−23.1%** |
| SwiGLU + weight-norm | 58.97 | −22.1% |
| PFDRationalGLU + QK-norm + weight-norm | 58.91 | −22.2% |
| SwiGLU + QK-norm | 75.14 | −0.7% |
| RationalGLU + QK-norm | 73.51 | −2.9% |
| PFDRationalGLU + QK-norm | 72.25 | −4.5% |

> Weight normalization (−21.8 ppl) is the dominant improvement — it eclipses all FFN activation variant gains.

### FFN activation variants (no norm additions)

| Variant | Val PPL (ep 2) | Δ vs SwiGLU | Time/epoch |
|---------|---------------|-------------|-----------|
| **PFDRationalGLU** | **73.00** | **−3.5%** | ~1975s (+60%) |
| RationalGLU | 74.37 | −1.7% | ~1424s (+15%) |
| **Baseline (SwiGLU)** | **75.68** | — | ~1234s |
| FirstOrderPFDRational | 76.77 | +1.4% | ~2029s |
| Rational (non-gated) | 78.38 | +3.6% | ~1357s |
| RBF G1-B (input-driven) | 81.62 | +7.8% | ~1691s |
| RBF G1-A (cross-kernel) | 83.56 | +10.4% | ~1994s |
| RBF G0 (element-wise) | 92.70 | +22.4% | ~2294s |
| RBF G2 (Sinkhorn) | 110.28 | +45.7% | ~2771s |

---

## Tests

```bash
pytest rbf_ffn/tests/ -v
```

---

## Model Types

| `model_type` | Architecture | Notes |
|---|---|---|
| `baseline` | SwiGLU FFN (LLaMA-style) | Reference; apply `linear_weight_norm` for best results |
| `rationalglu` | Learnable rational gate × linear value | Best quality/cost activation variant (+36 params) |
| `pfd_rationalglu` | PFD rational gate × linear value | Best activation overall; +60% training overhead |
| `rational` | Linear up → rational activation → linear down | No gating; weaker than SwiGLU |
| `pfd_rational` | Linear up → PFD rational activation → linear down | No gating variant of PFD |
| `first_order_pfd_rational` | Shared projection, sin(u+φ) gate | 33% fewer FFN params; near-SwiGLU at 3 epochs |

---

## Configs Reference

| Config file | model_type | Norm additions |
|---|---|---|
| `baseline.yaml` | baseline | — |
| `baseline_qk_norm.yaml` | baseline | qk_norm |
| `baseline_weight_norm.yaml` | baseline | weight_norm |
| `baseline_qk_norm.yaml` + edits | baseline | qk_norm + weight_norm (best overall) |
| `baseline_adaptive_weight_norm.yaml` | baseline | qk_norm + adaptive weight_norm |
| `rationalglu_ffn.yaml` | rationalglu | — |
| `rationalglu_qk_norm.yaml` | rationalglu | qk_norm |
| `pfd_rationalglu_ffn.yaml` | pfd_rationalglu | — |
| `pfd_rationalglu_qk_norm.yaml` | pfd_rationalglu | qk_norm |
| `pfd_rationalglu_qk_norm_weight_norm.yaml` | pfd_rationalglu | qk_norm + weight_norm |
| `rational_ffn.yaml` | rational | — |
| `pfd_rational_ffn.yaml` | pfd_rational | — |
| `first_order_pfd_rational_ffn.yaml` | first_order_pfd_rational | — |

For detailed results, methodology, and analysis see `OVERVIEW.md` and `experiments/analysis.md`.
