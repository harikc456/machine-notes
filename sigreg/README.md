# SIGReg Transformer Experiments

Experiments with transformers trained under an auxiliary **SIGReg** (Signature Regularisation) loss that forces hidden-state distributions toward a standard Gaussian.

## Architecture

Unlike the Llama-style blocks in `rbf_ffn`, these transformers have **no residual connections and no normalisation layers by default**:

```
token_embedding → [attn → ffn] × N → lm_head
```

Each block simply computes `x = ffn(attn(x))`. The network must route signal through weights alone, making hidden-state distributions meaningful targets for regularisation.

## SIGReg Losses

Two complementary losses are available, selected via `sigreg_loss_type` in config:

| Type | What it matches | How |
|------|-----------------|-----|
| `strong` | All moments (Maximum Entropy Cloud) | Empirical CF vs. Gaussian CF, integrated over t ∈ [-5, 5] |
| `weak` | 2nd moment only (Spherical Cloud) | Frobenius distance of sample covariance to identity |
| `both` | Strong + weak, equal weight | Mean of the two above |

The total training loss is:

```
total_loss = ce_loss + sigreg_weight * sigreg_loss
```

The SIGReg loss is computed over flattened token representations `(B*T, d_model)` collected from the configured layers. Both losses use a random sketch of dimension `sigreg_sketch_dim` (default 64) to keep cost manageable at large `d_model`.

## Setup

Install dependencies from the project root (shared with `rbf_ffn`):

```bash
uv sync
```

The WikiText-103 dataset is shared with `rbf_ffn`. On first run the BPE-65536 tokenizer is trained on the WikiText-103 training split (~5 minutes), then each split is tokenised and cached to `rbf_ffn/data_cache/`. Subsequent runs load from cache instantly.

## Running Training

```bash
# Strong loss baseline (ECF → Gaussian, all layers)
python -m sigreg.train --config sigreg/configs/baseline.yaml

# Weak loss only (covariance → identity, all layers)
python -m sigreg.train --config sigreg/configs/weak_loss.yaml

# Override number of epochs
python -m sigreg.train --config sigreg/configs/baseline.yaml --n_epochs 10
```

Experiment outputs (metrics, config copy, checkpoints) are written to `sigreg/experiments/<timestamp>_plain_sigreg_<type>_d<dim>/`.

## Config Reference

| Field | Default | Description |
|---|---|---|
| `d_model` | 256 | Model dimension |
| `n_heads` | 8 | Number of attention heads |
| `n_layers` | 6 | Number of transformer blocks |
| `ffn_hidden` | 688 | SwiGLU hidden dimension |
| `dropout` | 0.0 | Dropout probability (typically 0 — no norms to pair with) |
| `qk_norm` | `true` | QK normalisation (prevents attention entropy collapse without LayerNorm) |
| `use_residual` | `false` | Add skip connections around attn and ffn |
| `norm_type` | `"none"` | Norm layer: `"none"` \| `"rmsnorm"` \| `"layernorm"` |
| `seq_len` | 512 | Sequence length |
| `vocab_size` | 65536 | Vocabulary size — matches the custom BPE-65536 tokeniser trained in `sigreg/data.py` |
| `tie_embeddings` | `true` | Tie lm_head weights to token embedding |
| `sigreg_loss_type` | `strong` | `"strong"` \| `"weak"` \| `"both"` |
| `sigreg_weight` | 0.1 | λ — scales the auxiliary loss relative to CE |
| `sigreg_sketch_dim` | 64 | Random projection dimension for ECF / covariance estimator |
| `sigreg_layers` | `all` | `"all"` (every block output) \| `"last"` (final block only) |
| `lr` | 3e-4 | AdamW learning rate (Muon LR is set to `lr × 6.67`) |
| `weight_decay` | 0.1 | AdamW weight decay |
| `warmup_ratio` | 0.02 | Fraction of optimizer steps used for linear LR warmup |
| `grad_clip` | 1.0 | Gradient norm clip value |
| `grad_accum_steps` | 1 | Gradient accumulation steps (1 = no accumulation) |

## Tuning Tips

- **`sigreg_weight`**: Start at 0.1 for `strong`, 0.01 for `weak` (the Frobenius norm is larger in magnitude).
- **`sigreg_layers: last`**: Cheaper and less disruptive than `all`. Try this first if CE loss degrades.
- **`qk_norm: true`**: Recommended — without residual connections and RMSNorm, attention logits can blow up.
- **`dropout: 0.0`**: Dropout is of limited benefit without normalisation layers; leave it off unless overfitting.

## File Layout

```
sigreg/
├── config.py           # SIGRegConfig dataclass and YAML loader
├── data.py             # BPE-65536 tokenizer (lazy build) + WikiText-103 dataloaders
├── losses.py           # sigreg_strong_loss, sigreg_weak_loss, sigreg_loss
├── train.py            # Training loop
├── models/
│   ├── block.py        # TransformerBlock (CausalSelfAttention + SwiGLUFFN, optional residual/norm)
│   └── model.py        # SIGRegCausalLM, build_optimizer_groups
├── configs/
│   ├── baseline.yaml   # Strong loss, λ=0.1
│   └── weak_loss.yaml  # Weak loss, λ=0.01
└── tests/
    └── test_data.py    # Unit tests for data.py (offline, no network)
```
