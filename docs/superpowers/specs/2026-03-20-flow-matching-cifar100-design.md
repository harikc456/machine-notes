# Flow Matching on CIFAR-100 — Design Spec

**Date:** 2026-03-20
**Status:** Approved

## Goal

Implement Rectified Flow (Liu et al. 2022) on CIFAR-100 with a DiT (Diffusion Transformer) vector field network and classifier-free guidance (CFG). The project follows existing repo conventions: dataclass configs loaded from YAML, timestamped experiment dirs, JSONL metrics logging, and matplotlib plots.

---

## Module Structure

```
flow_matching/
├── __init__.py
├── config.py          # FlowConfig dataclass + load_config
├── data.py            # CIFAR-100 DataLoaders (train aug + val)
├── model.py           # DiT: PatchEmbed, DiTBlock, DiT (vector field network) + build_optimizer_groups
├── train.py           # Training loop, JSONL logging, plot generation
├── sample.py          # Euler sampler + save sample grid
└── configs/
    ├── dit_cfg.yaml   # Default config (AdamW, CFG, CIFAR-100)
    ├── dit_muon.yaml  # Muon variant
    └── dit_small.yaml # Smaller model for quick smoke tests
```

Experiment outputs land in `flow_matching/experiments/<timestamp>_<run_name>/`:
- `config.yaml` — run config copy
- `metrics.jsonl` — one JSON line per log step: `step, train_loss, lr`
- `plot.png` — training loss curve
- `samples_step_<N>.png` — grid of generated samples saved every `sample_every` steps
- `ckpt.pt` — model state dict + step, saved every `save_every` steps

**CLI:**
```
python -m flow_matching.train --config flow_matching/configs/dit_cfg.yaml
python -m flow_matching.train --config flow_matching/configs/dit_cfg.yaml --n_steps 50000
python -m flow_matching.sample --config flow_matching/configs/dit_cfg.yaml --checkpoint <exp_dir>/ckpt.pt --cfg_scale 3.0
```

`train.py` exposes `--config` (required) and `--n_steps` (optional int override). `sample.py` exposes `--config`, `--checkpoint`, `--cfg_scale`, `--n_steps_euler` (default 100), and `--out` (output PNG path).

---

## Data

CIFAR-100: 50k train / 10k val images, 100 classes, 32×32 RGB.

- **Train augmentations:** `RandomCrop(32, pad=4)` + `RandomHorizontalFlip` + normalize with CIFAR-100 mean/std
- **Val augmentations:** normalize only
- `data.py` mirrors the `hyvit/data/cifar.py` pattern adapted for CIFAR-100

---

## Training Objective (Rectified Flow)

At each training step:
1. Sample clean image `x_1` and Gaussian noise `x_0 ~ N(0, I)`
2. Sample time `t ~ Uniform(0, 1)`
3. Construct `x_t = (1 - t) * x_0 + t * x_1`
4. Target velocity: `v = x_1 - x_0` (constant along the straight-line path)
5. Loss: `MSE(model(x_t, t, class) - v)`

**Classifier-free guidance (CFG):**
- During training, replace class label with null token (index 100) with probability `p_uncond` (default `0.1`)
- At inference, guided velocity: `v_guided = v_uncond + cfg_scale * (v_cond - v_uncond)` via doubled batch

---

## Model Architecture

### PatchEmbed

Splits 32×32×3 images into non-overlapping patches of size `patch_size=4`, yielding 64 tokens. Implemented as `Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)`. Patch embed weights are excluded from the Muon group (same convention as `grokking`).

### Time & Class Conditioning (adaLN-Zero)

Each DiT block uses adaptive layer norm (adaLN-Zero, Peebles & Xiao 2023):
- Time embedding: sinusoidal → linear projection to `d_model`
- Class embedding: `nn.Embedding(101, d_model)` where index 100 is the learned null token
- Conditioning signal: `c = time_embed(t) + class_embed(y)`
- MLP maps `c` to scale/shift/gate parameters for attention and FFN sublayers
- Time and class embedding weights are excluded from the Muon group

### DiTBlock

Pre-norm transformer block: `LayerNorm → Self-Attention → LayerNorm → FFN` with adaLN-Zero gates on both sublayers. No cross-attention, no causal masking.

### DiT (full model)

```
PatchEmbed → [DiTBlock × n_layers] → LayerNorm → Linear(d_model, patch_size² × 3) → unpatchify
```

Output shape: `(B, 3, 32, 32)` — predicted velocity field.

### Default Hyperparameters

| Param | Default (`dit_cfg`) | Small (`dit_small`) |
|---|---|---|
| `d_model` | 384 | 192 |
| `n_heads` | 6 | 3 |
| `n_layers` | 12 | 6 |
| `patch_size` | 4 | 4 |
| `mlp_ratio` | 4.0 | 4.0 |
| `dropout` | 0.0 | 0.0 |

---

## Training Loop

Step-based (not epoch-based), consistent with `grokking`:

1. Sample batch `(x_1, y)` from cycling train loader
2. Sample `x_0 ~ N(0, I)`, `t ~ Uniform(0, 1)`
3. Drop class labels → null token (index 100) with probability `p_uncond`
4. Compute `x_t`, forward pass, MSE loss against `v = x_1 - x_0`
5. Backward, grad clip, optimizer step, scheduler step
6. Every `log_every` steps: log `step, train_loss, lr` to JSONL and print
7. Every `sample_every` steps: run Euler sampler, save 10×10 sample grid
8. Every `save_every` steps: save `ckpt.pt`

---

## Euler Sampler (`sample.py`)

```
x = N(0, I)
for t in linspace(1, 0, n_steps_euler+1)[:-1]:   # integrate noise → data
    dt = -1 / n_steps_euler
    v  = model(x, t, class, cfg_scale)             # CFG doubled-batch
    x  = x + v * dt
```

CFG is applied via the doubled-batch trick (conditional + unconditional in one forward pass).

---

## Optimizers

Same pattern as `grokking/model.py`:

**`build_optimizer_groups(model)`** splits parameters:
- **Muon group:** `param.ndim == 2` matrices, excluding patch embed conv weights and time/class embedding weights (identified by tensor identity)
- **AdamW group:** all remaining parameters (embeddings, biases, norms, scalars)

**AdamW mode:** single AdamW over all parameters; `lr=adamw_lr`, `weight_decay=weight_decay`, `betas=(0.9, 0.999)`.

**Muon mode:** Muon for 2D matrices, AdamW for rest; separate `muon_lr` and `adamw_lr`; shared `LambdaLR` cosine + warmup scheduler.

---

## Config Schema (`FlowConfig`)

```python
@dataclass
class FlowConfig:
    # Data
    data_root: str = "data/"
    seed: int = 42

    # Model
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 12
    patch_size: int = 4
    mlp_ratio: float = 4.0

    # Flow matching
    p_uncond: float = 0.1

    # Training
    n_steps: int = 200_000
    batch_size: int = 128
    optimizer: str = "adamw"       # adamw | muon
    adamw_lr: float = 1e-4
    muon_lr: float = 0.02          # only used when optimizer=muon
    weight_decay: float = 0.0
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0
    log_every: int = 100
    sample_every: int = 5_000
    save_every: int = 10_000

    # Sampling
    n_steps_euler: int = 100
    cfg_scale: float = 3.0
```

`betas` are hard-coded to `(0.9, 0.999)` in `train.py` and not YAML-serialised.

---

## Testing Strategy

Tests live in `flow_matching/tests/`:

- **`test_config.py`** — `FlowConfig` loads from YAML, rejects unknown keys, validates `optimizer` field
- **`test_data.py`** — `build_loaders` returns correct batch shapes `(B, 3, 32, 32)` and label range `[0, 99]`
- **`test_model.py`** — forward pass produces output shape `(B, 3, 32, 32)`; null class token (index 100) accepted; `build_optimizer_groups` splits params correctly (no patch embed or time/class embed weights in Muon group)
- **`test_train.py`** — smoke run for 5 steps with tiny config (`d_model=64, n_layers=2, n_heads=2, batch_size=4`) completes without error
- **`test_sample.py`** — Euler sampler produces output shape `(B, 3, 32, 32)`; CFG doubled-batch forward pass returns correct shape

---

## Success Criteria

1. `python -m flow_matching.train --config flow_matching/configs/dit_cfg.yaml` runs without error, producing `metrics.jsonl`, `plot.png`, `ckpt.pt`, and sample grids in a timestamped experiment dir.
2. Training loss decreases monotonically over the first 1k steps.
3. `python -m flow_matching.sample` generates a valid PNG grid from a saved checkpoint.
4. Both `optimizer: adamw` and `optimizer: muon` complete a 10-step smoke run without error.
5. All tests pass.
6. Module is importable as `flow_matching` and follows repo style conventions.
