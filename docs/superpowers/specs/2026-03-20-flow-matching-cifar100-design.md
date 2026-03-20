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

Experiment outputs land in `flow_matching/experiments/<timestamp>_<optimizer>/`:
- `config.yaml` — run config copy
- `metrics.jsonl` — one JSON line per log step: `step, train_loss, lr`
- `plot.png` — training loss curve
- `samples_step_<N>.png` — grid of generated samples saved every `sample_every` steps
- `ckpt.pt` — model state dict + step, saved every `save_every` steps

The experiment dir name is `<YYYYMMDD_HHMMSS_ffffff>_<cfg.optimizer>`, mirroring `grokking`'s `<timestamp>_<operation>_<optimizer>` pattern. There is no `run_name` config field; the optimizer name serves as the run identifier.

**CLI:**
```
python -m flow_matching.train --config flow_matching/configs/dit_cfg.yaml
python -m flow_matching.train --config flow_matching/configs/dit_cfg.yaml --n_steps 50000
python -m flow_matching.sample --config flow_matching/configs/dit_cfg.yaml --checkpoint <exp_dir>/ckpt.pt --cfg_scale 3.0
```

`train.py` exposes `--config` (required) and `--n_steps` (optional int override). `sample.py` exposes `--config`, `--checkpoint`, `--cfg_scale`, `--n_steps_euler`, and `--out` (output PNG path). CLI values override the corresponding config fields when provided; otherwise the config file values are used. `cfg_scale` and `n_steps_euler` live in `FlowConfig` (not a separate sample config) so that in-training sample grids during `train.py` use consistent defaults without a second config file.

---

## Data

CIFAR-100: 50k train / 10k val images, 100 classes, 32×32 RGB.

- **Train augmentations:** `RandomCrop(32, pad=4)` + `RandomHorizontalFlip` + normalize with CIFAR-100 mean/std
- **Val augmentations:** normalize only
- `data.py` mirrors the `hyvit/data/cifar.py` pattern adapted for CIFAR-100

**`build_loaders(cfg: FlowConfig, num_workers: int = 4) -> tuple[DataLoader, DataLoader]`**
Returns `(train_loader, val_loader)`. Train loader shuffles with `drop_last=True`; val loader does not shuffle.

---

## Training Objective (Rectified Flow)

Convention: `t=0` is pure noise, `t=1` is pure data.

At each training step:
1. Sample clean image `x_1` and Gaussian noise `x_0 ~ N(0, I)`
2. Sample time `t ~ Uniform(0, 1)`
3. Construct `x_t = (1 - t) * x_0 + t * x_1`
4. Target velocity: `v = x_1 - x_0` (constant along the straight-line path, pointing noise → data)
5. Loss: `MSE(model(x_t, t, class) - v)`

**Classifier-free guidance (CFG):**
- During training, replace class label with null token (index 100) with probability `p_uncond` (default `0.1`)
- At inference, the guided velocity is computed externally in `sample.py` (see Euler Sampler section)

---

## Model Architecture

### PatchEmbed

Splits 32×32×3 images into non-overlapping patches of size `patch_size=4`, yielding 64 tokens (8×8 grid). Implemented as `Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)` followed by flatten + transpose to `(B, 64, d_model)`. The conv weight (`patch_embed.proj.weight`) is excluded from the Muon group by tensor identity (same convention as `grokking`).

**Positional embeddings:** Fixed 2D sinusoidal positional embeddings of shape `(1, 64, d_model)` are added to the patch tokens after `PatchEmbed`, before the first `DiTBlock`. These are buffers (not learnable parameters), so they do not appear in any optimizer group.

### Time & Class Conditioning (adaLN-Zero)

Each DiT block uses adaptive layer norm (adaLN-Zero, Peebles & Xiao 2023):
- Time embedding: sinusoidal → linear projection to `d_model`
- Class embedding: `nn.Embedding(101, d_model)` where index 100 is the learned null token for CFG
- Conditioning signal: `c = time_embed(t) + class_embed(y)`
- A shared adaLN MLP (one per block) maps `c → 6 * d_model`, producing six vectors: `shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp`
- The final linear layer of the adaLN MLP is **zero-initialized** (the "Zero" in adaLN-Zero), so blocks act as identity at initialization

**Step-by-step DiTBlock forward:**
```
shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = adaln_mlp(c).chunk(6, dim=-1)

normed = (1 + scale_msa) * norm1(x) + shift_msa    # modulate before attention
x = x + gate_msa * attn(normed)                     # gate residual

normed = (1 + scale_mlp) * norm2(x) + shift_mlp    # modulate before FFN
x = x + gate_mlp * ffn(normed)                      # gate residual
```

The `(1 + scale)` formulation ensures that at initialization (zero-init MLP → scale=0, shift=0, gate=0) the LayerNorm output passes through unchanged, and the residual contribution is zero — blocks act as identity at init.

The time linear projection weight, class embedding weight, and all adaLN MLP weights (including their 2D linear layers) are **excluded from the Muon group** and go to AdamW. These are conditioning-path weights; applying Muon's Newton-Schulz orthogonalization to them is inappropriate.

### DiTBlock

Pre-norm transformer block: `LayerNorm → Self-Attention → LayerNorm → FFN` with adaLN-Zero modulation on both sublayers. No cross-attention, no causal masking.

**Forward signature:** `DiTBlock.forward(x: Tensor, c: Tensor) -> Tensor`
- `x`: `(B, 64, d_model)` — patch tokens
- `c`: `(B, d_model)` — conditioning signal (time + class)

### DiT (full model)

```
PatchEmbed → add pos_embed → [DiTBlock × n_layers] → LayerNorm → Linear(d_model, patch_size²×3) → unpatchify
```

**unpatchify:** reshape `(B, 64, 48)` → `(B, 8, 8, 4, 4, 3)` → permute → `(B, 3, 32, 32)`.
Concretely: `x.reshape(B, 8, 8, 4, 4, 3).permute(0, 5, 1, 3, 2, 4).contiguous().reshape(B, 3, 32, 32)`.
`.contiguous()` is required before the final `reshape` because `permute` produces a non-contiguous tensor.

Output shape: `(B, 3, 32, 32)` — predicted velocity field.

**Forward signature:** `DiT.forward(x: Tensor, t: Tensor, y: Tensor) -> Tensor`
- `x`: `(B, 3, 32, 32)` — noisy image at time t
- `t`: `(B,)` — time values in [0, 1]
- `y`: `(B,)` — class indices in [0, 100] (100 = null token)

### Default Hyperparameters

| Param | Default (`dit_cfg`) | Small (`dit_small`) |
|---|---|---|
| `d_model` | 384 | 192 |
| `n_heads` | 6 | 3 |
| `n_layers` | 12 | 6 |
| `patch_size` | 4 | 4 |
| `mlp_ratio` | 4.0 | 4.0 |
| `dropout` | 0.0 | 0.0 |

`dropout` is a config field (default `0.0`) applied to attention and FFN inside each DiTBlock.

---

## Training Loop

Step-based (not epoch-based), consistent with `grokking`:

1. Sample batch `(x_1, y)` from cycling train loader
2. Sample `x_0 ~ N(0, I)`, `t ~ Uniform(0, 1)`
3. Drop class labels → null token (index 100) with probability `p_uncond`
4. Compute `x_t = (1-t)*x_0 + t*x_1`, forward pass `model(x_t, t, y)`, MSE loss against `v = x_1 - x_0`
5. Backward, grad clip, optimizer step, scheduler step
6. Every `log_every` steps: log `step, train_loss, lr` to JSONL and print. In both AdamW and Muon mode, `lr` is the current AdamW optimizer's learning rate (i.e., `adamw_lr * lr_lambda(step)`). In Muon mode this reflects the shared schedule applied to the AdamW sub-group.
7. Every `sample_every` steps: switch to `model.eval()`, run Euler sampler (from `sample.py`) with `y = torch.arange(100)` (one sample per class, giving a 10×10 grid of 100 images), using `cfg.n_steps_euler` and `cfg.cfg_scale`, save sample grid, then switch back to `model.train()`
8. Every `save_every` steps: save `ckpt.pt` containing `{"model": state_dict, "step": step}`

---

## Euler Sampler (`sample.py`)

Convention: integrate forward from `t=0` (noise) to `t=1` (data).

```python
# sample.py — standalone CFG Euler sampler
def euler_sample(model, y, cfg_scale, n_steps, device):
    # y: (B,) class indices in [0, 99]
    B = y.shape[0]
    x = torch.randn(B, 3, 32, 32, device=device)
    dt = 1.0 / n_steps
    y_null = torch.full_like(y, 100)   # null token

    for i in range(n_steps):
        t = i / n_steps                # scalar t in [0, 1)
        t_batch = torch.full((B,), t, device=device)

        # Two forward passes; or doubled batch for efficiency
        v_cond   = model(x, t_batch, y)
        v_uncond = model(x, t_batch, y_null)
        v = v_uncond + cfg_scale * (v_cond - v_uncond)

        x = x + v * dt
    return x
```

CFG is applied **externally in `sample.py`** via two separate forward passes (or a doubled batch `torch.cat([y, y_null])` for efficiency). The model's `forward` signature does **not** accept `cfg_scale` — it is a plain `(x, t, y) -> v` function.

---

## Optimizers

Same pattern as `grokking/model.py`:

**`build_optimizer_groups(model)`** splits parameters:
- **Muon group:** `param.ndim == 2` matrices that are **not** in the excluded set
- **AdamW group:** all remaining parameters (embeddings, biases, norms, adaLN MLP weights, scalars)
- **Excluded from Muon** (go to AdamW) — evaluation order is: name-prefix check first, then ndim check:
  1. If the parameter name starts with `"time_embed."`, `"class_embed."`, or contains `"adaln_mlp."` → AdamW (name-prefix exclusion)
  2. Else if `param.ndim == 2` → Muon (standard 2D matrix rule)
  3. Else → AdamW (biases, norms, conv weights, etc.)

  `patch_embed.proj.weight` is ndim=4 and has no matching name prefix; it falls through to rule 3 (AdamW) without any explicit check needed.

**AdamW mode:** single AdamW over all parameters; `lr=adamw_lr`, `weight_decay=weight_decay`, `betas=(0.9, 0.98)` (matching repo convention from `grokking`).

**Muon mode:** Muon for eligible 2D matrices, AdamW for rest; separate `muon_lr` and `adamw_lr`. The AdamW sub-optimizer uses `betas=(0.9, 0.98)` and `weight_decay=cfg.weight_decay`, matching the pure-AdamW mode. A shared `lr_fn` (same warmup + cosine lambda as AdamW mode) is passed to `LambdaLR(muon_opt, lr_fn)` and `LambdaLR(adamw_opt, lr_fn)` independently — each optimizer scales the lambda by its own base LR (`muon_lr` and `adamw_lr` respectively), identical to the `grokking` pattern.

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
    dropout: float = 0.0

    # Flow matching
    p_uncond: float = 0.1

    # Training
    n_steps: int = 200_000
    batch_size: int = 128
    optimizer: str = "adamw"       # adamw | muon
    adamw_lr: float = 1e-4
    muon_lr: float = 0.02          # only used when optimizer=muon
    weight_decay: float = 0.0     # 0.0 is appropriate for generative models; grokking uses 1.0 for memorisation pressure
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0
    log_every: int = 100
    sample_every: int = 5_000
    save_every: int = 10_000

    # Sampling
    n_steps_euler: int = 100
    cfg_scale: float = 3.0

    def __post_init__(self) -> None:
        valid_opts = {"adamw", "muon"}
        if self.optimizer not in valid_opts:
            raise ValueError(f"optimizer must be one of {valid_opts}, got {self.optimizer!r}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if not (0.0 < self.p_uncond < 1.0):
            raise ValueError(f"p_uncond must be in (0, 1), got {self.p_uncond}")
```

`betas` are hard-coded to `(0.9, 0.98)` in `train.py` (matching `grokking` convention) and are not YAML-serialised.

`load_config(path: str | Path) -> FlowConfig` mirrors `grokking/config.py` exactly: reads YAML with `yaml.safe_load`, rejects unknown keys with `ValueError`, applies dataclass defaults for missing keys.

---

## Testing Strategy

Tests live in `flow_matching/tests/`:

- **`test_config.py`** — `FlowConfig` loads from YAML, rejects unknown keys, `__post_init__` raises on invalid `optimizer`
- **`test_data.py`** — `build_loaders` returns correct batch shapes `(B, 3, 32, 32)` and label range `[0, 99]`
- **`test_model.py`** — forward pass `model(x, t, y)` produces output shape `(B, 3, 32, 32)`; null class token (index 100) accepted; `build_optimizer_groups` places no adaLN MLP or embedding weights in the Muon group
- **`test_train.py`** — smoke run for 5 steps with tiny config (`d_model=64, n_layers=2, n_heads=2, batch_size=4`) completes without error and writes `ckpt.pt` at `save_every` steps
- **`test_sample.py`** — `euler_sample` with 2 steps produces output shape `(B, 3, 32, 32)`; two-pass CFG combination produces correct shape

---

## Success Criteria

1. `python -m flow_matching.train --config flow_matching/configs/dit_cfg.yaml` runs without error, producing `metrics.jsonl`, `plot.png`, `ckpt.pt`, and sample grids in a timestamped experiment dir.
2. Training loss shows a clear downward trend over the first 1k steps (not necessarily monotonic step-to-step).
3. `python -m flow_matching.sample` generates a valid PNG grid from a saved checkpoint.
4. Both `optimizer: adamw` and `optimizer: muon` complete a 10-step smoke run without error.
5. All tests pass.
6. Module is importable as `flow_matching` and follows repo style conventions.
