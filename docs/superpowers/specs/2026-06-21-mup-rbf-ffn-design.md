# muP (Maximal Update Parameterization) for rbf_ffn

**Date:** 2026-06-21
**Status:** Design approved

---

## Motivation

The current `rbf_ffn` training loop uses a single flat `muon_lr` and `adamw_lr` tuned at `d_model=256`. When sweeping to larger widths the learning rates need manual retuning — features grow or shrink with width in standard parameterization (SP). muP fixes this: with three coordinated changes (init scaling, LR scaling, logit scaling) the optimal LR transfers exactly from the proxy width to any target width.

---

## Scope

Three files change; no new dependencies; backward compatible (`mup=False` is the default and is numerically identical to SP).

Out of scope: Kronecker-factored layers (`kronecker_mlp`, `kronecker_delta_mlp`) — muP with factored layers is a separate research question and those modules are not `nn.Linear` instances so `_apply_mup_init` skips them silently.

---

## Architecture

### 1. Config (`rbf_ffn/config.py`)

Three new fields added to `ModelConfig`:

```python
mup: bool = False
mup_base_width: int = 256    # proxy model width at which muon_lr/adamw_lr are tuned
mup_init_std: float = 0.02   # init std at base width
```

`mup_scale = mup_base_width / d_model`. At `d_model == mup_base_width`, `mup_scale == 1.0` so the base-width model is numerically identical to SP.

Validation in `__post_init__`: reject `mup=True` with `mup_base_width <= 0`.

### 2. Model init (`rbf_ffn/models/model.py`)

New top-level function:

```python
def _apply_mup_init(model: CausalLM, cfg: ModelConfig) -> None:
    std = cfg.mup_init_std * math.sqrt(cfg.mup_base_width / cfg.d_model)
    tied_id = id(model.token_embedding.weight)
    for module in model.modules():
        if isinstance(module, nn.Linear) and id(module.weight) != tied_id:
            nn.init.normal_(module.weight, mean=0.0, std=std)
```

Covers: Q/K/V/O projections, FFN up/gate/down, untied lm_head, LoRA A/B. Skips: tied embedding weight (input layer — init unchanged). Biases and norm weights are not touched.

Two additions at the end of `CausalLM.__init__`:

```python
self.mup_scale: float = (cfg.mup_base_width / cfg.d_model) if cfg.mup else 1.0
if cfg.mup:
    _apply_mup_init(self, cfg)
```

### 3. Logit scaling (`CausalLM.forward`)

Last line before return becomes:

```python
return self.lm_head(x) * self.mup_scale, hs
```

`torch.compile` folds the `* 1.0` away when muP is off. With tied embeddings the weight is the input-layer embedding; the `mup_scale` multiplier applied to the logits is the correct muP output-layer correction without untying.

### 4. Optimizer construction (`rbf_ffn/train.py`)

Replaces the current optimizer block:

```python
muon_params, adamw_params = build_optimizer_groups(model)
mup_scale = (cfg.mup_base_width / cfg.d_model) if cfg.mup else 1.0

muon = Muon(muon_params, lr=cfg.muon_lr * mup_scale, momentum=0.95)

if cfg.mup:
    emb_id = id(model.token_embedding.weight)
    adamw = AdamW([
        {"params": [p for p in adamw_params if id(p) != emb_id],
         "lr": cfg.adamw_lr * mup_scale},
        {"params": [p for p in adamw_params if id(p) == emb_id],
         "lr": cfg.adamw_lr},
    ], weight_decay=cfg.adamw_wd, betas=(0.9, 0.95))
else:
    adamw = AdamW(adamw_params, lr=cfg.adamw_lr,
                  weight_decay=cfg.adamw_wd, betas=(0.9, 0.95))
```

All non-embedding AdamW params (biases, RMSNorm weights, sigma_raw, etc.) receive `adamw_lr * mup_scale`. The embedding receives `adamw_lr` unchanged. `build_optimizer_groups` interface is unchanged.

`LambdaLR` wraps both optimizers as before — warmup/cosine schedule composes correctly with the muP base LR.

### 5. Experiment naming (`rbf_ffn/train.py`)

```python
if cfg.mup:
    norm_tags += f"_mup{cfg.mup_base_width}"
```

---

## Usage

To run a proxy model (tune LR here):

```yaml
d_model: 256
mup: true
mup_base_width: 256
muon_lr: 0.02      # tune this
adamw_lr: 3e-4     # tune this
```

To transfer to a wider model:

```yaml
d_model: 1024
mup: true
mup_base_width: 256   # same base
muon_lr: 0.02         # unchanged — transfers from proxy
adamw_lr: 3e-4        # unchanged
```

The effective Muon LR at 1024 width is `0.02 * (256/1024) = 0.005`.

---

## Files Changed

| File | Change |
|------|--------|
| `rbf_ffn/config.py` | 3 new fields + `__post_init__` validation |
| `rbf_ffn/models/model.py` | `_apply_mup_init()`, `CausalLM.__init__`, `CausalLM.forward` |
| `rbf_ffn/train.py` | optimizer construction, experiment naming |

`build_optimizer_groups` interface is unchanged; existing tests unaffected.
