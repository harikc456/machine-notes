# RBF-FFN: WikiText-103 Experiment Design

**Date:** 2026-03-12
**Status:** Design approved
**Extends:** `2026-03-12-rbf-ffn-design.md`

---

## Goal

Replace the toy random-data training loop with a rigorous ablation study on WikiText-103. The Llama transformer (RMSNorm + RoPE + SwiGLU) serves as the baseline. The only variable between baseline and RBF runs is the FFN module; all other architecture components are identical.

---

## Architecture

### Shared Attention Stack

Both baseline and RBF models use identical attention components:

- **RMSNorm** — `nn.RMSNorm(d_model, elementwise_affine=True)` (available since PyTorch 2.4). No mean-centering, only RMS scaling with learnable weight.
- **RotaryEmbedding** — RoPE applied to Q and K inside attention. `head_dim = d_model // n_heads` (derived in code, not a config field).
- **CausalSelfAttention** — multi-head self-attention with causal mask, RoPE, no bias in any projection. The causal mask is handled by passing `is_causal=True` to `F.scaled_dot_product_attention` inside `forward`. No explicit mask tensor is constructed or stored. Signature: `forward(self, x: Tensor) -> Tensor` where `x: (B, N, d_model)`.

### Baseline Block (`LlamaBlock` in `transformer_block.py`)

```python
def forward(self, x):
    x = x + self.attn(self.norm1(x))
    x = x + self.ffn(self.norm2(x))
    return x
```

**SwiGLUFFN** (`llama_ffn.py`):
```python
gate = F.silu(self.gate_proj(x))   # Linear(d_model, ffn_hidden, bias=False)
up   = self.up_proj(x)             # Linear(d_model, ffn_hidden, bias=False)
return self.down_proj(gate * up)   # Linear(ffn_hidden, d_model, bias=False)
```

`ffn_hidden = 688` (≈ 8/3 × d_model, standard Llama convention for d_model=256).

### RBF Block (`RBFBlock` in `transformer_block.py`)

```python
def forward(self, x):
    x = x + self.attn(self.norm1(x))
    x = x + self.ffn(self.norm2(x))   # norm2(x) is passed into RBFFFN
    return x
```

**Call chain for the FFN sub-layer:**
1. `self.norm2(x)` — outer pre-block RMSNorm normalises relative to the attention sublayer output
2. `RBFFFN.forward(normed_x)` — RBFFFN applies its own internal `self.norm` (`nn.RMSNorm`) to the already-normed input before the RBF expansion

**Double normalisation is intentional.** The internal `self.norm` has its own learnable (γ, β) and decouples the RBF input distribution from whatever scale `norm2` outputs. See the parent spec for full justification. Do not remove either norm.

**Change to `RBFFFN`:** Replace `self.norm = nn.LayerNorm(D)` with `self.norm = nn.RMSNorm(D)` in `__init__`. Additionally, change `self.down_proj = nn.Linear(down_in, D)` to `nn.Linear(down_in, D, bias=False)` to match the no-bias Llama convention and eliminate a confound between baseline and RBF variants. No other changes to `RBFFFN`.

### Block Selection in `model.py`

```python
BlockClass = LlamaBlock if cfg.model_type == "baseline" else RBFBlock
self.blocks = nn.ModuleList([BlockClass(cfg) for _ in range(cfg.n_layers)])
```

Both `LlamaBlock` and `RBFBlock` live in `transformer_block.py` and take `cfg: RBFFFNConfig` as their sole constructor argument.

### Full Model (`CausalLM` in `model.py`)

```
token_embedding  nn.Embedding(vocab_size, d_model)
  → N × Block    (LlamaBlock or RBFBlock)
  → RMSNorm(d_model)
  → LM head      nn.Linear(d_model, vocab_size, bias=False), weight-tied
```

**Weight tying:** After construction, `self.lm_head.weight = self.token_embedding.weight`.

**Optimizer deduplication:** When building parameter groups, iterate `model.named_parameters()` tracking `seen_ids = set()`; skip any `param` whose `id(param)` is already in `seen_ids` before applying the group rule.

### Model Scale (Tiny, ~18M parameters)

| d_model | n_heads | head_dim | n_layers | ffn_hidden | K | vocab_size | seq_len |
|---|---|---|---|---|---|---|---|
| 256 | 8 | 32 | 6 | 688 | 5 | 50257 | 512 |

---

## Data

**Dataset:** `load_dataset("wikitext", "wikitext-103-raw-v1")`. Each split has a `text` field. Filter entries where `text.strip() == ""`. Concatenate remaining entries with `"\n"`. Tokenise with `tiktoken.get_encoding("r50k_base")` (vocab_size = 50257; do not use `cl100k_base`). Chunk into sequences of exactly `seq_len` tokens, discarding trailing remainder.

**Cache:** `Path(__file__).parent / "data_cache"` (relative to `data.py`; created automatically; gitignored). Cache filenames embed `seq_len` to auto-invalidate on changes: `train_r50k_{seq_len}.pt`, `valid_r50k_{seq_len}.pt`, `test_r50k_{seq_len}.pt`.

**DataLoader:**

| Setting | Train | Val/Test |
|---|---|---|
| shuffle | True | False |
| drop_last | True | False |
| num_workers | 4 | 4 |
| pin_memory | True | True |

---

## Training

### Loss

Input: `tokens[:, :-1]`, target: `tokens[:, 1:]`. Token count per batch: `n_tokens = batch_size * (seq_len - 1)`.

```python
loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), reduction='mean')
loss_sum    += loss.item() * n_tokens
token_count += n_tokens
# end of epoch:
train_loss = loss_sum / token_count   # nats per token
train_ppl  = math.exp(train_loss)
```

Validation: same accumulation under `torch.no_grad()`.

### Optimiser

Rules applied in order (first match wins) after deduplication by `id(param)`:

1. If `"sigma_raw"` in parameter name → **AdamW** (explicit override regardless of shape)
2. Else if param is the token embedding weight → **AdamW**
3. Else if `param.ndim == 2` → **Muon**
4. Else → **AdamW**

Concrete assignments:

| Parameter | Shape | Group |
|---|---|---|
| Attn Q, K, V, O weights | (d_model, d_model) | Muon |
| SwiGLU gate_proj, up_proj | (ffn_hidden, d_model) | Muon |
| SwiGLU down_proj | (d_model, ffn_hidden) | Muon |
| RBF `down_proj.weight` | (d_model, d_model·K) | Muon |
| G1A `mix.weight` | (d_model·K, d_model·K) | Muon |
| G1B `proj.weight` | (d_model·K, d_model) | Muon |
| Token embedding / tied LM head | (vocab_size, d_model) | AdamW |
| RMSNorm weight vectors | (d_model,) | AdamW |
| `sigma_raw` — all variants | any shape | AdamW (rule 1) |
| G0 `w` — bare `nn.Parameter` | (d_model·K,) | AdamW |
| G0 `b` — bare `nn.Parameter` | (d_model·K,) | AdamW |

**Muon import:** `from torch.optim import Muon` (available in PyTorch ≥ 2.10). Instantiate as `Muon(muon_params, lr=cfg.muon_lr, momentum=0.95)`.

| Optimiser | lr | Other |
|---|---|---|
| Muon | 0.02 | momentum=0.95 |
| AdamW | 3e-4 | wd=0.1, β=(0.9, 0.95) |

### LR Schedule

Two `LambdaLR` schedulers sharing the same function, stepped together once per training step:

```python
def lr_lambda(step: int) -> float:
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
```

- `steps_per_epoch = len(train_dataloader)` (use DataLoader length directly)
- `total_steps = n_epochs * steps_per_epoch`
- `warmup_steps = int(warmup_ratio * total_steps)`

### Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
```

Single call over all parameters (true global norm). Called after `loss.backward()`, before both `.step()` calls.

### Seed

```python
torch.manual_seed(cfg.seed)   # before model construction
g = torch.Generator()
g.manual_seed(cfg.seed)
train_loader = DataLoader(..., generator=g)
```

---

## Experiment Artifacts

Directory:
```
rbf_ffn/experiments/<timestamp>_<model_type>_<gate_variant>_<sigma_variant>_d<d_model>_K<K>/
```

Baseline example: `20260312_133237_baseline_G0_global_d256_K5` — `gate_variant=G0` and `sigma_variant=global` are the config defaults written into `baseline.yaml`; they are ignored by the model but present in the directory name for a uniform naming scheme.

Files:

| File | Contents |
|---|---|
| `config.yaml` | Exact config copy |
| `metrics.jsonl` | One JSON line per epoch (see below) |
| `checkpoint_best.pt` | State at lowest `val_loss`; `best_val_loss` initialised to `float("inf")` |
| `checkpoint_final.pt` | State at end of training |

**Checkpoint format:**
```python
{
    "model":           model.state_dict(),
    "optimizer_muon":  muon.state_dict(),
    "optimizer_adamw": adamw.state_dict(),
    "scheduler_muon":  sched_muon.state_dict(),
    "scheduler_adamw": sched_adamw.state_dict(),
    "epoch":           int,
    "val_loss":        float,
    "val_ppl":         float,
}
```

**`metrics.jsonl` per-epoch fields:**
- Always present: `epoch`, `train_loss`, `train_ppl`, `val_loss`, `val_ppl`, `epoch_time_s`
- When `cfg.model_type == "rbf"`: `sigma_mean`, `sigma_std` (computed over all `softplus(sigma_raw)` values; `sigma_std=0.0` for `global` variant which has a scalar `sigma_raw`). When `cfg.model_type == "baseline"`: these fields are absent.

---

## File Structure

### New Files

```
rbf_ffn/
  models/
    attention.py        # RMSNorm, RotaryEmbedding, CausalSelfAttention
    llama_ffn.py        # SwiGLUFFN
    model.py            # CausalLM
  data.py               # WikiText-103 loader, tokeniser, cache, Dataset
  data_cache/           # gitignored; .pt cache files
  configs/
    baseline.yaml       # new
```

### Modified Files

```
rbf_ffn/
  models/
    rbf_ffn.py          # LayerNorm → RMSNorm; down_proj bias=False
    transformer_block.py # Add LlamaBlock; update RBFBlock to use CausalSelfAttention
  config.py             # Add new fields (see below)
  train.py              # Full rewrite
  configs/g0_baseline.yaml
  configs/g1a_cross_kernel.yaml
  configs/g1b_input_driven.yaml
  configs/g2_sinkhorn.yaml
  configs/sigma_b_per_center.yaml
  configs/sigma_c_per_dim.yaml
```

### New Tests

```
rbf_ffn/tests/
  test_attention.py   # RMSNorm, RoPE, causal mask
  test_llama_ffn.py   # SwiGLU shape, gradient flow, bias=False
  test_model.py       # CausalLM shape, param count, weight tying,
                      # optimizer group membership
```

---

## Config Fields

**Implementation order constraint:** Update `RBFFFNConfig` to add the new fields first, then update the YAML files. Since `load_config` raises on unknown keys, adding keys to YAMLs before adding them to the dataclass will break loading. Do not update YAMLs until the dataclass change is committed.

Pre-existing fields (unchanged): `d_model`, `n_heads`, `n_layers`, `K`, `centers`, `sigma_init`, `sigma_variant`, `gate_variant`, `sinkhorn_iters`, `dropout`, `seq_len`, `vocab_size`.

New fields:

| Field | Default | Description |
|---|---|---|
| `model_type` | `"rbf"` | `"baseline"` or `"rbf"` |
| `ffn_hidden` | 688 | SwiGLU hidden dim; silently ignored by RBF model |
| `seed` | 42 | Global random seed |
| `n_epochs` | 10 | Training epochs |
| `batch_size` | 32 | Sequences per batch |
| `muon_lr` | 0.02 | Muon learning rate |
| `adamw_lr` | 3e-4 | AdamW learning rate |
| `adamw_wd` | 0.1 | AdamW weight decay |
| `warmup_ratio` | 0.02 | Fraction of total steps for LR warmup |
| `grad_clip` | 1.0 | Global gradient norm clip |

All YAML files (existing and new) should include all fields explicitly, even those matching defaults, to make each config self-documenting.

---

## Ablation Matrix

| Config file | model_type | gate_variant | sigma_variant |
|---|---|---|---|
| `baseline.yaml` | `baseline` | `G0` | `global` |
| `g0_baseline.yaml` | `rbf` | `G0` | `global` |
| `g1a_cross_kernel.yaml` | `rbf` | `G1A` | `global` |
| `g1b_input_driven.yaml` | `rbf` | `G1B` | `global` |
| `g2_sinkhorn.yaml` | `rbf` | `G2` | `global` |
| `sigma_b_per_center.yaml` | `rbf` | `G0` | `per_center` |
| `sigma_c_per_dim.yaml` | `rbf` | `G0` | `per_dim` |
