# Medusa-1 Training — Design Spec

**Date:** 2026-06-18
**Status:** Approved

---

## Overview

A standalone `medusa/` module that trains K Medusa-1 decoding heads on top of Gemma 4 E2B's last hidden state, using the RyokoAI/ShareGPT52K dataset. The module is completely independent of `mtp_draft/`. Training only — no inference or tree-attention.

---

## Architecture

### `medusa/models/medusa_model.py` — `MedusaModel`

Input: `h` of shape `(B, d_model)` — the teacher's last hidden state at an answer token position.
Output: `logits` of shape `(B, K, vocab)`.

Each head k applies a single-layer residual MLP (paper formula):

```
logits_k = W2_k · SiLU(W1_k · h + h)
```

- `W1_k`: `Linear(d_model, d_model, bias=False)` — initialized to **zero** so heads start as identity mappings.
- `W2_k`: `Linear(d_model, vocab, bias=False)` — initialized from the **frozen teacher LM head weight** (tied initialization, not tied at runtime).
- Both `W1_k` and `W2_k` are trainable.

The model-agnostic boundary is `d_model` — everything else is independent of the teacher architecture.

Config fields that determine model structure: `d_model`, `vocab_size`, `n_heads` (K).

No frozen buffers inside the model (teacher is run separately at cache time).

---

## Data Pipeline

### `medusa/cache.py` — Phase 1

1. Load RyokoAI/ShareGPT52K from HuggingFace.
2. Apply Gemma 4 E2B chat template to each conversation, tokenize.
3. Run teacher (`AutoModelForCausalLM`) in `eval()` + `no_grad()`, collect `last_hidden_state`.
4. At each answer token position `t`, extract `hidden[t]` and build targets `[y_{t+1}, ..., y_{t+K}]` (padding with `-100` where the answer is too short for K tokens ahead).
5. Int8 quantize each hidden vector (per-vector scale + zero-point), write sharded `.pt` files to `cache_dir`.

Each shard is a list of dicts: `{"hidden": Tensor(n_pos, d_model, dtype=int8), "scale": Tensor(n_pos), "zero": Tensor(n_pos), "targets": Tensor(n_pos, K, dtype=int64)}`.

Config fields: `cache_dir`, `cache_shard_size`, `n_heads` (K, needed to build target windows), `max_answer_len` (cap on tokens extracted per turn).

### `medusa/data.py` — `MedusaDataset`

- Iterates over sharded `.pt` files, flattens to per-position items.
- Dequantizes hidden on access: `h_float = (h_int8.float() - zero) * scale`.
- Returns `(hidden: Tensor(d_model), targets: Tensor(K))`.
- No context IDs — Medusa heads only see the hidden state.

---

## Training

### `medusa/train.py` — Phase 2

Loss (Medusa-1, distance-weighted CE):

```
L = Σ_{k=0}^{K-1}  λ^k · CE(logits[:, k, :], y_{t+k+1})
```

- `λ = lambda_decay` (default 0.8).
- `-100` targets are ignored by `F.cross_entropy(ignore_index=-100)`.
- Optimizer: AdamW with cosine LR schedule and linear warmup.
- Gradient clipping: `grad_clip` (default 1.0).
- Best val-loss checkpoint saved to `checkpoints/best.pt`. Checkpoint contains head weights only (no LM head init weight — that's reloaded from teacher at load time).

---

## Config

### `medusa/config.py` — `MedusaConfig`

| Field | Default | Notes |
|---|---|---|
| `n_heads` | 5 | Number of Medusa heads (K) |
| `d_model` | 1536 | Teacher hidden size (Gemma 4 E2B) |
| `vocab_size` | — | Set from teacher tokenizer |
| `lambda_decay` | 0.8 | Per-head loss weight decay |
| `teacher_model_id` | `google/gemma-4-e2b-it` | Used in cache phase |
| `dataset_id` | `RyokoAI/ShareGPT52K` | HuggingFace dataset |
| `max_answer_len` | 256 | Max tokens per turn for target extraction |
| `cache_dir` | `medusa/cache` | Shard storage |
| `cache_shard_size` | 5000 | Items per shard |
| `lr` | 3e-4 | AdamW LR |
| `weight_decay` | 0.1 | AdamW weight decay |
| `grad_clip` | 1.0 | Gradient clipping norm |
| `batch_size` | 32 | Training batch size |
| `n_epochs` | 3 | Training epochs |
| `warmup_steps` | 200 | LR warmup |
| `seed` | 42 | RNG seed |
| `val_split` | 0.05 | Fraction of data held out for validation |

---

## File Layout

```
medusa/
├── cache.py              # Phase 1: extract + int8-quantize last hidden states
├── data.py               # MedusaDataset over cached shards
├── train.py              # Phase 2: Medusa-1 training loop
├── config.py             # MedusaConfig dataclass + load_config
├── models/
│   ├── __init__.py
│   └── medusa_model.py   # K single-layer residual MLP heads
├── configs/
│   └── default.yaml
├── checkpoints/          # best.pt written here during training
├── __init__.py
└── tests/
    └── test_medusa_model.py
```

---

## Usage

**Step 1 — extract teacher features**

```bash
python -m medusa.cache --config medusa/configs/default.yaml
```

**Step 2 — train the heads**

```bash
python -m medusa.train --config medusa/configs/default.yaml
```

**Tests**

```bash
pytest medusa/tests/
```

---

## Out of Scope

- Medusa-2 (joint backbone fine-tuning)
- Inference / tree attention / candidate verification
- Integration with `mtp_draft/`
