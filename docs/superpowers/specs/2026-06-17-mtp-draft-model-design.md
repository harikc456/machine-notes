# MTP Draft Model Design

**Date**: 2026-06-17
**Status**: Approved

## Overview

A Multi-Token Prediction (MTP) draft model for speculative decoding that combines
ideas from Medusa (parallel independent heads) and EAGLE (teacher feature-level
autoregression). The draft model is a lightweight cross-attention decoder (<50M
trainable params) that uses Gemma 4 E2b's intermediate hidden states as queries and
the context token embeddings as keys/values. Draft tokens are predicted in parallel
with no assumed relation between them; the position offset `i` is injected as a
sinusoidal step embedding (diffusion-style). A single LM head (the frozen teacher LM
head + trainable LoRA) is shared across all draft positions. Tree candidates are
assembled at inference time via relative log-prob thresholding and verified by the
teacher in one tree-attention forward pass.

Dataset: `hotpotqa/hotpot_qa` (HotpotQA). Hardware target: 16 GB GPU, 16 GB RAM,
15 GB disk.

---

## Architecture

### High-Level Data Flow

```
Teacher Gemma 4 E2b (frozen during draft training)
┌──────────────────────────────────────────────────┐
│  embed  │  layer_low  │  layer_mid  │  layer_high │
└──────────────────────────────────────────────────┘
     │           │              │              │
  frozen       h_low          h_mid          h_high
   emb           └──────────────┬─────────────┘
     │                    KromHC fusion (trained)
     │                          │  Q_fused  (B, d_draft)
     │                          │
     │          + step_emb(i)   │  ← sinusoidal + MLP, one per i in 1..max_draft
     │                          │  Q  (B, max_draft, d_draft)
     │                          │
context_ids ──► ctx_proj ───────┤  K, V  (B, seq_len, d_draft)
(frozen embed)  (trained)       │
                        ┌───────┴──────────┐
                        │  n_blocks ×      │
                        │  CrossAttnBlock  │  KVShared attention + SwiGLU FFN
                        └───────┬──────────┘
                                │  (B, max_draft, d_draft)
                           out_proj (trained)   d_draft → d_teacher (2048)
                                │
                    Frozen Gemma LM head + LoRA A,B (rank 16)
                                │
                         logits  (B, max_draft, vocab)
```

### Cross-Attention Block

Each block is a standard pre-norm cross-attention + SwiGLU FFN:

```
x = x + CrossAttn(Q=x, K=ctx, V=ctx)   # KVShared: single kv_proj, K=V
x = x + SwiGLU_FFN(x)
```

`cross_attn_block.py` implements a bespoke `CrossAttnBlock` where Q is projected
from the draft query tensor and K, V are both projected from the context tensor via
a single shared `kv_proj` (borrowing the K=V sharing idea from `KVSharedAttention`
in `rbf_ffn`, but with separate query and context inputs — not a direct reuse of
that class). No causal mask is applied between the `max_draft` query positions —
they are fully independent.

XSA (Exclusive Self-Attention orthogonalisation) is optional and controlled by a
config flag `use_xsa: bool = False`.

### KromHC Multi-Layer Fusion

Uses the head-mixer concept from `rbf_ffn/models/head_mixer.py` (not the full
`KromHCWrapper`, which wraps whole transformer blocks). Takes the three teacher
hidden states `(h_low, h_mid, h_high)`, each of shape `(B, d_teacher)`, and
produces a single `(B, d_draft)` fused vector. Concretely:

1. Per-layer linear: `d_teacher → d_draft` (three separate projections, one per layer).
2. The three projected vectors are stacked to `(B, 3, d_draft)` and treated as
   "heads"; the KromHC head-mixer computes a learned weighted combination across
   the 3 pseudo-heads to produce the final `(B, d_draft)` output.

### Step Embedding

Position offset `i ∈ {1, …, max_draft}` is embedded as:

```
step_emb(i) = MLP(sinusoidal_embed(i, dim=d_draft))
```

where `sinusoidal_embed` uses the standard DDPM/transformer sinusoidal formula and
the MLP is a two-layer network (d_draft → 4·d_draft → d_draft, SiLU activation).
The result is added to `Q_fused` for each draft position before the cross-attention
blocks, producing `Q_i = Q_fused + step_emb(i)`.

### LoRA LM Head

The teacher's LM head weight `W ∈ R^{vocab × d_teacher}` is frozen. A LoRA adapter
adds `B·A` where `A ∈ R^{r × d_teacher}`, `B ∈ R^{vocab × r}`, `r=16` (configurable
via `lora_rank`). The forward pass is:

```
logits = x @ (W + B @ A).T
```

`A` is initialised with kaiming uniform; `B` is initialised to zero so the adapter
starts as a no-op.

### Parameter Budget

| Component | Params | Trainable |
|---|---|---|
| Teacher embedding (256k × 2048) | 524M | No |
| Teacher LM head (2048 × 256k) | 524M | No |
| LoRA on LM head (r=16) | 131k | Yes |
| KromHC fusion (3 × 2048→512 + mixer) | ~3.4M | Yes |
| ctx_proj (2048→512) | 1.0M | Yes |
| Step embedding + MLP | ~0.3M | Yes |
| `n_blocks` × CrossAttnBlock (default 4) | ~15.6M | Yes |
| out_proj (512→2048) | 1.0M | Yes |
| **Trainable total (defaults)** | **~21M** | Yes |

`n_blocks` and `ffn_hidden` are configurable; the above uses `n_blocks=4`,
`ffn_hidden=1366` (SwiGLU 4/3× rule applied to d_draft=512).

---

## Configuration

```python
@dataclass
class MTPConfig:
    # Draft model dimensions
    d_draft: int = 512
    n_blocks: int = 4
    ffn_hidden: int = 1366        # SwiGLU hidden dim; set to 0 to auto-compute as 8/3 * d_draft
    n_heads: int = 8
    n_kv_heads: int = 1           # 1 = MQA (KVShared); matches rbf_ffn n_kv_heads convention
    dropout: float = 0.0
    use_xsa: bool = False         # XSA orthogonalisation in cross-attn

    # Teacher
    teacher_model_id: str = "google/gemma-4-e2b-it"
    teacher_layers: list[int] = field(default_factory=lambda: [4, 9, 17])
    d_teacher: int = 2048

    # Training
    max_draft: int = 8
    lambda_decay: float = 0.8     # per-position loss weight: λ^(i-1)
    lora_rank: int = 16
    max_prompt_len: int = 256

    # Inference
    tau: float = 2.0              # relative log-prob threshold for tree pruning
    max_tree_nodes: int = 256     # cap on total candidate paths in tree

    # Optimiser
    lr: float = 3e-4
    batch_size: int = 16
    n_epochs: int = 3
    warmup_steps: int = 200
    seed: int = 42

    # Data / cache
    cache_dir: str = "mtp_draft/cache"
    cache_n_answer_positions: int = 8   # anchor positions cached per example
```

---

## Data Pipeline

### HotpotQA Prompt Format

```
Question: {question}\n\nContext:\n{paragraph_1}\n...\n{paragraph_10}\n\nAnswer:
```

Tokenised with the Gemma tokenizer. Prompt truncated to `max_prompt_len=256` tokens
(context paragraphs truncated first, question preserved). The final token of the
formatted prompt (before any answer token) is the **handoff position**.

### Phase 1 — Feature Extraction (cache.py)

Runs once. Teacher loaded on GPU, `batch_size=4`, `torch.no_grad()`.

For each example:
- Forward pass through frozen Gemma 4 E2b up to the last needed layer.
- Extract hidden states at `teacher_layers = {4, 9, 17}` for positions
  `[handoff, handoff+1, ..., handoff + cache_n_answer_positions - 1]`.
- Quantise to int8 (per-tensor dynamic quantisation; store scale + zero-point).
- Write to sharded `.pt` files (5k examples per shard) under `cache_dir/`.

**Disk budget:**
```
90k × 8 positions × 3 layers × 2048 dims × 1 byte (int8) ≈ 4.4 GB
Gemma weights (HuggingFace cache):                         ≈ 4.0 GB
Draft checkpoints + code + HotpotQA raw:                   ≈ 0.6 GB
Total:                                                     ≈ 9.0 GB  (< 15 GB)
```

GPU peak during extraction: ~5 GB (Gemma 4 GB + activations).

### Phase 2 — Draft Model Training (train.py)

Teacher NOT loaded. Cache shards streamed from disk.

For each `(example, anchor index j ∈ 0..cache_n_answer_positions-1)`:
- `Q_raw  = dequantise(cache[example, j, :, :])` → `(3, d_teacher)` bf16
- `tokens = prompt_ids[:handoff + j]`             → context for K, V
- `targets = token_ids[handoff+j+1 : handoff+j+1+max_draft]` → supervision

GPU peak during training: ~2 GB (draft model + frozen embed/LM head + optimiser states).

---

## Training Objective

Single forward pass produces logits for all `max_draft` positions simultaneously.

```
L = Σ_{i=1}^{max_draft}  λ^(i-1) · CrossEntropy(logits[:, i-1, :], targets[:, i-1])
```

`λ=0.8` by default. Positions where `targets` would exceed the answer length are
masked from the loss.

Optimiser: AdamW for all trainable parameters (LoRA, KromHC, cross-attn blocks,
projections, step embed). No Muon (model is small and cross-attn projections are
not the same regime as the rbf_ffn experiments). Cosine LR schedule with linear
warmup.

---

## Inference — Tree Construction (tree.py)

After one draft forward pass:

```python
def build_tree(logits: Tensor, tau: float) -> list[list[int]]:
    """
    logits: (max_draft, vocab)
    Returns all candidate paths whose per-position tokens satisfy:
        log_prob(token) >= log_prob(top_token_at_position) - tau
    Paths are the Cartesian product of per-position candidate sets,
    pruned to a maximum of MAX_TREE_NODES (config, default 256) by
    retaining only the highest-scoring paths by sum of log-probs.
    """
```

The resulting tree is fed to the teacher's tree-attention verification pass
(standard EAGLE-2 / Medusa verification protocol, not implemented here — the
draft model produces the tree; a separate inference harness runs verification).

`tau` is a runtime parameter; no retraining needed to adjust tree width.

---

## File Structure

```
mtp_draft/
├── config.py
├── data.py                    # HotpotQA tokenisation + FeatureDataset
├── cache.py                   # Phase 1 feature extraction
├── train.py                   # Phase 2 training loop
├── tree.py                    # Inference tree construction
├── models/
│   ├── __init__.py
│   ├── fusion.py              # KromHC multi-layer fusion
│   ├── step_embed.py          # Sinusoidal + MLP step embedding
│   ├── cross_attn_block.py    # CrossAttnBlock (KVShared + SwiGLU)
│   ├── draft_model.py         # MTPDraftModel top-level module
│   └── lora_lm_head.py        # Frozen LM head + LoRA adapter
├── configs/
│   └── default.yaml
└── tests/
    ├── test_fusion.py
    ├── test_step_embed.py
    ├── test_cross_attn_block.py
    ├── test_draft_model.py
    └── test_tree.py
```

---

## Out of Scope

- Tree-attention verification kernel (belongs in an inference harness, not this package).
- Multi-GPU / distributed training.
- Dynamic teacher layer selection (layers are a fixed config parameter).
- EAGLE-3-style training-time test (second-step distribution shift correction).
