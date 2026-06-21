# mtp_draft

A lightweight Multi-Token Prediction (MTP) draft model for speculative decoding, distilled from Gemma 4 E2B hidden states. The draft model predicts up to `max_draft` future tokens in a single forward pass using intermediate teacher features rather than re-running the teacher.

## Idea

Instead of running the full teacher at every draft step (as in standard speculative decoding), we extract hidden states from a few intermediate teacher layers at the current position and train a small cross-attention network to predict the next `max_draft` tokens. At inference, the teacher generates one token per verified step; the draft model proposes a tree of candidates that the teacher verifies in parallel.

This is closest in spirit to [EAGLE](../wiki/entities/eagle.md): the draft model consumes teacher features rather than raw embeddings. The difference is multi-head KromHC feature fusion across layers and an explicit draft tree built from relative log-prob thresholds.

## Pipeline

```
Phase 1 — cache          Phase 2 — train          Inference
──────────────────        ────────────────          ─────────
Gemma 4 E2B               FeatureDataset            teacher hidden states
  → hidden states           (int8 on disk)            → MTPDraftModel
  at layers [3,8,14,17]   → MTPDraftModel            → build_tree()
  → int8 quantised         → weighted CE loss         → teacher verify
  → sharded .pt files      → cosine LR + AdamW
```

## Architecture

```
teacher_hiddens (B, n_layers, d_teacher)
        │
   TeacherFeatureFusion
   ├── per-layer Linear(d_teacher → d_draft)
   └── KromHCHeadMixer  →  q_fused (B, d_draft)
        │
   + step embeddings  →  Q (B, max_draft, d_draft)
        │         teacher_hiddens[:,-1,:]    context_ids (B, seq_len)
        │                    │                      │
        │            anchor_proj                frozen token embedding
        │         Linear(d_teacher→d_draft)          │
        │                    │                ctx_proj Linear(d_teacher→d_draft)
        │                    └──────── cat ──────────┘
        │                                    │
   n_blocks × CrossAttentionBlock  ←  context (B, 1+seq_len, d_draft)
   ├── cross-attn Q=draft, KV=context
   └── LoRA on K/V projections
        │
   out_proj  Linear(d_draft → d_teacher)
        │
   LoRA LM head (frozen W + trainable A, B)  →  logits (B, max_draft, vocab)
```

**`TeacherFeatureFusion`** (`models/fusion.py`) — projects each teacher layer independently, mixes across heads with KromHC, then collapses with a learned softmax-weighted sum over layers (`layer_weights`, zero-initialised → uniform at start). `n_teacher_layers` must be a power of 2.

**`MTPDraftModel`** (`models/draft_model.py`) — the full model. Frozen buffers: teacher token embedding, LM head weight. Trainable: `TeacherFeatureFusion` (including `layer_weights`), `StepEmbedding`, `anchor_proj` (last cached layer → d_draft global context token), `ctx_proj` (token embeddings → d_draft), `CrossAttentionBlock`s, `out_proj` (d_draft→d_teacher), and LM head LoRA adapters.

**`CrossAttentionBlock`** (`models/cross_attn_block.py`) — cross-attention where Q comes from the draft sequence and KV come from the projected context embeddings. LoRA adapters on K and V.

**`LoRALMHead`** (`models/lora_lm_head.py`) — frozen teacher LM head weight with trainable low-rank adapter. `logits = x @ (W + B @ A).T`; B is zero-initialised so the adapter starts as a no-op.

**`build_tree`** (`tree.py`) — given `(max_draft, vocab)` logits, expands a draft tree by admitting tokens whose log-prob is within `tau` nats of the top token at each depth. Returns at most `max_tree_nodes` paths.

## File layout

```
mtp_draft/
├── cache.py          # Phase 1: extract + int8-quantise teacher hidden states
├── data.py           # FeatureDataset over cached .pt shards
├── train.py          # Phase 2: training loop
├── config.py         # MTPConfig dataclass
├── tree.py           # build_tree for inference
├── configs/
│   └── default.yaml
├── models/
│   ├── draft_model.py
│   ├── fusion.py
│   ├── cross_attn_block.py
│   ├── lora_lm_head.py
│   └── step_embed.py
└── tests/
```

## Usage

**Step 1 — extract teacher features**

```bash
python -m mtp_draft.cache --config mtp_draft/configs/default.yaml
```

Runs Gemma 4 E2B over ToolAlpaca, extracts hidden states at `teacher_layers`, int8-quantises them, and writes sharded `.pt` files to `cache_dir`. Only needs to run once.

**Step 2 — train the draft model**

```bash
python -m mtp_draft.train --config mtp_draft/configs/default.yaml
```

Trains on cached features. Best checkpoint (by val loss) saved to `mtp_draft/checkpoints/best.pt`. Frozen weights (token embedding, LM head) are excluded from the checkpoint.

**Tests**

```bash
pytest mtp_draft/tests/
```

## Config reference (`configs/default.yaml`)

| Key | Default | Notes |
|-----|---------|-------|
| `d_draft` | 512 | Draft model hidden dim |
| `n_blocks` | 4 | Cross-attention blocks |
| `ffn_hidden` | 2048 | SwiGLU hidden dim |
| `n_heads` | 8 | Attention heads |
| `dropout` | 0.1 | Applied in cross-attn blocks |
| `lora_rank` | 16 | LoRA rank on K/V projections |
| `teacher_model_id` | `google/gemma-4-e2b-it` | |
| `teacher_layers` | `[3, 8, 14, 17]` | Must be power-of-2 count |
| `d_teacher` | 1536 | Gemma 4 E2B hidden size |
| `max_draft` | 8 | Tokens predicted per forward pass |
| `lambda_decay` | 0.8 | Per-position loss weight decay |
| `tau` | 2.0 | Log-prob threshold for tree expansion |
| `max_tree_nodes` | 256 | Max paths in draft tree |
| `max_prompt_len` | 256 | Context token budget |
| `cache_n_answer_positions` | 8 | Anchor positions per example |
