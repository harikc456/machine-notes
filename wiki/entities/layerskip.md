---
title: LayerSkip
created: 2026-05-14
updated: 2026-05-14
type: entity
tags: [inference, architecture, speculative, optimization]
sources: [raw/papers/2404.16710v2.pdf]
confidence: high
---

# LayerSkip

**LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding**
*Mostafa Elhoushi, Akshat Shrivastava, Diana Liskovich, Basil Hosmer, Bram Wasti, Liangzhen Lai, Anas Mahmoud, Bilge Acun, Saurabh Agarwal, Ahmed Roman, Ahmed A. Aly, Beidi Chen, Carole-Jean Wu — Meta, arXiv:2404.16710, 2024*

## Overview

LayerSkip introduces a training procedure and inference strategy that allow a single model to operate at variable depths: either exiting early for speed, or using early layers as an internal draft model for self-speculative decoding.

## Training: Layer Dropout

During training, each layer is independently dropped with probability `p_l`:
- Early layers: low dropout probability (always well-trained)
- Deep layers: high dropout probability (rarely all used during training)

This conditions every prefix of layers to produce a useful hidden state — the model learns to "be ready to answer" at any depth.

An **early exit loss** is also added: the LM head is applied at intermediate layers, and prediction loss is computed there too (with a decay weighting that emphasizes deeper layers more). This provides a direct gradient signal to intermediate representations.

## Inference: Two Modes

### Mode 1: Hard Early Exit

Run the full batch to exit layer e (instead of L). Use the shared LM head at layer e.

- **Speedup**: proportional to `e/L` (skip `L-e` layers per token)
- **Quality**: degrades with smaller e; the layer dropout training minimizes this degradation

### Mode 2: Self-Speculative Decoding

Use early layers (0..e) as the draft, full model (0..L) as the verifier:

```
Draft phase:  run layers 0..e → generate γ draft tokens
Verify phase: run layers 0..L on all γ+1 positions in parallel
              → accept/reject via speculative sampling
```

The verification pass re-uses the KV states computed in the draft phase for layers 0..e, so the verifier only needs to run layers e+1..L for the positions that need verification. This further reduces the cost.

**Guarantee**: same distributional guarantee as standard speculative decoding — output distribution is identical to full-model autoregressive generation.

## Advantages over Standard Speculative Decoding

| | Standard Spec Decoding | LayerSkip Self-Speculative |
|---|---|---|
| Draft model | Separate model (extra memory) | Own early layers (no extra memory) |
| Training | Independent for draft + target | Single model with layer dropout |
| Draft quality | Depends on sibling model | Bounded by early-exit quality |
| Memory footprint | 2 models | 1 model |

## Results

- Up to 2.16× speedup on summarization with Llama-2-7B
- Self-speculative mode outperforms hard early exit for the same compute budget
- Works on Llama-2 (7B/13B/34B/70B) and CodeLlama

## See Also

- [[early-exit-inference]] — concept page covering the broader early exit landscape (SWIFT, DASH)
- [[speculative-decoding]] — general speculative decoding framework
- [[continuous-batching]] — serving scheduler; LayerSkip affects per-token layer count, orthogonal
