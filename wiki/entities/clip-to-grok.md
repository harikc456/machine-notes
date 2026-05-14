---
title: Clip to Grok
created: 2026-05-14
updated: 2026-05-14
type: entity
tags: [normalization, training, grokking, optimization]
sources: [raw/papers/cliptogrok.pdf]
confidence: high
---

# Clip to Grok

**Clip to Grok: Weight Norm Clipping for Accelerated Generalization**
*Vladimir Volchkov & Aviad Rivlin, Independent Researchers*

## Core Claim

Per-row weight norm clipping on decoder layers (attention projections, MLP layers, LayerNorm parameters — but **not** token embeddings or output head) accelerates [[grokking]] by **39–249×** across six algebraic tasks (modular arithmetic ×4, S₅ permutation, their composition), while eliminating the need for weight decay.

## Method

After every optimizer step, project each weight row onto the ℓ₂ ball of radius `c = max_norm`:

```
w_row ← w_row · min(1, c / ||w_row||₂)
```

`max_norm = 2.0` is the default for modular multiplication. This is **one-directional** (only shrinks, never amplifies) — unlike full [[weight-normalization]] which enforces bidirectional norm constraints.

## Four Theoretical Connections

1. **Goldilocks zone (Omnigrok)**: Norm clipping collapses the task-complexity-dependent Goldilocks zone timescale, motivating `max_norm ≈ w_c` as the optimal target. Edge initialization (normalizing only embeddings + final LayerNorm + output head) achieves zero grokking failures across 300 runs.

2. **α^L depth scaling law**: motivates the *edge initialization* strategy — normalize boundary layers, leave interior layers to evolve freely.

3. **Sign-based optimizers**: Lion and SignSGD produce uniform ±lr updates that combine naturally with ℓ₂ clipping to stabilize training across learning rates. Lion+Clip achieves fastest convergence across all configurations.

4. **Softmax Collapse prevention**: unconstrained weight growth leads to logit collapse that stalls generalization post-memorization. Clipping eliminates this.

## Key Observations

- **Grokfast's gradient EMA filter** ≈ sign-based optimization when passed through Adam's second-moment normalization — suggesting prior grokking acceleration methods may work through *implicit norm-constrained dynamics*
- **No weight decay needed**: clipping replaces the regularization role of weight decay entirely
- **Hyperparameter simplicity**: `max_norm` is fixed across optimizers, architectures, and LRs; determined by task complexity, not per-run tuning
- **Edge initialization**: normalizing only boundary layers (embeddings + final LayerNorm + output head) achieves zero failures on 8-layer models with 1.6M parameters

## Relationship to findings.md

The rbf_ffn `linear_weight_norm` uses bidirectional norm enforcement (not just clipping). The `max_only` variant tested in §6.2 is effectively the Clip-to-Grok mode — and it was catastrophically worse (75.54 vs 58.16 ppl). This aligns with the paper's setting: Clip-to-Grok is designed for **grokking** (algebraic tasks with delayed generalization), not general language modeling perplexity. The grokking dynamic may require different norm regime than autoregressive LM.

## See Also

- [[weight-normalization]] — full bidirectional norm reparameterization
- [[grokking]] — the delayed generalization phenomenon this paper studies
- [[weight-norm-training]] — practical normalization synthesis
