---
title: Weight Normalization in Transformer Training
created: 2026-05-14
updated: 2026-05-14
type: concept
tags: [normalization, training, attention]
sources: [raw/papers/1602.07868v3.pdf, raw/papers/2010.04245v1.pdf, raw/papers/cliptogrok.pdf]
confidence: high
---

# Weight Normalization in Transformer Training

## Overview

Normalizing weight magnitudes is one of the most impactful interventions for transformer training stability and convergence — often larger than architectural activation choices. This page synthesizes the three normalization papers in the wiki alongside empirical findings from `rbf_ffn/findings.md`.

## Three Normalization Techniques

### Linear Weight Normalization ([[weight-normalization]])
Reparameterize `w = (g / ||v||) · v`. Decouples length from direction. Applied to all linear layer weight rows.

**Effect in rbf_ffn (§6.1)**: ~21 ppl improvement at 3 epochs. Single largest factor — dominates all architectural choices (activation functions, gate variants, attention variants).

### Query-Key Normalization ([[qknorm]])
Replace `q·k` dot product with cosine similarity scaled by a learnable parameter. Bounds pre-softmax logits.

**Effect in rbf_ffn (§6.1)**: +0.5–1 ppl consistently on top of weight norm. Low cost. Worth including always.

### Weight Norm Clipping ([[clip-to-grok]])
Unidirectional: clip row norms above a threshold `max_norm`, never scale up. Designed for grokking tasks.

**Effect in rbf_ffn (§6.2)**: catastrophic regression in LM (75.54 vs 58.16 ppl). The bidirectional constraint of full weight norm is essential for language modeling.

## Practical Ordering (from rbf_ffn experiments)

| Technique | Gain | Cost | Recommended? |
|---|---|---|---|
| `weight_norm` (bidirectional) | ~21 ppl | ~8% overhead | ✅ Always |
| `qk_norm` | ~0.7 ppl | ~6% overhead | ✅ Always |
| `max_only` clipping | −17 ppl (regression) | negligible | ❌ Not for LM |

## Interaction with Architecture Choices

**Weight norm erases activation variant advantages** (§6.1): PFDRationalGLU's +3.5% advantage over SwiGLU (measured without normalization) disappears completely under weight norm. The architectural difference is absorbed into the normalization's landscape smoothing.

**This is the key practical lesson**: when adding weight normalization, re-evaluate architecture choices. An activation variant that "wins" without norm may be equivalent with norm — saving implementation complexity.

**XSA's gain *persists* under weight norm** (§8.8): unlike activation variants, XSA's +1.3 ppl advantage over SwiGLU under wnorm indicates a structural inductive bias, not just an optimization conditioning effect.

## Relationship to Training Stability

Both weight norm and [[hyper-connections]] (mHC/mHC-lite/KromHC) address training stability, but from different angles:

| Approach | What it constrains | Why |
|---|---|---|
| Weight normalization | Weight vector norms (per row) | Conditions optimization landscape |
| mHC / KromHC | Residual mixing matrix (doubly-stochastic) | Prevents gradient explosion through depth |
| QKNorm | Pre-softmax attention logits | Prevents softmax saturation |

These are complementary — weight norm and mHC target different failure modes and can in principle be combined.

## See Also

- [[weight-normalization]] — original paper
- [[qknorm]] — attention normalization
- [[clip-to-grok]] — unidirectional clipping for grokking
- [[grokking]] — the generalization dynamics that norm clipping accelerates
- [[orthogonal-residual-streams]] — a different perspective on what normalization achieves
- [[hyper-connections]] — stability via residual matrix constraints
- [[normalization-free-transformers]] — point-wise replacements for LayerNorm (Derf, DyT); related design philosophy
- [[derf]] — `erf(αx+s)` as a drop-in norm replacement; surpasses LayerNorm across modalities
