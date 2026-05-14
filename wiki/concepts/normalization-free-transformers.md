---
title: Normalization-Free Transformers
created: 2026-05-14
updated: 2026-05-14
type: concept
tags: [normalization-free, architecture, normalization, training]
sources: [raw/papers/2512.10938v2.pdf]
confidence: high
---

# Normalization-Free Transformers

## Overview

Normalization layers (LayerNorm, RMSNorm, BatchNorm) have been considered indispensable for training stability in deep networks. Recent work shows they can be replaced — and even surpassed — by **point-wise functions**: elementwise mappings applied independently to each activation with no statistics computed across tokens or channels.

## Why Normalization Layers Have Costs

Standard normalization: `y = γ · (x − μ) / (σ² + ε)^0.5 + β`

- Computes `μ`, `σ²` across a group (channels for LayerNorm, batch for BatchNorm)
- Requires extra memory reads/writes for statistics
- Batch-size sensitive (BatchNorm), or forces synchronization in distributed training
- Couples elements within the normalization group

## The Point-Wise Replacement Paradigm

A point-wise function `f(x)` operates independently on each scalar activation — no grouping, no statistics.

| Method | Function | Key property |
|---|---|---|
| DyT (Zhu et al. 2025) | `tanh(αx)` | Bounded, zero-centered; matches LayerNorm |
| **Derf** ([[derf]]) | `erf(αx + s)` | Bounded, zero-centered, shifted; **surpasses** LayerNorm |

**Derf** is the current strongest result: ViT 82.8% vs LayerNorm 82.3% on ImageNet-1K, with gains attributed to better generalization (higher training loss, lower test loss).

## Four Key Properties of Effective Point-Wise Functions

Ablation studies in the Derf paper isolate four properties:
1. **Zero-centeredness**: function passes near zero — not shifted like sigmoid
2. **Boundedness**: output range is finite — prevents activation explosion
3. **Center sensitivity**: steep slope near zero — strong gradient signal for typical activations
4. **Monotonicity**: preserves input ordering

The `erf` function satisfies all four optimally. `tanh` satisfies all four but has heavier tails. Sigmoid (non-zero-centered) and ReLU (unbounded from above) violate one each.

## Relationship to Other Normalization Techniques

Normalization-free architecture is **distinct from weight normalization**:

| Technique | What it normalizes | When | Purpose |
|---|---|---|---|
| LayerNorm / RMSNorm | Activation statistics | Forward pass | Stabilize activations |
| [[weight-normalization]] | Weight vector norms | Optimizer step | Condition optimization landscape |
| Derf / DyT | Replaces LayerNorm entirely | Forward pass | Same effect, no statistics overhead |
| [[qknorm]] | Q and K vectors in attention | Forward pass | Bound pre-softmax logits |

These can be combined: weight normalization and Derf address different parts of the training pipeline.

## Interesting Note: dynamic_erf in rbf_ffn

The rbf_ffn project tested a `dynamic_erf` norm type in §9.4 (MoE + dynamic_erf + wnorm), which underperformed standard RMSNorm by 6.55 ppl. This may reflect an incorrect parameterization or initialization of the erf function rather than the Derf approach's failure — the published Derf uses learnable per-channel `α` and `s`, which may matter significantly.

## Open Questions

- Does Derf's improvement generalize to autoregressive language modeling (vs vision/DNA)?
- Can Derf be combined with [[weight-normalization]] for additive gains?
- What happens at very large scale (100B+) — does the normalization-free paradigm hold?

## See Also

- [[derf]] — Dynamic erf, the strongest current point-wise function
- [[weight-normalization]] — complementary weight-level normalization
- [[weight-norm-training]] — synthesis of normalization techniques
- [[qknorm]] — query-key normalization for attention
