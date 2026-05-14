---
title: Derf (Dynamic erf)
created: 2026-05-14
updated: 2026-05-14
type: entity
tags: [normalization-free, architecture, training, normalization]
sources: [raw/papers/2512.10938v2.pdf]
confidence: high
---

# Derf — Dynamic erf

**Stronger Normalization-Free Transformers**
*Mingzhi Chen, Taiming Lu, Jiachen Zhu, Mingjie Sun, Zhuang Liu — Princeton, NYU, CMU, arXiv:2512.10938, Dec 2025*

## Core Idea

Replace normalization layers (LayerNorm, RMSNorm) with a **point-wise function** applied independently to each element:

```
Derf(x) = erf(αx + s)
```

where `erf` is the rescaled Gaussian CDF (S-shaped, bounded in [-1, +1]), and `α`, `s` are learnable per-channel scalars. Unlike LayerNorm, Derf operates without computing statistics across tokens or channels — it is a pure element-wise mapping.

## Motivation: Normalization-Free Design

Standard normalization layers compute activation statistics (mean, variance) across a group (channel or token), introducing:
- Memory access overhead (extra reads/writes for statistics)
- Batch-size sensitivity (batch norm) or synchronization cost (layer norm in distributed settings)
- Coupling between elements in the same normalization group

**DyT (Dynamic Tanh, Zhu et al. 2025)** established that point-wise functions can match normalization layers. Derf advances this by finding a *stronger* function that **surpasses** normalization layers.

## Why erf Works Best

Systematic search over point-wise function properties identified four key properties:
- **Zero-centeredness**: function centered near zero (like tanh, erf — not sigmoid)
- **Boundedness**: output in a finite range (prevents activation explosion)
- **Center sensitivity**: steep slope near zero (high gradient for small activations)
- **Monotonicity**: preserves ordering

`erf(αx + s)` satisfies all four. Compared to `tanh(αx)` (DyT):
- The shift `s` allows asymmetric activation behavior
- erf's Gaussian CDF shape has gentler tails than tanh, giving smoother gradient flow

## Results

| Method | ViT ImageNet-1K | DiT FID↓ | DNA acc |
|---|---|---|---|
| LayerNorm | 82.3% | 45.91 | 86.9% |
| DyT | 82.5% | 45.66 | 86.9% |
| **Derf** | **82.8%** | **43.94** | **87.3%** |

Evaluated across vision (classification + generation), speech, and DNA modeling. Gains are attributed to **stronger generalization** rather than enhanced fitting capacity — Derf has higher training loss than LayerNorm models in evaluation mode, suggesting it finds flatter minima.

## Why Gains Come from Generalization, Not Fitting

After optimization, Derf models exhibit *higher training loss* (evaluated without dropout etc.) than LayerNorm models with similar validation performance. This is the hallmark of better generalization — the model has learned a more transferable representation. This connects to the [[grokking]] theme: norm constraint → Goldilocks zone → better generalization.

## Relationship to Other Work

- Builds on **DyT** (Zhu et al. 2025) which showed point-wise functions can match, but not surpass, LayerNorm
- Connects to [[weight-normalization]]: both decouple magnitude from direction, but at different levels (weight vectors vs. activation values)
- The `dynamic_erf` norm type was tested in the rbf_ffn MoE experiments (§9.4 of findings.md) — it underperformed RMSNorm by 6.55 ppl in that setting. However, the rbf_ffn implementation may differ from Derf's parameterization; worth revisiting with correct α, s init
- [[weight-norm-training]] synthesis applies to weight norms, not activation norms — these are complementary

## See Also

- [[normalization-free-transformers]] — broader landscape of norm-free architectures
- [[weight-normalization]] — weight-level (not activation-level) normalization
- [[weight-norm-training]] — practical normalization synthesis
