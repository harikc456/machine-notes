---
title: VISReg
created: 2026-07-06
updated: 2026-07-06
type: entity
tags: [ssl, training]
sources: [raw/papers/2606.02572v1.pdf]
confidence: medium
---

# VISReg — Variance-Invariance-Sketching Regularization

*Wu, Balestriero, Levine (Altos Labs / Brown University), preprint, June 2026*

## Core Claim

Fixes two complementary weaknesses in existing non-contrastive SSL regularizers: VICReg's
covariance term only enforces second-order decorrelation (not full distributional shape), while
[[lejepa]]'s SIGReg enforces full distributional shape but has a vanishing gradient under collapse
and couples scale to shape.

## Method

Keeps VICReg's variance term for scale control, replaces its covariance term with a
**Sliced-Wasserstein-Distance sketching objective** that aligns the embedding distribution with an
isotropic Gaussian along random 1D projections. Decoupling scale (variance term) from shape
(sketching term) gives robust gradients under collapse plus the interpretability/flexibility of
VICReg's decomposed loss.

## Results

- ImageNet-1K (ViT-B/16): SOTA OOD generalization vs. DINO/VICReg/SIGReg; trades ~3% in-domain
  linear-probe accuracy vs. DINO for stronger transfer; on par with DINO on dense prediction
  (linear segmentation).
- ImageNet-22K (ViT-L/14): matches DINOv2's OOD performance using 10× less pretraining data.
- Most robust of the compared methods on low-quality, long-tailed, low-rank data regimes.
- Linear time/memory complexity (no O(D²) covariance matrix).

## Relationship to Broader Themes

Part of the same push as [[lejepa]] to remove training heuristics (EMA, stop-gradient,
teacher-student asymmetry — see [[jepa]]) from self-supervised/joint-embedding training, replacing
them with theoretically motivated regularizers. Where LeJEPA's SIGReg enforces isotropy via a
characteristic-function normality test, VISReg enforces it via Sliced-Wasserstein sketching and
explicitly separates the scale/shape objectives SIGReg conflates.

## See Also

- [[lejepa]] — SIGReg-based provable JEPA; VISReg directly addresses its two limitations
- [[jepa]] — background on Joint Embedding Predictive Architectures and the collapse problem
- [[lewm]] — pixel-based JEPA world model using SIGReg
