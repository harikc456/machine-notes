---
title: LeJEPA
created: 2026-05-14
updated: 2026-05-14
type: entity
tags: [ssl, architecture, training]
sources: [raw/papers/2511.08544v3.pdf]
confidence: high
---

# LeJEPA — Latent-Euclidean JEPA

**LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics**
*Randall Balestriero (Brown/Meta-FAIR) & Yann LeCun (NYU/Meta-FAIR), arXiv:2511.08544, Nov 2025*

## Core Claim

Existing JEPA training is brittle — it relies on stop-gradients, teacher-student EMA schedules, asymmetric views, and careful hyperparameter tuning to avoid representation collapse. LeJEPA replaces all of this with a **principled two-term objective** derived from theory: prediction loss + SIGReg.

## Theoretical Foundation

**Key result**: The isotropic Gaussian distribution is the *optimal* embedding distribution that minimizes downstream prediction risk across broad task families (both linear and nonlinear probes). This transforms JEPA design from ad-hoc heuristic exploration to a principled optimization target.

Two axioms a good JEPA must satisfy:
1. Solve the prediction task (predict future latent from current latent + action)
2. Enforce an isotropic Gaussian distribution on the embeddings

## SIGReg: Sketched Isotropic Gaussian Regularization

The key technical contribution — a novel objective that enforces isotropic Gaussian distribution on embeddings:

- **Random projections**: project embeddings onto multiple random 1D directions
- **Normality test**: apply a characteristic-function-based normality test to each 1D projection
- **Aggregate**: the aggregated statistics encourage the full distribution to match isotropic Gaussian

**Properties**:
- **Provably correct**: uniquely enforces isotropic Gaussian (unlike contrastive losses or covariance constraints that only enforce partial properties)
- **Linear time and memory complexity** in both dimension and sample size — scales to large models
- **Heuristics-free**: no stop-gradient, no EMA, no teacher-student, no scheduler
- **~50 lines of code**

## LeJEPA Results

- ViT-H/14 on ImageNet-1K: **79% top-1** (linear probe with frozen backbone)
- Consistent gains across 10+ datasets, 60+ architectures, vision + non-vision domains
- Training loss strongly correlates with downstream performance (94.52% Spearman) — enables model selection without supervised probing
- Stable training even on 1.8B ViT-g models

## Connection to LeWM

SIGReg is the same regularizer used in [[lewm]] (LeWorldModel). LeJEPA establishes the theory; LeWM applies it to pixel-based world models for control. They share authorship (Balestriero) and the same anti-collapse mechanism.

## Relationship to Broader Themes

- SIGReg's isotropic Gaussian enforcement is related to the [[orthogonal-residual-streams]] theme: forcing representations to span the full space (isotropic) rather than collapsing to a subspace (dimensional collapse) is another form of "write new, orthogonal information"
- The provable anti-collapse guarantee connects to [[hyper-connections]]' doubly-stochastic constraint — both are principled replacements for empirical training heuristics

## See Also

- [[lewm]] — application to pixel-based world models using the same SIGReg
- [[jepa]] — background on Joint Embedding Predictive Architectures
- [[orthogonal-residual-streams]] — related theme of anti-collapse inductive bias
