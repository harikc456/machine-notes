---
title: LeWorldModel (LeWM)
created: 2026-05-14
updated: 2026-05-14
type: entity
tags: [world-model, ssl, architecture, training]
sources: [raw/papers/2603.19312v2.pdf]
confidence: high
---

# LeWorldModel (LeWM)

**LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels**
*Lucas Maes*, Quentin Le Lidec*, Damien Scieur, Yann LeCun, Randall Balestriero — Mila/Montréal, NYU, Samsung SAIL, Brown, arXiv:2603.19312, Mar 2026*

## Overview

LeWM is the first JEPA that trains **stably end-to-end from raw pixels** using only two loss terms. It applies the theoretical framework of [[lejepa]] to the world model / control setting, replacing the fragile heuristics of prior methods (stop-gradient, EMA, pretrained encoders) with a principled collapse-prevention mechanism.

**Scale**: 15M parameters, trainable on a single GPU in a few hours. Plans 48× faster than foundation-model-based world models.

## Architecture

```
Encoder (shared weights):
  pixels o_t  →  latent z_t

Predictor:
  (z_t, action a_t)  →  ẑ_{t+1}   (predicted next latent)

Loss = MSE(ẑ_{t+1}, z_{t+1})      [prediction loss]
     + λ · SIGReg(z_t, z_{t+1})   [anti-collapse regularizer]
```

The encoder and predictor are jointly optimized — no frozen encoder, no pretrained backbone, no auxiliary supervision. One hyperparameter: λ (the SIGReg weight).

## SIGReg (Sketched Isotropic Gaussian Regularization)

Prevents representation collapse by enforcing that the latent embedding distribution matches an isotropic Gaussian. Applied per-step to both `z_t` and `z_{t+1}`.

- Projects embeddings onto random 1D directions
- Tests each projection for normality via characteristic function matching
- Aggregates statistics to align full distribution to isotropic Gaussian

See [[lejepa]] for the theoretical justification and standalone SSL results.

## Advantages over Prior World Models

| | PLDM | DINO-WM | Dreamer | TD-MPC | LeWM |
|---|---|---|---|---|---|
| End-to-end | ✅ | ❌ | ✅ | ❌ | ✅ |
| Pixel-based | ✅ | ✅ | ✅ | ❌ | ✅ |
| Task-agnostic | ✅ | ✅ | ❌ | ❌ | ✅ |
| Reconstruction-free | ❌ | ✅ | ❌ | ✅ | ✅ |
| Provable anti-collapse | ❌ | ❌ | ❌ | ❌ | ✅ |
| # hyperparameters | 6 | many | many | many | **1** |

## Results

- Competitive across diverse 2D and 3D control tasks (manipulation, navigation, locomotion)
- 48× faster planning than foundation-model-based world models
- Latent space probing confirms meaningful physical structure (position, velocity, orientation)
- Surprise evaluation: model reliably detects physically implausible events — demonstrating genuine physical understanding

## Domain Note

LeWM is primarily a control/RL paper rather than an LLM paper. Its relevance to this wiki is:
1. SIGReg connects to [[orthogonal-residual-streams]] — both enforce that representations span full space rather than collapsing to subspaces
2. The anti-collapse theme parallels the training stability work in [[hyper-connections]] and [[weight-normalization]]
3. The JEPA framework (predict in latent space, not pixel space) may be relevant to future LLM architectural directions

## See Also

- [[lejepa]] — theoretical foundation and SIGReg derivation
- [[jepa]] — background on Joint Embedding Predictive Architectures
- [[orthogonal-residual-streams]] — related anti-collapse inductive bias in LLMs
