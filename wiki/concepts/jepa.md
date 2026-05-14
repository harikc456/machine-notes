---
title: Joint Embedding Predictive Architecture (JEPA)
created: 2026-05-14
updated: 2026-05-14
type: concept
tags: [ssl, architecture, world-model]
sources: [raw/papers/2511.08544v3.pdf, raw/papers/2603.19312v2.pdf]
confidence: high
---

# Joint Embedding Predictive Architecture (JEPA)

## Overview

JEPA is a self-supervised learning framework introduced by LeCun. Instead of predicting raw observations (pixels, tokens), JEPA predicts the **latent representation** of future observations from the latent representation of current observations.

```
Encoder: o_t  →  z_t          (observation to latent)
Predictor: (z_t, context)  →  ẑ_{t+1}   (predict next latent)
Loss: distance(ẑ_{t+1}, z_{t+1})
```

The key insight: predicting in latent space allows the model to ignore unpredictable low-level details (pixel noise, irrelevant features) and focus on semantically meaningful structure.

## The Collapse Problem

Naive JEPA training trivially collapses: the encoder can map all inputs to the same constant embedding, making prediction loss zero. Preventing this is the central engineering challenge.

Existing approaches rely on heuristics:
- **Stop-gradient / EMA target encoder**: teacher-student architecture where the target encoder is a slow-moving average of the online encoder
- **Asymmetric view generation**: online and target encoders see different augmentations
- **Explicit normalization**: whitening layers, contrastive losses
- **Multi-term objectives**: balance multiple loss terms with carefully tuned weights

These approaches work empirically but lack theoretical grounding and require careful hyperparameter tuning.

## Principled Anti-Collapse: SIGReg (LeJEPA)

[[lejepa]] (Balestriero & LeCun, 2025) provides the first principled solution:

**Theorem**: The isotropic Gaussian distribution uniquely minimizes downstream prediction risk over broad task families.

This transforms the design goal from "prevent collapse" (a negative constraint) to "achieve isotropic Gaussian embeddings" (a positive target). SIGReg enforces this via random projections + normality tests.

Result: no stop-gradient, no EMA, no asymmetric views, no tuning — just prediction loss + SIGReg.

## JEPA for World Models (LeWM)

[[lewm]] (Maes et al., 2026) applies the LeJEPA framework to the world model setting, where observations are video frames and "context" includes actions. The same two-term objective (prediction + SIGReg) trains a pixel-based world model end-to-end on a single GPU.

## Relationship to LLM Architecture

JEPAs operate on continuous observations (images, video) while LLMs operate on discrete tokens. However, the conceptual connection is strong:

- Autoregressive LLMs predict the next token's embedding — a discrete JEPA
- The collapse prevention challenge parallels the training stability problem in [[hyper-connections]] and [[orthogonal-residual-streams]]
- SIGReg's isotropic Gaussian target is related to the goal of [[orthogonal-residual-streams]]: representations should span the full space (isotropic) rather than concentrate in a subspace

## Open Questions

- Can SIGReg be applied to token embeddings in LLMs to improve representation quality?
- Is there an LLM analogue of the "attention similarity bias" problem in JEPA collapse?
- Does the JEPA prediction objective offer any advantage over autoregressive cross-entropy for learning structure?

## See Also

- [[lejepa]] — provable JEPA with SIGReg
- [[lewm]] — JEPA world model for pixel-based control
- [[orthogonal-residual-streams]] — related anti-collapse inductive bias in LLMs
- SSL (self-supervised learning) background is covered inline above
