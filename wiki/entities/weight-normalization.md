---
title: Weight Normalization
created: 2026-05-14
updated: 2026-05-14
type: entity
tags: [normalization, training, optimization]
sources: [raw/papers/1602.07868v3.pdf]
confidence: high
---

# Weight Normalization

**Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks**
*Tim Salimans & Diederik P. Kingma, OpenAI, arXiv:1602.07868, 2016*

## Core Idea

Reparameterize each weight vector **w** as:

```
w = (g / ||v||) · v
```

where **v** is a direction vector and `g` is a scalar magnitude. This **decouples the length of the weight vector from its direction**, improving the conditioning of the optimization problem and speeding up SGD convergence.

## Motivation

The conditioning number of the Hessian at the optimum determines how easily SGD converges. When length and direction are entangled in a single parameter, curvature in the loss landscape is pathological — gradient steps that improve direction also distort magnitude in unpredictable ways. By making `g` and **v** independent, gradient updates in each direction are approximately scaled by the inverse of the corresponding curvature, similar to natural gradient methods but far cheaper.

## Properties

- **No batch dependencies**: unlike batch normalization, weight norm is deterministic per example — safe for RNNs, RL, generative models
- **No extra memory**: normalization is applied to parameters once at step time, not per forward pass
- **Lower overhead than batch norm**: allows more optimizer steps in the same wall-clock time
- **Inspired by but distinct from batch normalization**: both improve the Fisher matrix conditioning; batch norm does it via activation statistics, weight norm does it via reparameterization

## Variants Studied in rbf_ffn Research

The `linear_weight_norm` implementation in the rbf_ffn project applies a target norm constraint (target=2.0 by default). Key findings (see `rbf_ffn/findings.md §6`):
- **Weight normalization is the single largest improvement** — ~21 ppl gain at 3 epochs on WikiText-103, dwarfing all activation variants
- **Bidirectional constraint is essential**: `max_only` clipping (§6.2) catastrophically regresses — weight norm must enforce both shrinking and scaling, not just clipping
- **Erases activation variant advantages**: PFDRationalGLU's +3.5% advantage over SwiGLU disappears under weight norm (§6.1)

## Connection to Clip to Grok

[[clip-to-grok]] studies weight norm clipping specifically in the context of grokking — showing that clamping row norms after each optimizer step accelerates generalization by 39–249× and connects to the same Goldilocks zone dynamics. The clipping variant is a subset of weight normalization (max_only mode), but the grokking analysis provides a theoretical explanation for why norm constraints matter.

## See Also

- [[clip-to-grok]] — weight norm clipping for grokking acceleration
- [[qknorm]] — analogous normalization applied to Q and K in attention
- [[orthogonal-residual-streams]] — broader context of norm constraints for training stability
- [[weight-norm-training]] — synthesis of normalization techniques for transformer training
