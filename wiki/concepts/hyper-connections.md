---
title: Hyper-Connections
created: 2026-05-14
updated: 2026-05-19
type: concept
tags: [architecture, residual, training]
sources: [raw/papers/2512.24880v2.pdf, raw/papers/2601.05732v1.pdf, raw/papers/2601.21579v1.pdf]
confidence: high
---

# Hyper-Connections (HC) Family

## Overview

Hyper-Connections (HC) generalize standard residual connections by introducing **dynamic residual matrices** that mix information across multiple parallel residual streams. The single scalar identity shortcut is replaced with a learned `n×n` matrix that combines n streams.

Standard residual: `X_{l+1} = X_l + F(X_l)`

Hyper-connection: `X_{l+1} = H^res_l X_l + H^post_l F(H^pre_l X_l)`

Where `H^res, H^pre, H^post` are learned residual matrices that aggregate and redistribute information across n streams (stream width n).

**Benefits of HC over standard residual:**
- More expressive feature propagation
- Accelerates convergence
- Allows different residual streams to specialize

## The Stability Problem

Unconstrained residual matrices break the **identity mapping property** of standard residuals. In deep networks, the composition of unconstrained matrices `∏ H^res_l` can diverge from identity, causing:
- Gradient explosions
- Feature mean drift across layers
- Training instabilities at scale

This motivated the **doubly-stochastic constraint**: if `H^res` has all row/column sums = 1, its spectral norm is bounded by 1, preventing gradient explosion.

## The Doubly-Stochastic Constraint

A matrix where all rows and columns sum to 1. The set of doubly-stochastic matrices is the **Birkhoff polytope**. Key properties:
- Spectral norm ≤ 1
- Closed under multiplication
- Vertices are permutation matrices (Birkhoff-von Neumann theorem)

## Methods to Achieve Doubly-Stochasticity

| Method | Approach | Exact? | Param Complexity | PyTorch Native? |
|---|---|---|---|---|
| [[mhc]] | Sinkhorn-Knopp iterations (20) | ❌ (approx) | O(n³C) | ❌ |
| [[mhc-lite]] | Convex combo of all n×n permutation matrices | ✅ | O(n·n!) | ✅ |
| [[kromhc]] | Kronecker product of 2×2 DS matrices | ✅ | O(n²C) | ✅ |

For detailed comparison see [[hyper-connections-variants]].

## Deployment

- [[mhc]] is used in [[deepseek-v4]] at trillion-parameter scale
- Research on [[mhc-lite]] (Yang & Gao) and [[kromhc]] (Zhou et al.) represents the community improving on DeepSeek's approach

## Alternative Approach: Attention Residuals

[[attnres]] (Kimi Team, Mar 2026) takes a fundamentally different approach: instead of maintaining m parallel streams, it gives each layer direct softmax attention over *all* preceding layer outputs using a single learned pseudo-query per layer. In scaling ablations, Full AttnRes (loss 1.737) and Block AttnRes (1.746) both outperform mHC-lite (1.747) and mHC (1.747) at the 436M scale, while Block AttnRes uses only **5.5d memory I/O per layer** vs mHC's 34d (m=4 streams). The two methods are architecturally orthogonal — HC widens the residual stream, AttnRes selects across depth — and could in principle be combined.

## Open Questions

- Can Kronecker-product construction be extended beyond stream width n = 2^K?
- Does stream width n have diminishing returns beyond a certain point?
- How does HC interact with [[mixture-of-experts]] at large scale?
- Can HC multi-stream and AttnRes cross-layer access be combined?

## See Also

- [[mhc]] — SK-based approximation
- [[mhc-lite]] — permutation-based exact construction
- [[kromhc]] — Kronecker-product exact construction
- [[hyper-connections-variants]] — full comparison
- [[attnres]] — alternative: depth-wise softmax attention over all preceding layers
- [[deepseek-v4]] — large-scale deployment context
