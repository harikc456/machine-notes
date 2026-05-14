---
title: mHC (Manifold-Constrained Hyper-Connections)
created: 2026-05-14
updated: 2026-05-14
type: entity
tags: [architecture, residual, training, deepseek]
sources: [raw/papers/2512.24880v2.pdf]
confidence: high
---

# mHC — Manifold-Constrained Hyper-Connections

**mHC: Manifold-Constrained Hyper-Connections**
*Zhenda Xie et al., DeepSeek-AI, arXiv:2512.24880, Dec 2025*

## Overview

mHC is DeepSeek's solution to the training instability problem in [[hyper-connections]] (HC). HC extends residual connections by introducing dynamic residual matrices that mix information across multiple residual streams — but unconstrained dynamic matrices can cause gradient explosions.

mHC constrains the residual matrices to the **Birkhoff polytope** (the set of doubly-stochastic matrices) via iterative **Sinkhorn-Knopp (SK) normalization**, restoring the identity mapping property and bounding spectral norm.

## Technical Details

### Doubly-Stochastic Constraint
- All row and column sums equal 1
- Spectral norm bounded by 1
- Set is closed under multiplication → prevents gradient explosions through depth

### mHC Architecture
- Residual matrix `H^res` updated via 20 SK iterations per forward pass
- Requires custom fused CUDA kernels for efficiency
- Stores per-iteration intermediate results only in forward pass; recomputes during backward

### Why Projection Matters
Standard HC: `X_{l+1} = H^res_l X_l + H^post_l F(H^pre_l X_l)`

Without constraint on `H^res`, the product `∏ H^res_{L-i}` can deviate from identity, causing feature mean drift across layers.

## Experimental Results

- Demonstrates performance improvements and superior scalability over standard HC
- Used in [[deepseek-v4]] at trillion-parameter scale
- Effective for training stability at scale where vanilla HC fails

## Limitations (addressed by successors)

1. **Approximation gap**: 20 finite SK iterations do not guarantee exact doubly-stochasticity — error accumulates through depth
2. **Engineering barrier**: requires highly specialized CUDA kernels; not portable to standard PyTorch stacks
3. **Parameter complexity**: O(n³C) where n = stream width — prohibitive for large n

These limitations motivated [[mhc-lite]] and [[kromhc]].

## See Also

- [[hyper-connections]] — background on HC
- [[mhc-lite]] — reparameterization that guarantees exact doubly-stochasticity without SK iterations
- [[kromhc]] — Kronecker-product approach that is both exact and parameter-efficient
- [[hyper-connections-variants]] — comparison of all variants
