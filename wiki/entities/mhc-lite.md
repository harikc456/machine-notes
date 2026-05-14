---
title: mHC-lite
created: 2026-05-14
updated: 2026-05-14
type: entity
tags: [architecture, residual, training]
sources: [raw/papers/2601.05732v1.pdf]
confidence: high
---

# mHC-lite

**mHC-lite: You Don't Need 20 Sinkhorn-Knopp Iterations**
*Yongyi Yang, Jianyang Gao, arXiv:2601.05732, Jan 2026*

## Overview

mHC-lite is a reparameterization of [[mhc]] that **eliminates SK iterations entirely** by directly constructing doubly-stochastic residual matrices as convex combinations of permutation matrices — a construction guaranteed exact by the Birkhoff-von Neumann theorem.

## Motivation

[[mhc]] uses 20 SK iterations to *approximate* doubly-stochasticity. This creates two problems:
1. **Approximation gap**: finite SK iterations can leave column sums as 1.92, 0.59, 0.59 (example in paper) — errors accumulate with depth, undermining stability
2. **Engineering complexity**: SK iterations require specialized fused CUDA kernels for competitive throughput

## Technical Approach

The **Birkhoff-von Neumann theorem** states that every doubly-stochastic matrix is a convex combination of permutation matrices. mHC-lite exploits this directly:

```
H^res = Σ_i a_i(l) · P_i    where Σ a_i = 1, a_i ≥ 0
```

The coefficients `a_i` are unconstrained scalars passed through softmax; permutation matrices are fixed. The result:
- **Exact** doubly-stochasticity by construction
- **Native matrix operations only** — no specialized kernels needed
- **Simpler backward pass** — no intermediate SK state to manage

## Key Results

- Matches or **exceeds** mHC performance on downstream tasks
- Higher training throughput than naive mHC (no SK kernel overhead)
- Eliminates residual instabilities observed in both HC and mHC
- Implementable in standard PyTorch without custom CUDA

## Limitations

Parameter complexity explodes: requires storing `n!` unique `n×n` permutation matrices where n = stream width. For n=8, that's 8! = 40,320 unique matrices — **factorial parameter explosion**. This is the limitation that [[kromhc]] addresses.

## See Also

- [[mhc]] — predecessor with SK iteration approximation
- [[kromhc]] — addresses mHC-lite's factorial parameter count via Kronecker products
- [[hyper-connections]] — background on the HC family
- [[hyper-connections-variants]] — full comparison table
