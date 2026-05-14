---
title: KromHC
created: 2026-05-14
updated: 2026-05-14
type: entity
tags: [architecture, residual, training]
sources: [raw/papers/2601.21579v1.pdf]
confidence: high
---

# KromHC

**KromHC: Manifold-Constrained Hyper-Connections with Kronecker-Product Residual Matrices**
*Wuyang Zhou, Yuxuan Gu, Giorgos Iacovides, Danilo Mandic, Imperial College London, arXiv:2601.21579, Jan 2026*

## Overview

KromHC constructs doubly-stochastic residual matrices as **Kronecker products of smaller 2×2 doubly-stochastic matrices** — achieving:
1. Exact doubly-stochasticity (like [[mhc-lite]])
2. Parameter efficiency O(n²C) (like [[mhc]], better than mHC-lite's O(n·n!))
3. Pure PyTorch implementation (like mHC-lite, unlike [[mhc]])

KromHC is the **only method** in the mHC family that simultaneously satisfies all three desiderata.

## Technical Approach

For stream width n = 2^K, KromHC constructs the residual matrix as:

```
H^res = P_1 ⊗ P_2 ⊗ ... ⊗ P_K
```

where each `P_k` is a 2×2 doubly-stochastic matrix (parameterized with 2 learnable scalars via softmax). This requires storing only **K = log₂(n) unique 2×2 matrices** — just 2 scalars each.

For n=8 (K=3): stores 3 × 2 = 6 scalars vs mHC-lite's 8! = 40,320 matrices.

### Why Kronecker Products Work
- Kronecker product of doubly-stochastic matrices is doubly-stochastic (closure property)
- The Birkhoff polytope is preserved under Kronecker products
- The construction is differentiable and implementable with `torch.kron`

## Comparison Summary

| Property | mHC | mHC-lite | KromHC |
|---|---|---|---|
| Exact doubly-stochastic | ⚠️ (approx) | ✅ | ✅ |
| Parameter efficient | ⚠️ O(n³C) | ❌ O(n·n!) | ✅ O(n²C) |
| PyTorch native | ❌ | ✅ | ✅ |

## Results

- Matches or outperforms SOTA mHC variants
- Significantly fewer trainable parameters than mHC-lite
- Eliminates training instabilities from both HC and mHC
- Scales well with increasing stream width n

## See Also

- [[mhc]] — original SK-based approach
- [[mhc-lite]] — convex combination approach (parameter explosion)
- [[hyper-connections]] — HC background
- [[hyper-connections-variants]] — full comparison
