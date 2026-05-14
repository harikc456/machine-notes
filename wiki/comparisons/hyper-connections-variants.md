---
title: Hyper-Connections Variants Comparison
created: 2026-05-14
updated: 2026-05-14
type: comparison
tags: [architecture, residual, training, comparison]
sources: [raw/papers/2512.24880v2.pdf, raw/papers/2601.05732v1.pdf, raw/papers/2601.21579v1.pdf]
confidence: high
---

# Hyper-Connections Variants: mHC vs mHC-lite vs KromHC

All three methods aim to constrain HC residual matrices to the Birkhoff polytope (doubly-stochastic matrices) to ensure training stability. They differ in how they achieve this and at what cost.

See [[hyper-connections]] for background on why this constraint matters.

## Comparison Table

| Criterion | [[mhc]] | [[mhc-lite]] | [[kromhc]] |
|---|---|---|---|
| Exact doubly-stochastic | ⚠️ approximate | ✅ exact | ✅ exact |
| Parameter complexity | O(n³C) | O(n·n!) | O(n²C) |
| PyTorch native | ❌ (custom CUDA) | ✅ | ✅ |
| SK iterations needed | 20 | 0 | 0 |
| Source | DeepSeek-AI | Yang & Gao (Michigan/Harvard/NTT/NTU) | Zhou et al. (Imperial) |
| Date | Dec 2025 | Jan 2026 | Jan 2026 |

## Deep Dive

### mHC (Sinkhorn-Knopp)
The original DeepSeek approach: parameterize `H^res` as `exp(W)` then apply 20 alternating row/column normalizations.

**The approximation gap problem** (from mHC-lite paper):
```
Input: [[0.5, α, α], [0.5, α, α], [α, 1, 1]] with α = 1e-13
After 20 SK iters: column sums ≈ [1.92, 0.59, 0.59]  (not 1.0)
```
This error accumulates across depth, potentially undermining the stability that mHC targets.

**Also**: requires specialized fused CUDA kernels to amortize repeated kernel launches across 20 iterations.

### mHC-lite (Birkhoff-von Neumann Reparameterization)
Birkhoff-von Neumann: every doubly-stochastic matrix is a convex combination of permutation matrices. So parameterize directly:
```
H^res = softmax(a) · [P_1, P_2, ..., P_{n!}]
```
Exact by construction. Pure matrix multiply. No special kernels.

**The parameter explosion**: For n=8, need 8! = 40,320 unique 8×8 matrices — storing 40,320 × 8 × 8 = 2,580,480 values per layer. Factorial growth makes large n infeasible.

### KromHC (Kronecker Product)
For n = 2^K, decompose the residual matrix as Kronecker product of K 2×2 DS matrices:
```
H^res = P_1 ⊗ P_2 ⊗ ... ⊗ P_K
```
Each `P_k` is `[[a, 1-a], [1-a, a]]` for one learnable scalar (passed through sigmoid). 

For n=8 (K=3): 3 scalars total (vs. mHC-lite's millions).

Doubly-stochastic by construction: Kronecker product of DS matrices is DS.

## Performance

All three variants **match or exceed standard mHC** in downstream performance. The differences are stability, throughput, and portability — not accuracy.

KromHC additionally eliminates the residual instabilities that mHC-lite's paper shows still exist in mHC (and to a lesser extent in vanilla HC).

## Verdict

**For new projects**: KromHC is strictly better than mHC on all dimensions except it wasn't in production first. Use KromHC if implementing from scratch (or mHC-lite if n is small). mHC remains relevant only for compatibility with existing DeepSeek infrastructure.

**For [[deepseek-v4]]**: Uses mHC. Future versions may adopt KromHC or mHC-lite.

## See Also

- [[hyper-connections]] — background on HC family
- [[mhc]] — DeepSeek original
- [[mhc-lite]] — exact, portable, parameter-heavy
- [[kromhc]] — exact, portable, parameter-efficient
- [[deepseek-v4]] — production deployment of mHC
