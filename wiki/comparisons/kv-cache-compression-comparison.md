---
title: KV Cache Compression Methods Comparison
created: 2026-05-14
updated: 2026-05-14
type: comparison
tags: [kv-cache, quantization, inference, comparison]
sources: [raw/papers/2306.14048v3.pdf, raw/papers/2502.02617v1.pdf, raw/papers/2504.19874v1.pdf]
confidence: high
---

# KV Cache Compression: H₂O vs PolarQuant vs TurboQuant

Three approaches to reducing KV cache memory, covering two fundamentally different strategies: eviction and quantization.

See [[kv-cache]] for background on why this matters.

## Comparison Table

| Dimension | [[h2o]] | [[polarquant]] | [[turboquant]] |
|---|---|---|---|
| Strategy | Eviction | Quantization | Quantization |
| All tokens retained? | ❌ | ✅ | ✅ |
| Memory reduction | ~20× (5% tokens) | >4.2× | ~3× (3.5 bits) |
| Normalization overhead | N/A | Eliminated (polar) | Eliminated (rotation) |
| Calibration needed | No | No | No |
| Needle-in-haystack safe | ❌ (risky) | ✅ | ✅ |
| Theoretical guarantees | Submodular bound | Empirical | Near-optimal bounds |
| Custom kernels | No | No | No |
| Venue | NeurIPS 2023 | arXiv Feb 2025 | arXiv Apr 2025 |

## H₂O: Eviction

**Core idea**: Attention scores follow a power-law — retain the "heavy hitters" (high accumulated attention) plus recent tokens; evict the rest.

**Strengths**:
- Highest theoretical compression (retain only 5% of tokens)
- Simple greedy policy, no per-token transform overhead
- Up to 29× throughput improvement demonstrated

**Weaknesses**:
- Irreversible — evicted tokens cannot be recovered
- Fails on tasks requiring retrieval of tokens that initially have low attention scores
- Quality degrades gracefully with compression but not gracefully for adversarial inputs

## PolarQuant: Polar Coordinate Quantization

**Core idea**: After random Hadamard preconditioning, KV embeddings are transformed to polar coordinates. The angle distribution is analytically known (tight, concentrated) — quantize without per-block normalization.

**Strengths**:
- All tokens preserved
- >4.2× compression with best quality vs. SOTA
- Elegant theoretical motivation

**Weaknesses**:
- Recursive polar transform has non-trivial compute cost
- Still an empirical compression (not information-theoretically optimal)

## TurboQuant: Optimal Vector Quantization

**Core idea**: Random rotation makes coordinates i.i.d. Beta-distributed; apply optimal per-coordinate scalar quantizer; correct inner-product bias with 1-bit QJL residual.

**Strengths**:
- All tokens preserved
- **Information-theoretically near-optimal** (proved within 2.7× of lower bound)
- Dual objective: MSE and inner product both optimized
- Applies beyond KV cache to nearest neighbor search

**Weaknesses**:
- Two-stage pipeline adds complexity
- QJL residual is a bit-width overhead
- Near-optimal but not optimal

## Design Convergence

Both PolarQuant and TurboQuant independently arrived at **random preconditioning** as the key to eliminating normalization overhead. The shared insight: after random rotation, the distribution of each coordinate is analytically known and concentrated — making per-block normalization unnecessary. They differ in what they do with the preconditioned vectors (polar transform vs. scalar quantization + QJL).

## Recommended Use

- **Maximum compression, non-critical tasks**: H₂O (20× compression)
- **Long-context retrieval with compression**: PolarQuant or TurboQuant (safe)
- **Principled bounds + dual-use (NN search)**: TurboQuant
- **Orthogonal combination**: Quantize retained tokens (H₂O + TurboQuant)

## See Also

- [[kv-cache]] — background and broader landscape
- [[quantization]] — general quantization context
