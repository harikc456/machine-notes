---
title: KV Cache Compression Methods Comparison
created: 2026-05-14
updated: 2026-05-16
type: comparison
tags: [kv-cache, quantization, inference, comparison]
sources: [raw/papers/2306.14048v3.pdf, raw/papers/2502.02617v1.pdf, raw/papers/2504.19874v1.pdf, raw/papers/2604.04921v1.pdf]
confidence: high
---

# KV Cache Compression: H₂O vs TriAttention vs PolarQuant vs TurboQuant

Four approaches to reducing KV cache memory, covering two fundamentally different strategies: eviction and quantization.

See [[kv-cache]] for background on why this matters.

## Comparison Table

| Dimension | [[h2o]] | [[triattention]] | [[polarquant]] | [[turboquant]] |
|---|---|---|---|---|
| Strategy | Eviction | Eviction | Quantization | Quantization |
| All tokens retained? | ❌ | ❌ | ✅ | ✅ |
| Memory reduction | ~20× (5% tokens) | 10.7× (matched accuracy) | >4.2× | ~3× (3.5 bits) |
| Q/K space | Post-RoPE | **Pre-RoPE** | N/A | N/A |
| Importance signal | Attention accumulation | Trigonometric series + norms | Polar transform | Random rotation + QJL |
| Calibration needed | No | Yes (offline, cheap) | No | No |
| Needle-in-haystack safe | ❌ (risky) | Better (stable scoring) | ✅ | ✅ |
| Theoretical guarantees | Submodular bound | Empirical | Empirical | Near-optimal bounds |
| Long reasoning | ❌ | ✅ (10.7× on AIME25) | Untested | Untested |
| Venue | NeurIPS 2023 | arXiv Apr 2026 | arXiv Feb 2025 | arXiv Apr 2025 |

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

## TriAttention: Pre-RoPE Eviction

**Core idea**: Work in pre-RoPE space, where Q/K vectors are concentrated around stable non-zero centers ("Q/K concentration"). Approximate attention logits as a trigonometric series in Q-K distance, enabling reliable importance scoring from centers computed via offline calibration.

**Strengths**:
- Stable across positions — not limited to recent queries like post-RoPE methods
- Combines directional information (trigonometric series) with norm signal (norm-based complement)
- 2.5× throughput or 10.7× KV memory reduction on AIME25 while matching Full Attention accuracy

**Weaknesses**:
- Requires offline calibration (cheap but an extra step)
- Still eviction — evicted tokens are lost
- More complex scoring pipeline vs. H₂O's simple accumulation

## Design Convergence

Both PolarQuant and TurboQuant independently arrived at **random preconditioning** as the key to eliminating normalization overhead. The shared insight: after random rotation, the distribution of each coordinate is analytically known and concentrated — making per-block normalization unnecessary. They differ in what they do with the preconditioned vectors (polar transform vs. scalar quantization + QJL).

TriAttention exploits a different structural property: **Q/K concentration in pre-RoPE space** enables attention patterns to be predicted from stable centers, bypassing the rotation instability that limits attention-based eviction methods.

## Recommended Use

- **Maximum compression, non-critical tasks**: H₂O (20× compression)
- **Long-context reasoning (chain-of-thought, AIME)**: TriAttention (10.7× with matched accuracy)
- **Long-context retrieval with compression**: PolarQuant or TurboQuant (retain all tokens)
- **Principled bounds + dual-use (NN search)**: TurboQuant
- **Orthogonal combination**: Quantize retained tokens (TriAttention + TurboQuant for eviction + quantization)

## See Also

- [[kv-cache]] — background and broader landscape
- [[quantization]] — general quantization context
