---
title: KV Cache Compression Methods Comparison
created: 2026-05-14
updated: 2026-05-19
type: comparison
tags: [kv-cache, quantization, inference, comparison]
sources: [raw/papers/2306.14048v3.pdf, raw/papers/2502.02617v1.pdf, raw/papers/2504.19874v1.pdf, raw/papers/2604.04921v1.pdf, papers/spectralquant.pdf]
confidence: high
---

# KV Cache Compression: H₂O vs TriAttention vs PolarQuant vs TurboQuant vs SpectralQuant

Five approaches to reducing KV cache memory, covering two fundamentally different strategies: eviction and quantization.

See [[kv-cache]] for background on why this matters.

## Comparison Table

| Dimension | [[h2o]] | [[triattention]] | [[polarquant]] | [[turboquant]] | [[spectralquant]] |
|---|---|---|---|---|---|
| Strategy | Eviction | Eviction | Quantization | Quantization | Quantization |
| All tokens retained? | ❌ | ❌ | ✅ | ✅ | ✅ |
| Memory reduction | ~20× (5% tokens) | 10.7× (matched accuracy) | >4.2× | 5.02× | **5.95×** |
| Q/K space | Post-RoPE | **Pre-RoPE** | N/A | N/A | N/A |
| Importance signal | Attention accumulation | Trigonometric series + norms | Polar transform | Random rotation + QJL | **Calibrated eigenvectors + selective QJL** |
| Calibration needed | No | Yes (offline, cheap) | No | No | **Yes (15s)** |
| Needle-in-haystack safe | ❌ (risky) | Better (stable scoring) | ✅ | ✅ | ✅ |
| Theoretical guarantees | Submodular bound | Empirical | Empirical | Near-optimal bounds (data-oblivious) | Bias-variance proof for selective QJL |
| Long reasoning | ❌ | ✅ (10.7× on AIME25) | Untested | Untested | Untested |
| Bits/element | N/A | N/A | N/A | 3.19 | **2.69** |
| Venue | NeurIPS 2023 | arXiv Apr 2026 | arXiv Feb 2025 | arXiv Apr 2025 | arXiv Apr 2026 |

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
- **Information-theoretically near-optimal** (proved within 2.7× of lower bound within data-oblivious class)
- Dual objective: MSE and inner product both optimized
- Applies beyond KV cache to nearest neighbor search
- Zero calibration cost

**Weaknesses**:
- Uniform QJL on all dimensions injects variance into noise dims (where it worsens MSE)
- Near-optimal within data-oblivious class — data-aware methods can do better

## SpectralQuant: Calibrated Spectral Quantization

**Core idea**: KV cache key vectors have d_eff ≈ 3–4% of head dimension carrying signal (universally across model families). Rotate into eigenvector coordinates, apply non-uniform quantization, and apply QJL error correction *only* to signal dimensions. 15s one-time calibration.

**Strengths**:
- All tokens preserved
- **Strictly dominates TurboQuant**: +1.7–2.8 pp cosine similarity *and* 18.6% better compression at same bit budget
- Perplexity identical to uncompressed inference
- 4.5× faster attention decoding than TurboQuant at 512 tokens
- Counterintuitive insight: *removing* QJL from noise dims improves quality (bias-variance argument)

**Weaknesses**:
- 15s calibration required (one-time per model, cheap but non-zero)
- Calibration stability: CV = 3.9% across splits (highly stable, but data-oblivious methods have zero variance here)

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

PolarQuant, TurboQuant, and SpectralQuant all use **rotation preconditioning** to eliminate normalization overhead. TurboQuant and PolarQuant use random rotation (data-oblivious); SpectralQuant uses calibrated eigenvector rotation (data-aware, 15s cost). The spectral concentration SpectralQuant exploits is empirically universal — d_eff ≈ 3-4% of head dim across all tested model families.

SpectralQuant's core advance over TurboQuant: recognizing that uniform QJL on 97% noise dimensions worsens MSE by injecting variance without reducing bias. This is formally provable via bias-variance decomposition and confirmed empirically (+3.0 pp cosine similarity just from removing QJL on noise dims).

TriAttention exploits a different structural property: **Q/K concentration in pre-RoPE space** enables attention patterns to be predicted from stable centers, bypassing the rotation instability that limits attention-based eviction methods.

## Recommended Use

- **Maximum compression, non-critical tasks**: H₂O (20× compression)
- **Long-context reasoning (chain-of-thought, AIME)**: TriAttention (10.7× with matched accuracy)
- **Long-context quantization, zero setup**: TurboQuant (data-oblivious bound)
- **Long-context quantization, best quality+compression**: SpectralQuant (15s calibration, strictly better than TurboQuant)
- **Principled bounds + dual-use (NN search)**: TurboQuant
- **Orthogonal combination**: Quantize retained tokens (TriAttention + SpectralQuant for eviction + quantization)

## See Also

- [[kv-cache]] — background and broader landscape
- [[quantization]] — general quantization context
- [[spectralquant]] — full SpectralQuant entity page
