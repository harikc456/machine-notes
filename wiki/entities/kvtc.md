---
title: kvtc
created: 2026-06-29
updated: 2026-06-29
type: entity
tags: [kv-cache, quantization, inference]
sources: [raw/papers/2511.01815v2.md]
confidence: high
---

# kvtc

**KV Cache Transform Coding for Compact Storage in LLM Inference**
*Konrad Staniszewski, Adrian Łańcucki — NVIDIA / University of Warsaw, ICLR 2026, arXiv:2511.01815*

## Overview

`kvtc` applies classical **transform coding** (the theory behind JPEG/image codecs) to KV cache compression. Instead of quantizing K and V tensors directly in the activation space, it first decorrelates features via PCA, then applies dynamic-programming bit allocation per principal component, then entropy-codes the result. The analogy is direct: PCA ↔ DCT, DP bit allocation ↔ JPEG quantization table, DEFLATE ↔ JPEG Huffman coding.

No model parameters are changed. Calibration is done once per (model, compression ratio). Compression and decompression happen between decode steps or at prefill→decode transition boundaries; the model always operates on uncompressed KV caches.

## Method

### Stage 1: Feature Decorrelation (PCA)
Compute SVD of centered calibration KV data: `C − μ = UΣVᵀ`. Store the orthonormal projection matrix V. At inference: `D = (X − μ)V` maps the KV cache into decorrelated principal components.

Cross-head alignment first: key heads from different attention layers share a common subspace up to orthogonal rotation. Aligning heads before PCA (via Procrustes-solved rotation R*) improves compression and reduces per-head calibration cost.

### Stage 2: Adaptive Quantization (Dynamic Programming)
DP allocates bits per principal component proportional to variance explained. Components with high variance (the "signal" dimensions) receive more bits; low-variance dimensions receive fewer or none. This mirrors allocating more JPEG bits to low-frequency DCT coefficients.

### Stage 3: Entropy Coding (DEFLATE/nvCOMP)
The quantized symbol stream is compressed with DEFLATE, exploiting remaining statistical structure. DEFLATE contributes 1.8–22× additional compression beyond quantization depending on the compression ratio setting.

### Special Handling
- **Sliding window**: W=128 most recent tokens excluded from compression (local coherence).
- **Attention sinks**: s=4 oldest tokens excluded from compression (critical for attention accuracy at high ratios).
- **Decompression**: inverse projection `V⊤` can be applied layer-by-layer as submatrices, enabling decoding to begin early during streaming prefill.

## Results

| Config | Compression Ratio | Accuracy |
|---|---|---|
| kvtc_8× | 9–10× | Near-lossless |
| kvtc_16× | 18–22× | ~Lossless on most benchmarks |
| kvtc_20× | ~20× | Maintained reasoning + long-context |
| kvtc_32× | 34–44× | Modest accuracy drop |
| kvtc_64× | 64–88× | Moderate accuracy drop |

- Benchmarks: GSM8K, MATH-500, LiveCodeBench, MMLU, LongBench, Qasper, RULER
- Models: Llama 3.1 8B, Llama 3.3 70B Instruct, Mistral NeMo 12B, R1-Qwen 2.5
- Consistently outperforms token eviction, inline INT8/INT4 quantization, and per-prompt SVD methods
- Calibration: ~10 min on H100 for 12B model; PCA basis adds 2.4% of model parameter count (Llama 3.3 70B)

## Serving Use Case

In multi-turn chat, reusable KV caches are shared via prefix matching. At moderate batch sizes, generated KV caches shorten their hot/warm residency on GPU HBM. A 20× lifetime extension from compression can determine whether a KV cache remains hot until it's needed again vs. requiring recomputation. KV transfer from prefill to decode nodes is also reduced proportionally.

## Contrast with Related Methods

[[turboquant]] and [[polarquant]] apply data-oblivious random rotation then quantize in a rotation-invariant space. kvtc uses a **calibrated** PCA basis — data-aware, more compact, explicitly captures the low-rank structure of KV caches. The tradeoff: calibration cost (one-time, cheap) vs. zero-cost but suboptimal random rotation.

[[spectralquant]] is conceptually similar: both exploit the spectral structure of KV caches, both use calibrated rotation. SpectralQuant focuses on inner-product preservation for attention computation quality. kvtc targets storage compression with entropy coding and is oriented toward serving-layer cache lifetime extension.

Token eviction methods ([[h2o]], [[triattention]], [[lazyeviction]]) irreversibly discard tokens; kvtc retains all tokens at high fidelity.

## See Also

- [[kv-cache]] — background and broader compression landscape
- [[quantization]] — general quantization overview
- [[kv-cache-compression-comparison]] — side-by-side comparison
- [[spectralquant]] — calibrated spectral quantization; different objective (attention quality vs. storage)
- [[turboquant]] — data-oblivious rotation baseline that kvtc dominates
