---
title: KV Cache Compression — Detail
created: 2026-07-06
updated: 2026-07-06
type: query
tags: [inference, kv-cache, quantization, attention, survey]
sources: []
confidence: high
---

# KV Cache Compression — Detail

Split out of [[inference-kv-speculative]] on 2026-07-06 (page-size threshold). Companion to
[[speculative-decoding-detail]]. For the high-level survey, see
[[inference-improvements-summary]]. For memory-impact framing, see
[[memory-inference-techniques]]. For research gaps, see [[memory-inference-research-gaps]].

---

## KV Cache

The KV cache stores past K and V tensors to avoid recomputation during autoregressive decoding. It is the primary memory bottleneck at long contexts and large batch sizes.

See [[kv-cache]] for background.

### KV Cache Pruning

**Goal**: evict tokens that are unlikely to be attended to, keeping only an important subset.

#### H₂O (Heavy-Hitter Oracle)

[[h2o]] — NeurIPS 2023

**Core insight**: attention score distributions are heavy-tailed. A small subset of tokens (heavy hitters) accumulate most of the attention score mass across all heads and layers.

**Algorithm**:
1. Maintain a running sum of attention scores per token across all heads
2. At each step, evict the lowest-scoring token when KV cache is full
3. Always keep recent tokens (recency window)

**Results**:
- Retains ~5% of tokens with negligible quality degradation on most benchmarks
- Up to 29× throughput increase at large batch sizes
- 1.9× lower OOM risk in long-context settings

**Risk**: retrieval tasks (needle-in-haystack) are vulnerable — the "needle" token may not be a heavy hitter in intermediate layers and gets evicted.

**Positioning**: H₂O solves the problem at inference time, with no retraining required. It's a drop-in policy.

See [[kv-cache-compression-comparison]] for side-by-side vs. quantization approaches.

---

### KV Cache Pruning — TriAttention (Pre-RoPE)

[[triattention]] — MIT / NVIDIA / ZJU, Apr 2026

**Problem with post-RoPE importance estimation**: RoPE rotates Q/K vectors by position, making only the most recent queries have up-to-date orientations. This creates a tiny, unstable observation window — H₂O's attention-accumulation signal is unreliable for long-context reasoning tasks (AIME, chain-of-thought).

**Key insight**: In pre-RoPE space, Q/K vectors are **highly concentrated around fixed non-zero centers** that remain stable across positions and contexts. This concentration makes attention logits predictable as a trigonometric series in Q-K distance — usable as a stable importance score that sees the entire sequence, not just a recent window.

**Scoring function**:
- *S_trig(k, Δ)*: trigonometric series from Q/K centers — captures distance preference (which positions each head prefers to attend to)
- *S_norm(k)*: norm-based complement — catches low-norm keys that distance-based scoring would miss
- Weighted by Q/K concentration (Mean Resultant Length R_f): high concentration → trigonometric score dominates; low → norm complement matters more

**Results on AIME25 (Qwen3-8B, 32K-token generation)**:
- **2.5× throughput** at same accuracy as Full Attention
- **10.7× KV memory reduction** at same accuracy as Full Attention
- R-KV achieves only ~half the accuracy at the same efficiency point

**Why it matters**: existing methods (H₂O, R-KV) effectively fail at long-context reasoning tasks. TriAttention makes aggressive KV compression viable for chain-of-thought and mathematical reasoning.

See [[triattention]] for the full method; [[kv-cache-compression-comparison]] for H₂O vs TriAttention vs quantization.

---

### KV Cache Compression (Quantization)

**Goal**: keep all tokens but represent K/V tensors at lower precision.

#### PolarQuant

[[polarquant]] — KV cache quantization via polar coordinate transformation.

**Key insight**: K/V vectors have directional structure. Instead of quantizing Cartesian (x, y) components, transform to polar coordinates (r, θ) and quantize independently.

- Magnitude `r`: varies smoothly → quantizable at low bits
- Phase `θ`: normalized to [0, 2π] → no outliers, uniform distribution, eliminates per-block normalization overhead

**Result**: >4.2× compression ratio with minimal quality loss.

#### TurboQuant

[[turboquant]] — near-optimal online vector quantization.

**Three-stage pipeline**:
1. **Random rotation** (random Hadamard transform): spreads outliers uniformly across all dimensions
2. **MSE quantizer**: near-optimal bit allocation given smoothed distribution
3. **1-bit QJL residual**: captures residual error with 1-bit quantization

**Result**: near-optimal quantization at 3.5 bits per value. Provably within 2.7× of the information-theoretic optimum within the data-oblivious class.

**Shared insight with PolarQuant**: both apply random Hadamard preconditioning to eliminate per-block normalization overhead. The transform makes the distribution easier to quantize without needing runtime statistics.

#### SpectralQuant

[[spectralquant]] — calibrated spectral KV quantization (Gopinath, Sentra/MIT, Apr 2026).

**Core discovery**: across 6 transformer models and 4 families (Qwen, Llama, Mistral, Gemma), KV cache key vectors have effective dimensionality d_eff ≈ 3–4% of head dimension — universally. On 128-dim heads, only ~4 dimensions carry signal; 124 carry noise. This 97% spectral gap is stable (CV = 3.9% across calibration splits).

**Key insight**: TurboQuant's uniform QJL correction on noise dimensions worsens MSE — on dimensions where the true signal is ≈0, correction adds variance without reducing bias. Selectively removing QJL from noise dims simultaneously improves quality *and* compression.

**Algorithm** (5 stages, 15s one-time calibration):
1. Compute empirical covariance Σ̂; extract eigenvectors U; set d_s = ⌈PR(Σ̂)⌉ ≈ 4
2. Spectral rotation: h̃ = U^⊤h; first d_s = signal, rest = noise
3. Non-uniform quantization: Lloyd-Max codebooks separately for signal/noise dims
4. Selective QJL: JL error correction on signal dims only
5. Decompression: reverse quantization + inverse rotation

**Results vs TurboQuant (3-bit)**: +1.7–2.8 pp cosine similarity across all four models; 5.95× vs 5.02× compression (−0.50 bits/element); 4.5× faster attention decoding at 512 tokens. Perplexity identical to uncompressed inference (9.51). Perfect needle-in-haystack to 8K tokens.

---

### Attention Compute Quantization (SageAttention Family)

Distinct from KV *storage* compression: these methods quantize the Q×Kᵀ and P×V Matmuls during the attention forward pass, reducing compute cost while keeping all tokens in cache.

#### [[sageattention]] — INT8 Q/K (ICLR 2025, Tsinghua)

Builds on FlashAttention-2 tiling. Q,K → INT8 (per-block); P,V kept in FP16 with FP16-accumulator (2× faster than FP32-accumulator on RTX4090/3090).

**Key challenge**: K has channel-wise outliers. Fix: K-smoothing — subtract per-channel mean of K before quantization (doesn't affect softmax output since constant row offsets cancel). Per-block scale aligned to FA2 tile granularity.

**Results**: 2.1× FA2, 2.7× xformers; 340 TOPS on RTX4090; near-zero end-to-end loss across LLMs, image-gen, video-gen.

#### [[sageattention2]] — INT4 Q/K + FP8 P/V (ICML 2025, Tsinghua)

**INT4 challenge C1 — narrow range [-7,+7]**: Per-thread quantization (groups tokens by GPU thread per PTX mma layout) with one scale per thread — no extra dequantization instruction. Plus Q-smoothing (subtract per-block Q mean). Combined `smooth Q+K > smooth Q > smooth K > no smoothing`.

**FP8 challenge C2 — FP22 accumulator**: The FP8 mma instruction uses FP22 internally. Fix: two-level accumulation — accumulate each P̃ block into a real FP32 buffer; correct at block boundaries.

**Results**: 3× FA2, 4.5× xformers; 481 TOPS on RTX4090. Hopper variant (SA2-8b) matches FlashAttention3(fp8) speed at higher accuracy.

#### [[sageattention3]] — FP4 NVFP4 Microscaling (NeurIPS 2025, Tsinghua / Shengshu)

Targets Blackwell GPUs (RTX5090 / GB200) with native FP4 Tensor Cores. NVFP4 = E2M1 encoding, 1×16 group quantization, E4M3 scale factors.

**P̃ quantization**: values in [0,1] → poor E4M3 scale factor range. Fix: two-level quantization (per-token normalize to [0, 448×6], then FP4 microscaling). Also explores 8-bit backward pass (SageBwd): lossless for fine-tuning, slower pretraining convergence.

**Results**: 1038 TOPS on RTX5090 — **5× FlashAttention2** on same hardware.

| Version | Q/K | P/V | TOPS | Speedup vs FA2 | Notes |
|---|---|---|---|---|---|
| SA1 | INT8 | FP16 | 340 | 2.1× | RTX4090+ |
| SA2 | INT4 (per-thread) | FP8 (2-level) | 481 | 3× | RTX4090/L20+ |
| SA3 | FP4 (microscaling) | FP4 | 1038 | 5× | Blackwell only |

---

## See Also

- [[inference-kv-speculative]] — overview page linking this and [[speculative-decoding-detail]]
- [[speculative-decoding-detail]] — companion detail page for speculative decoding
- [[inference-improvements-summary]] — full inference survey overview
- [[memory-inference-techniques]] — memory-focused inference survey with quantitative memory impact per technique
- [[memory-inference-research-gaps]] — methodological gaps, Pareto frontier analysis, untested compositions
- [[kv-cache-compression-comparison]] — H₂O vs TriAttention vs PolarQuant vs TurboQuant vs SpectralQuant head-to-head
- [[kv-cache]] — KV cache mechanics and bottleneck analysis
- [[h2o]] — heavy-hitter oracle eviction entity page
- [[triattention]] — pre-RoPE KV eviction entity page
- [[polarquant]] — polar coordinate KV quantization entity page
- [[turboquant]] — data-oblivious near-optimal KV quantization entity page
- [[spectralquant]] — calibrated spectral KV quantization entity page
- [[sageattention]] — INT8 attention compute quantization; 2.1× FA2
- [[sageattention2]] — INT4/FP8 attention compute quantization; 3× FA2
- [[sageattention3]] — FP4 attention compute quantization; 5× FA2 (Blackwell)
