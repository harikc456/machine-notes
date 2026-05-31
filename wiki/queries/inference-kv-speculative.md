---
title: KV Cache Compression and Speculative Decoding — Detail
created: 2026-05-19
updated: 2026-05-31
type: query
tags: [inference, kv-cache, quantization, speculative, survey]
sources: []
confidence: high
---

# KV Cache Compression and Speculative Decoding — Detail

Detailed companion to [[inference-improvements-summary]] for §3 (KV cache pruning and quantization) and §4 (speculative decoding). For memory-impact framing of these techniques, see [[memory-inference-techniques]]. For research gaps and untested compositions, see [[memory-inference-research-gaps]].

---

## 3. KV Cache

The KV cache stores past K and V tensors to avoid recomputation during autoregressive decoding. It is the primary memory bottleneck at long contexts and large batch sizes.

See [[kv-cache]] for background.

### 3a. KV Cache Pruning

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

### 3b. KV Cache Pruning — TriAttention (Pre-RoPE)

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

### 3c. KV Cache Compression (Quantization)

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

## 4. Speculative Decoding

[[speculative-decoding]] — Leviathan et al., ICML 2023

**Problem**: autoregressive LLMs are memory-bandwidth-bound, not compute-bound. The GPU can process many tokens in parallel but the algorithm forces sequential generation.

**Insight**: if the model is bandwidth-bound, extra compute is "free" — as long as we stay within the same memory access budget.

### Algorithm

1. A small **draft model** generates γ tokens autoregressively (fast, cheap)
2. The **target model** verifies all γ tokens in a single parallel forward pass
3. Accept tokens greedily using a rejection sampling scheme:
   - If draft token probability ≤ target probability at that position: **accept**
   - Otherwise: **reject** with probability `1 - p_target/p_draft`; resample from adjusted distribution; discard remaining draft tokens
4. Guaranteed: the output distribution exactly matches the target model (no approximation)

### Why It Works

The target model processes γ+1 tokens in one pass, costing roughly the same memory bandwidth as 1 token. If the draft model has high acceptance rate (α), the expected tokens per target forward pass is `(1 - α^{γ+1}) / (1 - α)` — approaching γ+1 when α is high.

### Results

- **2–3× speedup** on typical text generation benchmarks
- **Exact distributional match** to target model — not an approximation
- Works best when the draft model's distribution is close to the target model's

### The Draft Model

The draft model is the key variable:
- Smaller version of the same model family (e.g., Llama-3.1-8B drafts for Llama-3.1-70B)
- Specialized draft head trained on top of the target model's early layers
- Self-drafting (Medusa): multiple draft heads attached to the target model

### EAGLE Family: Feature-Level and Beyond

The EAGLE series reconsiders *what* the draft model predicts and how it is structured, achieving far higher acceptance rates than vanilla SD.

#### [[eagle]] — Feature-Level AR Drafting (Mar 2025)

**Key insight**: Feature sequences (second-to-top-layer hidden states) are smoother and easier to autoregressively predict than discrete token sequences. EAGLE trains a single lightweight transformer decoder plug-in that autoregressively predicts the next feature, then uses the frozen target LM head to convert it to a token distribution.

**Uncertainty resolution**: Since the next feature depends on which token was sampled (e.g., "am" vs "always" yield different continuations), EAGLE feeds the actual token sequence shifted one time step ahead as an additional input — resolving the sampling ambiguity.

**Draft accuracy**: ~0.8, vs ~0.6 for Medusa and lower for Lookahead. **2.7×–3.5× speedup** on LLaMA2-Chat 70B, lossless in both greedy and non-greedy settings.

#### [[eagle-2]] — Context-Dependent Dynamic Draft Trees (Jun 2024)

EAGLE uses a static draft tree (fixed number of candidates per position). EAGLE-2 observes acceptance rates are **context-dependent** — easy queries need fewer branches, hard ones need more. It leverages the fact that EAGLE's draft model is well-calibrated (its confidence scores ≈ true acceptance rates) to dynamically expand or prune the draft tree at runtime.

**No extra training needed.** Works directly on any EAGLE checkpoint. **3.05×–4.26× speedup**, 20–40% over EAGLE-1.

#### [[eagle-3]] — Training-Time Test (Apr 2025)

**Root cause of EAGLE-1/2 data scaling plateau**: the feature prediction loss (l_fea) constrains the draft model's expressiveness; scaling training data hits diminishing returns because the constraint, not data, is the bottleneck.

**Fix — direct token prediction + multi-layer feature fusion**: remove l_fea; predict tokens directly; fuse low-, mid-, and high-level target features as conditioning.

**Problem this creates**: distribution shift at step 2. Step 1 now produces an unconstrained vector â (not a true feature f̂), so step 2 sees out-of-distribution input at inference time.

**Training-time test**: during training, step 2 is fed â from step 1 (not the ground-truth feature). This exactly matches the inference distribution, closing the shift.

**Result**: **up to 6.5× speedup**, ~1.4× over EAGLE-2. Critically, acceptance rate now scales proportionally with training data — a data scaling law that was absent in EAGLE-1/2.

#### [[dflash]] — Block Diffusion for Parallel Drafting (ICML 2026)

**Root cause of all AR-based SD ceiling**: drafting is still sequential — T_draft = γ × t_step grows linearly with speculation length, capping practical speedups at ~2–3× even with high acceptance rates.

**Fix**: replace AR drafting with a **block diffusion adapter** conditioned on the target model's hidden features. All γ draft tokens are generated in a **single parallel forward pass** (T_draft = t_parallel, constant). "The target knows best" — large AR model hidden states implicitly encode multi-step future-token information; the diffusion adapter reads these features to generate high-quality parallel drafts without being large itself.

**Results on Qwen3-8B (SGLang)**:

| Benchmark | EAGLE-3 | DFlash |
|---|---|---|
| GSM8K | 2.23× | 5.15× |
| Math500 | 2.05× | 6.08× |
| AIME25 | 2.05× | 5.62× |
| HumanEval | 2.17× | 5.14× |
| MBPP | 1.93× | 4.65× |
| MT-Bench | 1.90× | 2.75× |

Over 6× lossless acceleration on math/code, **2.5× over EAGLE-3** across most tasks. MT-Bench lower (2.75×) — conversational tasks have less concentrated future-token signal in hidden states.

### Self-Speculative Decoding

Variant that eliminates the separate draft model. See [[early-exit-inference]] for full coverage.

- **[[layerskip]]** (Meta, 2024): layer dropout training → early layers (0..e) draft, full model verifies; reuses draft KV states; up to 2.16× speedup
- **SWIFT** (2025): no retraining; adaptively selects skip layers per token at runtime; 1.3–1.6×
- **DASH** (2025): MDP policy for per-token layer selection; input-aware

Trade-off: no extra model memory, but draft quality bounded by early-exit representation quality.

### Speculative Speculative Decoding (SSD / Saguaro)

[[saguaro]] — Kumar, Dao, May (Stanford / Princeton / Together AI), May 2026

**The remaining bottleneck in standard SD**: drafting and verification are still sequential — the draft model must wait for verification to finish before generating the next speculation. This idle time is the limiting factor.

**SSD eliminates this by running speculator and verifier on separate hardware in parallel**:
1. Draft model sends speculated tokens to verifier
2. While verification runs, the draft model **predicts the most likely verification outcomes** (k tokens accepted + which bonus token sampled)
3. Pre-speculates for each predicted outcome — stores in a "speculation cache"
4. When verification result arrives: cache hit → return pre-speculated tokens immediately (zero drafting latency); cache miss → synchronous fallback

**Key challenge — predicting the bonus token**: The bonus token is sampled from the residual distribution max(p_target − p_draft, 0). Saguaro uses draft logits to predict the most likely bonus token with ~90% accuracy.

**Results** (Llama-3.1-70B target, Llama-3.2-1B draft, TP=4 H100):
- **30% faster than strongest SD baselines** (vLLM, SGLang)
- **Up to 5× faster than autoregressive decoding**
- Lossless — same output distribution as target model
- Improves Pareto frontier across all batch sizes

**Distinction from tree-based SD**: tree methods increase *verifier* compute; SSD scales *speculator* compute with no extra verification overhead. Orthogonal and combinable.

---

## See Also

- [[inference-improvements-summary]] — full inference survey overview (architecture, serving, DLMs, cross-cutting table)
- [[memory-inference-techniques]] — memory-focused inference survey with quantitative memory impact per technique
- [[memory-inference-research-gaps]] — methodological gaps, Pareto frontier analysis, untested compositions
- [[kv-cache-compression-comparison]] — H₂O vs TriAttention vs PolarQuant vs TurboQuant vs SpectralQuant head-to-head
- [[kv-cache]] — KV cache mechanics and bottleneck analysis
- [[h2o]] — heavy-hitter oracle eviction entity page
- [[triattention]] — pre-RoPE KV eviction entity page
- [[polarquant]] — polar coordinate KV quantization entity page
- [[turboquant]] — data-oblivious near-optimal KV quantization entity page
- [[spectralquant]] — calibrated spectral KV quantization entity page
- [[speculative-decoding]] — speculative decoding concept page
- [[eagle]] — feature-level AR drafting; 2.7–3.5× lossless
- [[eagle-2]] — dynamic draft trees; 3.05–4.26×; no extra training
- [[eagle-3]] — training-time test; up to 6.5×; data scaling law
- [[dflash]] — block diffusion parallel drafting; 6×+; 2.5× over EAGLE-3
- [[saguaro]] — SSD: parallel drafting + verification on separate hardware
- [[layerskip]] — Meta's self-speculative decoding via layer dropout
- [[early-exit-inference]] — early exit and layer skipping (LayerSkip, SWIFT, DASH)
- [[block-diffusion]] — BD3-LM: DFlash's draft engine architecture
- [[diffusion-language-models]] — DLMs as AR model accelerators (DFlash)
