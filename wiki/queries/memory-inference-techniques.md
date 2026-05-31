---
title: Memory Reduction Techniques for LLM Inference
created: 2026-05-19
updated: 2026-05-31
type: query
tags: [survey, inference, quantization, kv-cache, attention, sparsity]
sources: []
confidence: high
---

# Memory Reduction Techniques for LLM Inference

Companion to [[memory-reduction-survey]] (training techniques). Covers serving-time memory reduction: weight quantization, KV cache management, architectural KV reduction, speculative decoding memory trade-offs, and expert offloading. For open research questions and untested compositions, see [[memory-inference-research-gaps]].

---

### 9. Weight Quantization (PTQ)

**Core idea:** After training, quantize model weights to lower-bit representations. Weights are loaded quantized; dequantized to FP16 for matmuls (W8A16) or kept quantized (W8A8).

**Memory impact:** INT8 → 2× reduction vs FP16 weights. INT4 → 4× reduction. A 70B FP16 model fits in 140 GB; INT4 shrinks it to ~35 GB — from two A100s to one.

**Key methods:**
- *GPTQ*: Layer-wise second-order quantization. Near-lossless to 4-bit for large models.
- *AWQ*: Identifies and protects weight channels with large activation magnitudes before quantization. Works without calibration data.
- *SmoothQuant*: Migrates quantization difficulty from activations to weights by scaling channels; enables W8A8 (fully quantized) inference.

**Trade-offs:** Quantization error accumulates across layers. Very low bitwidth (2-bit) suffers perplexity degradation. Extreme methods like BitNet (1-bit weights) require training from scratch.

**Wiki cross-reference:** [[quantization]] covers data types, calibration, and the random preconditioning insight (shared with KV quantization).

---

### 10. KV Cache Compression

The KV cache is often the dominant inference memory consumer at long sequence lengths or large batches (can exceed weight memory — see [[kv-cache]] for the formula).

#### 10a. Eviction — H₂O and TriAttention

**Core idea ([[h2o]]):** Not all tokens are equally important. H₂O maintains a budget-bounded cache by evicting tokens with low accumulated attention scores, always keeping recent tokens.

**Memory impact:** Reduces KV cache to a fixed fraction (e.g., 5–20%) of full sequence length. Memory bounded by budget size, independent of actual sequence length.

**Limitation:** Eviction is irreversible. H₂O's post-RoPE importance estimation is unstable at long contexts — only recent queries have up-to-date RoPE orientations, creating a tiny observation window that fails on long reasoning chains (AIME, chain-of-thought).

**TriAttention ([[triattention]], Apr 2026)** addresses the stability problem by working in **pre-RoPE space**, where Q/K vectors are concentrated around fixed centers that remain stable across all positions. Importance is scored via a trigonometric series in Q-K distance:

- Offline calibration: compute Q distribution centers once per model
- At inference: score each key using S_trig (distance preference) + S_norm (magnitude complement), weighted by per-head Q/K concentration
- Results on AIME25 (32K generation): **10.7× KV memory reduction** at matched accuracy vs Full Attention; competing methods achieve only ~half accuracy at the same memory budget

#### 10b. Quantization — PolarQuant, TurboQuant, SpectralQuant

Keep all tokens but reduce bit-width of stored K/V tensors:

- **[[polarquant]]:** Polar coordinate transformation; angular component quantized aggressively. Hadamard preconditioning eliminates per-block normalization overhead. >4.2× compression.
- **[[turboquant]]:** Random rotation (Hadamard) + MSE quantizer + 1-bit QJL residual. Data-oblivious, no calibration. Near-optimal within the data-oblivious class (≤2.7× of information-theoretic bound). 3.19 bits/element, 5.02× compression.
- **[[spectralquant]]:** Calibrated eigenvector rotation + selective QJL on signal dims only. Universal property: KV key vectors have effective dimensionality d_eff ≈ 3–4% of head dim across all tested model families. 15s one-time calibration. **Strictly dominates TurboQuant**: 2.69 bits/element, 5.95× compression, +1.7–2.8 pp cosine similarity. Perplexity identical to uncompressed inference.

**Shared insight:** All three use rotation preconditioning before quantization. PolarQuant and TurboQuant use random rotation (data-oblivious); SpectralQuant uses calibrated eigenvector rotation, enabling selective error correction only on the 3% of dimensions that carry signal — the key advance.

See [[kv-cache-compression-comparison]] for a direct head-to-head comparison.

#### 10c. Architectural KV Reduction

Modify the attention mechanism to produce fewer K/V pairs structurally:

- **MQA:** One shared K/V head for all query heads. KV cache shrinks by factor n_heads. Used in Falcon, Mistral. Extreme reduction; hurts quality at scale.
- **GQA:** Groups of g query heads share one K/V head. KV shrinks by n_heads/g. Used in Llama-3 (default g=8), Gemma.
- **MLA ([[deepseek-v4]]):** Projects K/V into a low-rank latent; only the latent is cached. K/V reconstructed at decode time. 5–13× KV reduction vs. standard MHA.
- **CSA/HCA ([[deepseek-v4]]):** Combines latent compression with selective full-head caching. 3.7–9.8× KV reduction vs DeepSeek-V3.2 at 1M-token context.

#### 10d. Eviction + Quantization

Quantization and eviction are **complementary**: quantization losslessly retains all tokens at reduced precision; eviction aggressively shrinks token count. A combined policy can achieve higher compression than either alone. See [[kv-cache-compression-comparison]] for the Pareto trade-off.

---

### 11. PagedAttention

**Core idea ([[paged-attention]]):** Fixed-size non-contiguous KV "pages" managed by a block table — analogous to OS virtual memory.

**Memory impact:** Eliminates internal and external fragmentation. Traditional serving wastes 20–80% of reserved memory. PagedAttention brings fragmentation to near-zero.

**Throughput effect:** 2–4× higher throughput vs TGI. Enables prefix caching: system prompts computed once, shared across all requests. See [[radix-attention]] for cross-request prefix sharing built on top.

---

### 12. Flash Attention at Inference

Flash Attention (see [[flash-attention]]) also reduces memory at inference time for the **prefill** phase. For long prompts, the N×N attention matrix at prefill is the peak memory moment. FA's O(N) memory profile makes 100K+ token prompts feasible on constrained hardware.

For the decode phase (one new token per step), attention memory is smaller and KV cache management (§10) dominates.

---

### 13. Speculative Decoding — Memory Perspective

**Core idea ([[speculative-decoding]]):** A small draft model generates candidate tokens; the large target model verifies a batch in parallel. Lossless 2–3× throughput improvement.

**Memory impact:** Standard SD *increases* total memory (draft model + target model loaded simultaneously). The benefit is throughput, not memory.

**EAGLE family ([[eagle]] / [[eagle-2]] / [[eagle-3]]):** All three share the same memory model: one small plug-in draft layer (~1 decoder layer) + frozen target. Memory overhead: negligible compared to target model. EAGLE-1/2/3 trade off increasingly on training compute, not inference memory. EAGLE-3 additionally requires a larger draft training dataset.

**DFlash ([[dflash]], ICML 2026):** Replaces the AR draft layer with a block diffusion adapter of comparable size — memory overhead similar to EAGLE. The gain is that draft cost is constant (one parallel forward pass) regardless of speculation length, not sequential like EAGLE. Throughput benefit larger than EAGLE-3 (6×+ vs 6.5×max) but from parallel compute, not less memory.

**Self-speculative ([[layerskip]]):** Uses the target model's own early layers as the draft — zero extra memory overhead. Trade-off: lower draft quality bounded by early-layer representational power.

**Speculative Speculative Decoding ([[saguaro]], May 2026):** Speculator and verifier run on separate hardware simultaneously. Memory per device unchanged vs. standard SD (draft + target split across devices). 30% over SD baselines, up to 5× over AR. Lossless.

---

### 14. Early Exit / Layer Skipping

**Core idea ([[early-exit-inference]]):** Adaptive policies skip later transformer layers for "easy" tokens.

**Memory impact:** Reduces compute and activation memory per token during decode; does not reduce weight memory (all layers still resident). At high skip rates, effective compute and intermediate buffer footprint shrink proportionally.

**Methods:** LayerSkip (Meta, training-time layer dropout), SWIFT (plug-and-play, no retraining), DASH (MDP policy, input-aware).

---

### 15. Residual Architecture Improvements (AttnRes)

[[attnres]] replaces fixed residual accumulation with learned softmax attention over preceding layer outputs — bounding the O(L) hidden-state magnitude growth of standard PreNorm.

**Memory overhead:** Block AttnRes (N≈8) stores N block-summary vectors per token — O(Nd) additional activation memory, negligible vs O(L·T·d) total at training time. Inference I/O: **5.5d per layer** vs 3d for standard residuals.

**Training benefit:** Bounded, periodic output magnitudes and more uniform gradient distribution across depth — more stable training dynamics without extra normalization memory overhead.

---

### 16. Mixture of Experts — Inference Memory

MoE activates only top_k experts per token, but all expert weights must reside in memory (or be offloaded). See [[mixture-of-experts]] for routing and load balancing details.

**Expert offloading:** For models where total expert weight exceeds VRAM (e.g., a 1.6T MoE), experts are offloaded to CPU RAM and prefetched on predicted routing decisions. [[engram]] demonstrates <3% overhead for a related 100B-parameter lookup table with prefetching.

**Continuous batching interaction ([[continuous-batching]]):** Many concurrent requests activate different experts — statistical multiplexing reduces effective per-request expert memory pressure.

---

## Summary Table

| Technique | Phase | Primary Memory Target | Key Trade-off |
|---|---|---|---|
| Weight quantization (INT8/INT4) | Inference | Model weights | Accuracy degradation at very low bits |
| KV eviction (H₂O) | Inference | KV cache | Irreversible; unstable at long context |
| KV eviction (TriAttention) | Inference | KV cache | Offline calibration; best for reasoning |
| KV quantization (PolarQuant/TurboQuant) | Inference | KV cache | Transform compute overhead |
| KV quantization (SpectralQuant) | Inference | KV cache | 15s one-time calibration; strictly better than TurboQuant |
| MQA / GQA / MLA / CSA | Inference | KV cache | Potential attention quality loss (MQA) |
| PagedAttention | Inference | KV cache fragmentation | Minimal (OS paging is near-zero cost) |
| Speculative decoding | Inference | — (increases memory) | Throughput gain, not memory gain |
| EAGLE / EAGLE-2 / EAGLE-3 | Inference | +1 small draft layer (negligible) | Throughput gain via better acceptance; EAGLE-3 unlocks data scaling |
| DFlash (block diffusion draft) | Inference | +1 small diffusion adapter (≈ EAGLE) | Constant draft cost → 6×+ throughput; 2.5× over EAGLE-3 |
| Saguaro (SSD) | Inference | — (split across devices) | Requires separate speculator hardware |
| Self-speculative decoding | Inference | Draft model weights (zero extra) | Lower draft quality |
| Early exit / layer skipping | Inference | Activation memory per token | Weight memory unchanged |
| Expert offloading | Inference | Expert FFN weights | PCIe bandwidth bottleneck |
| Flash Attention (inference) | Inference | Prefill attention matrix | Redundant tile reads (still net faster) |
| AttnRes (Block) | Inference | +O(Nd) depth-attn vs O(3d)/layer standard | Architectural; must be trained in |

---

## See Also

- [[memory-reduction-survey]] — training memory techniques (ZeRO, gradient checkpointing, LoRA, Flash Attention)
- [[memory-inference-research-gaps]] — methodological gaps, untested compositions, and unfalsified shared premises
- [[kv-cache]] — KV cache mechanics and bottleneck analysis
- [[kv-cache-compression-comparison]] — H₂O vs TriAttention vs PolarQuant vs TurboQuant vs SpectralQuant
- [[quantization]] — weight quantization in depth
- [[h2o]] — KV eviction via heavy-hitter oracle
- [[triattention]] — pre-RoPE KV eviction; best for long-context reasoning
- [[polarquant]] — polar KV quantization
- [[turboquant]] — data-oblivious near-optimal KV quantization
- [[spectralquant]] — calibrated spectral KV quantization
- [[paged-attention]] — OS-style KV memory management
- [[radix-attention]] — cross-request prefix sharing
- [[speculative-decoding]] — draft-verify inference speedup
- [[eagle]] — feature-level AR drafting; 2.7–3.5× lossless; minimal memory overhead
- [[eagle-2]] — dynamic draft trees; 3.05–4.26×; no extra training
- [[eagle-3]] — training-time test; up to 6.5×; data scaling law
- [[dflash]] — block diffusion parallel drafting; 6×+; constant draft cost
- [[saguaro]] — parallel draft+verify on separate hardware
- [[layerskip]] — self-speculative decoding via early exit
- [[early-exit-inference]] — early exit and layer skipping landscape
- [[deepseek-v4]] — MLA, CSA/HCA, FP8, Muon at frontier scale
- [[mixture-of-experts]] — MoE architecture and expert offloading
- [[continuous-batching]] — serving scheduler that composes with memory reduction
- [[attnres]] — Attention Residuals
- [[inference-improvements-summary]] — broader inference survey (architecture, serving, DLMs)
- [[inference-kv-speculative]] — deep-dive on KV compression + speculative decoding (including EAGLE family)
