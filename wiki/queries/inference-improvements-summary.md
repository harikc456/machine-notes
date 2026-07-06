---
title: LLM Inference Improvements — Structured Survey
created: 2026-05-14
updated: 2026-07-06
type: query
tags: [inference, architecture, quantization, kv-cache, speculative, attention, survey, training]
sources: []
confidence: high
---

# LLM Inference Improvements — Structured Survey

> Synthesis across wiki entities/concepts. Each section summarizes the technique landscape and points to detailed pages. For deep KV cache coverage, see [[kv-cache-compression-detail]]; for speculative decoding, see [[speculative-decoding-detail]]. For memory-impact framing, see [[memory-inference-techniques]].

---

## 1. Architecture Improvements

Structural changes that reduce KV cache footprint or compute per token at the model design level. Must be baked in at training time.

### Attention Head Sharing (MQA / GQA)

- **MQA (Multi-Query Attention)**: all heads share a single K/V pair. Extreme KV reduction; hurts model quality at scale.
- **GQA (Grouped-Query Attention)**: groups of heads share K/V pairs. Standard in Llama 3, Mistral. Balances quality vs. KV memory.
- Both address: KV cache grows as `batch × seq_len × n_layers × n_heads × head_dim × 2 × dtype_bytes`. Reducing `n_heads` for K/V is multiplicative.

### MLA + CSA/HCA (DeepSeek)

[[deepseek-v4]] pushed further with **CSA/HCA hybrid**:
- CSA (Compressed Sparse Attention): reduces long-context compute complexity
- HCA (Heavily Compressed Attention): further compresses attention at extreme context lengths
- Result: **27% FLOPs reduction, 10% KV cache** vs V3.2 at 1M-token context

[[deepseek-v3-2]] introduced **DSA (DeepSeek Sparse Attention)** — highly efficient attention for long-context scenarios.

### Sparse MoE

[[mixture-of-experts]]: only `top_k` experts activate per token. FLOPs/token fixed to `top_k / total_experts`. Trade-off: all expert weights must fit in memory (or be paged), even if inactive.

### Attention Residuals (AttnRes)

[[attnres]] — Kimi Team, Mar 2026. Replaces fixed residual accumulation with learned softmax attention over preceding layer outputs. One d-dimensional pseudo-query `w_l` per layer.

**Block AttnRes (practical variant):** Partition L layers into N blocks (N≈8); layers attend over block summaries. Memory: O(Nd) vs O(Ld) for Full AttnRes. Inference I/O: 5.5d/layer vs 34d for mHC (m=4); <2% latency overhead. Training: cross-stage caching reduces pipeline communication from O(C²) to O(P); <4% overhead.

**Results:** Block AttnRes = baseline trained with **1.25× more compute**. 48B model (Kimi Linear, 1.4T tokens): **+7.5 GPQA-Diamond, +3.6 Math, +3.1 HumanEval, +1.7 BBH, +1.1 MMLU** vs baseline. Mitigates PreNorm dilution → bounded output magnitudes and uniform gradient distribution across depth.

---

## 2. Weight Quantization

Compress model weights from training precision to reduce memory footprint and increase throughput.

### The Precision Ladder

| Dtype | Bits | Quality Impact | Notes |
|---|---|---|---|
| BF16 | 16 | Baseline | Preferred over FP16 for training stability (wider range) |
| INT8 | 8 | ~Free | Minimal quality loss; almost always worth it |
| FP8 | 8 | ~Free | Hardware-accelerated on H100+; used in DeepSeek-V4 training |
| INT4 | 4 | Noticeable but acceptable | Common for inference deployment |
| INT2–3 | 2–3 | Significant degradation | Requires careful calibration |

### PTQ vs QAT

- **PTQ (Post-Training Quantization)**: quantize after training. Fast, no retraining. Best with calibration data (GPTQ, AWQ).
- **QAT (Quantization-Aware Training)**: train with simulated quantization. Better quality, expensive.

### The Outlier Problem

Large transformer weights have outlier activations in specific channels that cause disproportionate quantization error. Key approaches:
- **SmoothQuant**: migrates outlier magnitude from activations to weights (scale invariance)
- **GPTQ**: layer-wise second-order quantization — minimizes weight perturbation effect on output
- **AWQ**: activation-aware; identifies and protects important weights

### Attention Computation Quantization

Orthogonal to weight quantization: quantize the Matmuls *within* the attention forward pass. The SageAttention series from Tsinghua applies this as a drop-in replacement for [[flash-attention]]:

| Version | Precision | TOPS (RTX4090) | Speedup vs FA2 | Hardware |
|---|---|---|---|---|
| [[sageattention]] (ICLR 2025) | INT8 Q/K, FP16 P/V | 340 | 2.1× | RTX4090/3090+ |
| [[sageattention2]] (ICML 2025) | INT4 Q/K, FP8 P/V | 481 | 3× | RTX4090/L20+ |
| [[sageattention3]] (NeurIPS 2025) | FP4 Q/K/P/V (microscaling) | 1038 | 5× | RTX5090 (Blackwell) |

Key technique shared across all: outlier smoothing (K-smoothing in SA1; Q+K-smoothing in SA2+). Near-zero end-to-end accuracy loss across LLMs, image-gen, video-gen. See [[quantization]] for full analysis.

---

## 3. KV Cache

The KV cache is the primary memory bottleneck at long contexts and large batch sizes. See [[kv-cache]] for background.

For the full treatment of pruning (H₂O, TriAttention) and quantization (PolarQuant, TurboQuant, SpectralQuant), see [[kv-cache-compression-detail]].

Key results: **TriAttention** achieves 10.7× KV reduction at matched accuracy for long-context reasoning (AIME25, 32K); **SpectralQuant** achieves 5.95× compression at full perplexity quality, strictly dominating TurboQuant (5.02×, −0.50 bits/element). Combining eviction + quantization is complementary — see [[kv-cache-compression-comparison]]. Architectural KV reduction (MQA/GQA/MLA/CSA) — see §1.

---

## 4. Speculative Decoding

Draft-then-verify paradigm for lossless inference speedup. See [[speculative-decoding-detail]] for the full algorithm walkthrough, rejection sampling proof, and self-speculative variants (LayerSkip, SWIFT, DASH).

Key results: **Standard SD** delivers 2–3× lossless speedup (exact distributional match to target). **[[eagle]]** (feature-level AR drafting, Mar 2025): 2.7×–3.5×. **[[eagle-2]]** (dynamic draft trees, Jun 2024): 3.05×–4.26×, 20-40% over EAGLE-1. **[[eagle-3]]** (direct token prediction + training-time test, Apr 2025): up to 6.5×; unlocks data scaling law. **[[dflash]]** (block diffusion parallel drafting, ICML 2026): constant draft cost regardless of draft length; 6×+ lossless, 2.5× over EAGLE-3. **[[dspark]]** (Jun 2026): DFlash backbone + Markov head eliminates suffix decay; confidence-scheduled verification with hardware-aware scheduler; +25–30% accepted length; +51% aggregate throughput in DeepSeek-V4-Flash at 80 TPS/user SLA. **Saguaro** (May 2026): orthogonal — parallelizes speculator and verifier on separate hardware; 30% over SD baselines, up to 5× over AR. **LayerSkip** self-speculative decoding: up to 2.16× speedup, zero extra model memory. **Nemotron 3 Ultra** ([[nemotron-3-ultra]], Jun 2026): shared-weight MTP heads trained jointly with the base model from pretraining, used as a built-in draft mechanism. See [[speculative-decoding]], [[speculative-decoding-detail]] for full detail.

---

## 5. Serving Infrastructure (Algorithmic)

Scheduling and memory management techniques that improve throughput at the serving layer — no model changes required. Full detail (Flash Attention, PagedAttention, RadixAttention, Continuous Batching + Chunked Prefill, DualPath) split out to [[serving-infrastructure-detail]] on 2026-07-06 (page-size threshold).

Key results: **Flash Attention** ([[flash-attention]]): 7.6× speedup on GPT-2, O(N) memory. **PagedAttention** ([[paged-attention]]): 2–4× throughput over TGI via OS-style KV paging. **RadixAttention** ([[radix-attention]]): 2–4× over vLLM for shared-prefix workloads via radix-tree prefix caching. **Continuous batching + chunked prefill** ([[continuous-batching]]): 4–10× decode throughput (SARATHI). **DualPath** ([[dualpath]]): 1.87× offline / 1.96× online throughput for agentic multi-turn serving via dual-path KV loading.

---

## 6. Early Exit / Layer Skipping

See [[early-exit-inference]] for full coverage.

Not all tokens need all layers. Adaptive computation routes easy tokens through fewer layers:

- **Hard early exit**: run to layer e < L; use intermediate LM head. Fast but quality-limited.
- **Self-speculative decoding**: early layers draft, full model verifies — lossless when verification accepts (→ [[inference-kv-speculative]] §4).
- **Per-token layer skipping** (DASH): MDP policy skips individual layers based on token difficulty; input-aware.

Gain depends on task difficulty distribution: summarization and code completion benefit more than complex multi-step reasoning.

---

## 8. Tokenization Efficiency

[[superbpe]] (Liu, Hayase et al., UW/NVIDIA/AI2, COLM 2025) extends BPE to learn "superword" tokens — single tokens spanning multiple whitespace-delimited words (multi-word expressions like *by the way*, *in the long run*). A **pretokenization curriculum** first learns subwords (standard BPE), then lifts the whitespace constraint so BPE can discover cross-word merges.

**Key results at 8B / 200k vocab**: 33% fewer tokens than BPE for the same text → **32% less inference compute**. Average downstream score: +4.0% over BPE on 30 tasks (+8.2% MMLU, 25/30 wins). No architecture or framework changes.

Orthogonal to all other techniques here: SuperBPE reduces the number of forward passes by reducing sequence length; all other methods improve what happens during each forward pass.

---

## 7. Diffusion Language Models as an Inference Paradigm

DLMs offer **parallel token generation** — a fundamentally different inference mode vs. AR decoding. See [[diffusion-language-models]] for the landscape; [[block-diffusion]] and [[i-dlm]] for entity pages.

**BD3-LM** [[block-diffusion]] (ICLR 2025): AR over blocks, discrete diffusion within each block. Restores KV caching and variable-length generation to DLMs. SOTA discrete DLM perplexity on LM1B.

**I-DLM** [[i-dlm]] (Together AI / UIUC / Princeton / Stanford, Apr 2026): converts pretrained AR models to DLMs via introspective-consistency training. Strict causal attention enables direct SGLang / PagedAttention integration. ISD decoding generates N tokens and verifies N prior tokens in a single forward pass. **First DLM to match same-scale AR quality** (Qwen3-8B on MATH-500); 3.1× over SDAR; TPS growth rate 549 vs SDAR 84.

---

## Cross-Cutting Themes

One-line trade-off summary for every technique above, plus the KV/speculative/attention-quant
detail-page techniques, moved to [[inference-technique-tradeoffs]] on 2026-07-06 (page-size
threshold).

## See Also

- [[inference-technique-tradeoffs]] — full cross-cutting trade-off comparison table
- [[kv-cache-compression-detail]] — full KV cache detail (H₂O, TriAttention, PolarQuant, TurboQuant, SpectralQuant, SageAttention family)
- [[speculative-decoding-detail]] — full speculative decoding detail (SD algorithm, EAGLE family, DFlash, DSpark, Saguaro, LayerSkip)
- [[serving-infrastructure-detail]] — full serving-layer detail (Flash Attention, PagedAttention, RadixAttention, Continuous Batching, DualPath)
- [[memory-inference-techniques]] — memory-focused inference survey with quantitative memory impact per technique
- [[memory-inference-research-gaps]] — methodological gaps, untested compositions, Pareto analysis
- [[attnres]] — Attention Residuals entity page
- [[kv-cache]] — KV cache fundamentals and bottleneck analysis
- [[kv-cache-compression-comparison]] — KV compression head-to-head
- [[spectralquant]] — calibrated spectral KV quantization; breaks TurboQuant's data-oblivious bound
- [[triattention]] — pre-RoPE KV compression; best for long-context reasoning
- [[speculative-decoding]] — detailed page with algorithm walkthrough, self-speculative, and SSD sections
- [[eagle]] — feature-level AR speculative drafting; 2.7–3.5× lossless
- [[eagle-2]] — dynamic draft trees; 3.05–4.26×; no extra training
- [[eagle-3]] — direct token prediction + training-time test; up to 6.5×; data scaling law
- [[dflash]] — block diffusion parallel drafting; constant draft cost; 6×+; 2.5× over EAGLE-3
- [[dspark]] — semi-AR (DFlash + Markov head) + confidence scheduler; +51% DeepSeek-V4-Flash throughput
- [[saguaro]] — SSD: parallel drafting + verification on separate hardware
- [[superbpe]] — superword tokenization; 33% fewer tokens; 32% less inference compute; +4.0% downstream
- [[early-exit-inference]] — early exit and layer skipping (LayerSkip, SWIFT, DASH)
- [[layerskip]] — Meta's self-speculative decoding via layer dropout
- [[diffusion-language-models]] — DLM landscape: BD3-LM, I-DLM, DFlash, Mercury
- [[block-diffusion]] — BD3-LM: AR-over-blocks + within-block diffusion
- [[i-dlm]] — introspective DLM: ISD decoding, AR-compatible serving
- [[flash-attention]] — IO-aware tiled attention kernel
- [[sageattention]] — INT8 quantized attention; 2.1× FA2, plug-and-play
- [[sageattention2]] — INT4/FP8 quantized attention; 3× FA2
- [[sageattention3]] — FP4 quantized attention; 5× FA2 (Blackwell)
- [[dualpath]] — dual-path KV loading for agentic inference; 1.87× offline throughput
- [[gqe]] — MoE on GQA query heads; 1.7–1.8× prefill at long context
- [[paged-attention]] — OS-style KV cache memory management
- [[radix-attention]] — radix tree cross-request prefix caching (SGLang)
- [[continuous-batching]] — iteration-level scheduling + chunked prefill
- [[mixture-of-experts]] — MoE fundamentals
- [[quantization]] — weight and attention quantization overview
- [[deepseek-v4]] — CSA+HCA and MoE at production scale
- [[nemotron-3-ultra]] — hybrid Mamba-Attention MoE with shared-weight MTP; ~6× throughput vs SOTA open LLMs
