---
title: LLM Inference Improvements — Structured Survey
created: 2026-05-14
updated: 2026-05-19
type: query
tags: [inference, architecture, quantization, kv-cache, speculative, attention, survey, training]
sources: []
confidence: high
---

# LLM Inference Improvements — Structured Survey

> Synthesis across wiki entities/concepts. Each section summarizes the technique landscape and points to detailed pages. For deep KV cache and speculative decoding coverage, see [[inference-kv-speculative]]. For memory-impact framing, see [[memory-inference-techniques]].

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

---

## 3. KV Cache

The KV cache is the primary memory bottleneck at long contexts and large batch sizes. See [[kv-cache]] for background.

For the full treatment of pruning (H₂O, TriAttention) and quantization (PolarQuant, TurboQuant, SpectralQuant), see [[inference-kv-speculative]].

Key results: **TriAttention** achieves 10.7× KV reduction at matched accuracy for long-context reasoning (AIME25, 32K); **SpectralQuant** achieves 5.95× compression at full perplexity quality, strictly dominating TurboQuant (5.02×, −0.50 bits/element). Combining eviction + quantization is complementary — see [[kv-cache-compression-comparison]]. Architectural KV reduction (MQA/GQA/MLA/CSA) — see §1.

---

## 4. Speculative Decoding

Draft-then-verify paradigm for lossless inference speedup. See [[inference-kv-speculative]] for the full algorithm walkthrough, rejection sampling proof, and self-speculative variants (LayerSkip, SWIFT, DASH).

Key results: **Standard SD** delivers 2–3× lossless speedup (exact distributional match to target). **Saguaro** (May 2026) parallelizes speculator and verifier on separate hardware, predicting verification outcomes with ~90% accuracy — 30% faster than strongest SD baselines, up to **5× over AR**. **LayerSkip** self-speculative decoding: up to 2.16× speedup, zero extra model memory. See [[saguaro]], [[speculative-decoding]], [[layerskip]].

---

## 5. Serving Infrastructure (Algorithmic)

Scheduling and memory management techniques that improve throughput at the serving layer — no model changes required.

### Flash Attention

[[flash-attention]] (Dao et al., 2022): tiles Q/K/V into SRAM blocks; computes attention with online softmax without materializing the full N×N matrix. IO complexity: O(Nd + N²/B) vs O(N²d). **7.6× speedup on GPT-2** (A100); O(N) memory — enables long contexts. Now default in PyTorch, HuggingFace, vLLM.

### PagedAttention

[[paged-attention]] (vLLM, 2023): fixed-size KV pages mapped via block tables (OS-style virtual memory). Eliminates 20–80% memory waste from fragmentation. **2–4× throughput over TGI** at same hardware. Enables prefix caching: system prompts computed once, shared across requests.

### RadixAttention

[[radix-attention]] (SGLang, 2023): radix tree maps token sequences → cached KV blocks. Longest matching prefix served on cache hit; LRU eviction clears unused leaves. **2–4× throughput improvement over vLLM** for shared-prefix workloads (system prompts, few-shot examples, agentic pipelines with repeated tool descriptions). Composes with PagedAttention: both active simultaneously.

### Continuous Batching + Chunked Prefill

[[continuous-batching]]: swap finished requests out at the token level (not request level); split long prompts into fixed-size chunks interleaved with decode steps. Results (SARATHI): **1.25–1.91× end-to-end**, **4–10× decode throughput**, 6.29× pipeline bubble reduction for GPT-3.

---

## 6. Early Exit / Layer Skipping

See [[early-exit-inference]] for full coverage.

Not all tokens need all layers. Adaptive computation routes easy tokens through fewer layers:

- **Hard early exit**: run to layer e < L; use intermediate LM head. Fast but quality-limited.
- **Self-speculative decoding**: early layers draft, full model verifies — lossless when verification accepts (→ [[inference-kv-speculative]] §4).
- **Per-token layer skipping** (DASH): MDP policy skips individual layers based on token difficulty; input-aware.

Gain depends on task difficulty distribution: summarization and code completion benefit more than complex multi-step reasoning.

---

## 7. Diffusion Language Models as an Inference Paradigm

DLMs offer **parallel token generation** — a fundamentally different inference mode vs. AR decoding. See [[diffusion-language-models]] for the landscape; [[block-diffusion]] and [[i-dlm]] for entity pages.

**BD3-LM** [[block-diffusion]] (ICLR 2025): AR over blocks, discrete diffusion within each block. Restores KV caching and variable-length generation to DLMs. SOTA discrete DLM perplexity on LM1B.

**I-DLM** [[i-dlm]] (Together AI / UIUC / Princeton / Stanford, Apr 2026): converts pretrained AR models to DLMs via introspective-consistency training. Strict causal attention enables direct SGLang / PagedAttention integration. ISD decoding generates N tokens and verifies N prior tokens in a single forward pass. **First DLM to match same-scale AR quality** (Qwen3-8B on MATH-500); 3.1× over SDAR; TPS growth rate 549 vs SDAR 84.

---

## Cross-Cutting Themes

| Technique | What it trades | Gain |
|---|---|---|
| AttnRes (Block) | O(Nd) depth-attention memory; architectural change at training time | 1.25× compute advantage; +7.5 GPQA-Diamond; mitigates PreNorm dilution |
| GQA/DSA/CSA+HCA | Model quality (marginal) | KV cache ↓ 10–90% |
| MoE | Memory (all experts must load) | FLOPs/token ↓ |
| INT4 weights | Quality (marginal at INT8, moderate at INT4) | Memory ↓ 2–4× |
| H₂O pruning | Retrieval quality | Throughput ↑ 29× |
| TriAttention | Offline calibration; still eviction | Throughput ↑ 2.5× or KV ↓ 10.7× at matched accuracy (reasoning) |
| PolarQuant / TurboQuant | Small quality loss | KV memory ↓ 3–5× |
| SpectralQuant | 15s calibration | KV memory ↓ 5.95×; +1.7–2.8 pp cosine sim vs TurboQuant; 4.5× decode speedup |
| Speculative decoding | Requires draft model | Latency ↓ 2–3× (lossless) |
| Saguaro (SSD) | Separate speculator hardware; prediction overhead | Latency ↓ 5× vs AR, 30% over SD (lossless) |
| Self-speculative (LayerSkip) | Draft quality vs separate model | Latency ↓ 1.3–2.2× (lossless, no extra memory) |
| Flash Attention | Recomputes during backward pass | Attention IO ↓ 7.6×; memory O(N) |
| PagedAttention | Block table indirection overhead | KV fragmentation ↓ ~0%; throughput ↑ 2–4× |
| RadixAttention | Tree lookup overhead | Cross-request prefix reuse; throughput ↑ 2–4× over vLLM |
| Continuous batching + chunked prefill | Scheduling complexity | Decode throughput ↑ 4–10× |
| Early exit / layer skipping | Quality on hard tokens | Latency ↓ 1.3–2× per token |
| BD3-LM (block diffusion) | Fixed block size hyperparameter | Parallel within-block generation + KV caching restored to DLMs |
| I-DLM (introspective DLM) | Training on 4.5B extra tokens | 3.1× over SDAR; matches AR quality; AR-serving-stack compatible |

## See Also

- [[inference-kv-speculative]] — full KV cache and speculative decoding detail (H₂O, TriAttention, PolarQuant, TurboQuant, SpectralQuant, SD algorithm, Saguaro, LayerSkip)
- [[memory-inference-techniques]] — memory-focused inference survey with quantitative memory impact per technique
- [[memory-inference-research-gaps]] — methodological gaps, untested compositions, Pareto analysis
- [[attnres]] — Attention Residuals entity page
- [[kv-cache]] — KV cache fundamentals and bottleneck analysis
- [[kv-cache-compression-comparison]] — KV compression head-to-head
- [[spectralquant]] — calibrated spectral KV quantization; breaks TurboQuant's data-oblivious bound
- [[triattention]] — pre-RoPE KV compression; best for long-context reasoning
- [[speculative-decoding]] — detailed page with algorithm walkthrough, self-speculative, and SSD sections
- [[saguaro]] — SSD: parallel drafting + verification on separate hardware
- [[early-exit-inference]] — early exit and layer skipping (LayerSkip, SWIFT, DASH)
- [[layerskip]] — Meta's self-speculative decoding via layer dropout
- [[diffusion-language-models]] — DLM landscape: BD3-LM, I-DLM, quality gap
- [[block-diffusion]] — BD3-LM: AR-over-blocks + within-block diffusion
- [[i-dlm]] — introspective DLM: ISD decoding, AR-compatible serving
- [[flash-attention]] — IO-aware tiled attention kernel
- [[paged-attention]] — OS-style KV cache memory management
- [[radix-attention]] — radix tree cross-request prefix caching (SGLang)
- [[continuous-batching]] — iteration-level scheduling + chunked prefill
- [[mixture-of-experts]] — MoE fundamentals
- [[quantization]] — weight quantization overview
- [[deepseek-v4]] — CSA+HCA and MoE at production scale
