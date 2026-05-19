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

> Synthesis across wiki entities/concepts. Each section summarizes the technique landscape and points to detailed pages.

---

## 1. Architecture Improvements

Structural changes that reduce the KV cache footprint or compute per token at the model design level.

### Attention Head Sharing (MQA / GQA)

- **MQA (Multi-Query Attention)**: all heads share a single K/V pair. Extreme KV reduction; hurts model quality at scale.
- **GQA (Grouped-Query Attention)**: groups of heads share K/V pairs. Standard in modern LLMs (Llama 3, Mistral). Balances quality vs. KV memory.
- Both address the same bottleneck: KV cache grows as `batch × seq_len × n_layers × n_heads × head_dim × 2 × dtype_bytes`. Reducing `n_heads` for K/V is multiplicative.

### MLA + CSA/HCA (DeepSeek)

[[deepseek-v3-2]] introduced **DSA (DeepSeek Sparse Attention)** — highly efficient attention mechanism reducing computational complexity for long-context scenarios.

[[deepseek-v4]] pushed further with **CSA/HCA hybrid**:
- CSA (Compressed Sparse Attention): reduces long-context compute complexity
- HCA (Heavily Compressed Attention): further compresses attention at extreme context lengths
- Result: **27% FLOPs reduction, 10% KV cache** vs V3.2 at 1M-token context

### Sparse MoE

[[mixture-of-experts]]: only `top_k` experts activate per token. The rest of the model sits idle during inference for that token.

- Capacity scales with total parameters, but FLOPs per token are fixed to `top_k / total_experts`
- DeepSeek-V4 uses fine-grained expert routing (128 routed + 1 shared expert)
- Tradeoff: all expert weights must fit in memory (or be paged), even if inactive

### Attention Residuals (AttnRes)

[[attnres]] — Kimi Team, Mar 2026

**Problem**: standard PreNorm residuals accumulate all layer outputs with fixed unit weights, causing hidden-state magnitudes to grow as O(L) with depth. Deeper layers must produce increasingly large outputs to influence the residual stream, progressively burying earlier representations.

**Insight**: the depth dimension mirrors the sequence dimension. Just as Transformers replaced RNNs with attention for sequence modeling, AttnRes replaces fixed residual accumulation with learned **softmax attention over preceding layer outputs**:

```
h_l = Σ_{i=0}^{l-1}  α_{i→l} · v_i      (α = softmax of learned dot products)
```

One d-dimensional pseudo-query `w_l` per layer is the only new parameter — negligible overhead.

**Block AttnRes (practical variant):** Partition L layers into N blocks (N≈8). Layers attend over block summaries rather than all L individual outputs, reducing memory and communication from O(Ld) to O(Nd).

**Infrastructure for scale:**
- Training: cross-stage caching reduces pipeline communication from O(C²) to O(P); overhead <4%
- Inference: two-phase compute (batched inter-block + sequential intra-block via online softmax merge); total I/O **5.5d per layer** vs 34d for mHC (m=4 streams); latency overhead <2%

**Results:**
- Scaling laws: Block AttnRes = baseline trained with **1.25× more compute**
- 48B model (Kimi Linear, 1.4T tokens) vs baseline: **+7.5 GPQA-Diamond, +3.6 Math, +3.1 HumanEval, +1.7 BBH, +1.1 MMLU**
- Training dynamics: mitigates PreNorm dilution → bounded output magnitudes, uniform gradient distribution across depth

**Why it matters for inference quality**: better depth-wise information flow, especially for multi-step reasoning tasks where later layers need access to specific earlier representations. AttnRes is an architectural change baked at training time (like GQA/MoE) — not a post-hoc optimization.

---

## 2. Weight Quantization

Compress model weights from their training precision to reduce memory footprint and increase throughput.

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

### Relationship to KV Cache

Weight quantization and KV cache quantization are **separate** problems:
- Weight quantization: reduces static memory for model parameters
- KV cache quantization (→ §3): reduces dynamic memory per request

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

**Result**: near-optimal quantization at 3.5 bits per value.

**Shared insight with PolarQuant**: both apply a random Hadamard preconditioning step to eliminate per-block normalization overhead. The transform makes the distribution easier to quantize without needing runtime statistics.

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

## 5. Serving Infrastructure (Algorithmic)

Scheduling and memory management techniques that improve throughput at the serving layer — no model changes required.

### Flash Attention

[[flash-attention]] (Dao et al., 2022) — IO-aware attention kernel.

**Problem**: standard attention materializes the full N×N attention matrix in HBM, making attention IO-bound (not compute-bound).

**Solution**: tile Q, K, V into SRAM blocks; compute attention with online softmax without ever writing the full matrix to HBM. IO complexity drops from O(N²d) to O(Nd + N²/B).

- 7.6× speedup on GPT-2 (A100); 1.6× on T5
- O(N) memory (vs O(N²)) — enables longer sequences or larger batches
- Now default in PyTorch, HuggingFace, vLLM; on FA4 for Blackwell GPUs

### PagedAttention

[[paged-attention]] (vLLM, Kwon et al., 2023) — OS-style virtual memory for KV cache.

**Problem**: contiguous KV cache allocation causes 60–80% memory waste from fragmentation. Short requests leave holes; long requests can't start until contiguous space is available.

**Solution**: fixed-size pages mapped via block tables (like OS page tables). Non-contiguous allocation, copy-on-write for parallel sampling, prefix sharing across requests.

- 2–4× throughput over HuggingFace TGI at same hardware
- Enables prefix caching: system prompts computed once, shared across all requests

### RadixAttention

[[radix-attention]] (SGLang, Zheng et al., 2023) — cross-request KV prefix caching via radix tree.

**Problem**: PagedAttention eliminates within-request fragmentation, but every new request still recomputes its prefix (system prompt, few-shot examples, earlier turns) from scratch.

**Solution**: a radix tree maps token sequences → cached KV blocks. On each new request, the longest matching prefix is looked up in O(prefix_length); only the unmatched suffix is prefilled. Nodes are reference-counted; LRU eviction clears the least recently used leaves when memory is full.

- **2–4× throughput improvement over vLLM** on shared-prefix workloads
- Greatest gains: chatbot deployments (system prompt shared across all requests), batch inference with shared few-shot examples, agentic pipelines with repeated tool descriptions
- Composes with PagedAttention: both active simultaneously — PagedAttention manages block layout, RadixAttention manages prefix reuse

### Continuous Batching + Chunked Prefill

[[continuous-batching]] — iteration-level scheduling with SARATHI-style chunked prefill.

**Problem**: static batching idles GPU slots when any request finishes; long prompts stall decode requests ("prefill monopolization").

**Two-part solution**:
1. **Continuous batching**: swap finished requests out at the token level, not request level; zero padding waste via ragged batching
2. **Chunked prefill** (SARATHI): split long prompts into fixed-size chunks; interleave with decode steps

Results (SARATHI): 1.25–1.91× end-to-end, 4–10× decode throughput improvement; 6.29× pipeline bubble reduction for GPT-3 with pipeline parallelism.

---

## 6. Early Exit / Layer Skipping

See [[early-exit-inference]] for full coverage.

Not all tokens need all layers. Adaptive computation routes easy tokens through fewer layers:

- **Hard early exit**: run to layer e < L; use intermediate LM head. Fast but quality-limited.
- **Self-speculative decoding**: early layers draft, full model verifies — lossless when verification accepts (→ §4).
- **Per-token layer skipping** (DASH): MDP policy skips individual layers based on token difficulty; input-aware.

Gain depends on task difficulty distribution: summarization and code completion benefit more than complex multi-step reasoning.

---

## 7. Diffusion Language Models as an Inference Paradigm

Diffusion language models (DLMs) offer **parallel token generation** — a fundamentally different inference mode vs. autoregressive decoding. Two recent systems bring DLMs to parity with AR quality while delivering real serving efficiency gains.

See [[diffusion-language-models]] for the landscape; [[block-diffusion]] and [[i-dlm]] for entity pages.

### Block Diffusion (BD3-LM)

[[block-diffusion]] — Arriola et al., ICLR 2025

Interpolates between AR and diffusion by being **autoregressive over blocks** and applying discrete diffusion **within each block**. Key inference properties restored vs. standard DLMs:
- **KV caching**: block-causal attention means prior blocks' KV pairs are cached exactly like AR
- **Variable-length generation**: block-AR structure supports arbitrary sequence lengths
- **Within-block parallelism**: all L' tokens in a block denoised in parallel, with parallel token sampling

Sets SOTA perplexity among discrete DLMs on LM1B; with tuned noise schedule matches AR perplexity.

### I-DLM (Introspective Diffusion Language Model)

[[i-dlm]] — Yu, Jian et al. (Together AI / UIUC / Princeton / Stanford), Apr 2026

Converts pretrained AR models into DLMs using **introspective-consistency training** (causal attention + logit shift + all-masked objective). Key serving properties:

- **AR-compatible inference**: strict causal attention enables direct integration into SGLang, continuous batching, paged KV cache — no special DLM serving stack needed
- **Introspective Strided Decoding (ISD)**: single forward pass simultaneously generates N new tokens (masked positions) and verifies N prior tokens (introspection positions) against the causal anchor distribution
- **Compute efficiency**: ISD is the only DLM decoding method above the efficiency break-even line at practical acceptance rates (p ≥ 0.83)

**Results** vs. prior DLMs (8B model, concurrency=32):
- 3.1× higher throughput than SDAR; 4× over LLaDA-2.1-mini (16B)
- First DLM to match strong same-scale AR quality (matches Qwen3-8B on MATH-500)
- TPS growth rate 549 (vs SDAR: 84) — batching efficiency scales with TPF

**Why it matters for inference**: I-DLM reframes the AR-vs-diffusion tradeoff — rather than choosing between quality (AR) and parallelism (DLM), ISD delivers both via a single unified forward pass that generates and verifies simultaneously.

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
| PolarQuant / TurboQuant | Small quality loss | KV memory ↓ 3–4× |
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

- [[attnres]] — Attention Residuals: depth-wise softmax attention over preceding layers; Block AttnRes is a drop-in for training
- [[kv-cache]] — KV cache fundamentals and bottleneck analysis
- [[kv-cache-compression-comparison]] — H₂O vs TriAttention vs PolarQuant vs TurboQuant head-to-head
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
