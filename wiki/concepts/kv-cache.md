---
title: KV Cache
created: 2026-05-14
updated: 2026-07-21
type: concept
tags: [kv-cache, inference, attention, quantization]
sources: [raw/papers/2306.14048v3.pdf, raw/papers/2502.02617v1.pdf, raw/papers/2504.19874v1.pdf, raw/papers/2604.04921v1.pdf, raw/papers/2602.21548v2.md, raw/papers/2606.20945v2.md, raw/papers/2506.15969v3.md, raw/papers/2511.01815v2.md, raw/papers/2606.15007v1.pdf, raw/papers/2510.26692v2.md, raw/papers/2607.02770v1.md]
confidence: high
---

# KV Cache

## What It Is

During autoregressive decoding, attention layers compute Key (K) and Value (V) embeddings for every input token. Rather than recompute these at every generation step, models **cache** them — the KV cache stores all past K and V tensors across all layers and attention heads.

**Memory cost**: `batch_size × seq_len × n_layers × n_heads × head_dim × 2 × dtype_bytes`

For a 30B-parameter model with batch=128 and seq_len=1024: ~180 GB of KV cache — often larger than model weights.

## Why It's a Bottleneck

1. **Scales with sequence length** — long-context models (1M tokens in [[deepseek-v4]]) face enormous KV memory
2. **Scales with batch size** — each concurrent request needs its own cache
3. **Memory bandwidth bound** — loading the full KV cache per decode step dominates latency

## Compression Approaches

### Eviction (token-level compression)
Keep only a subset of tokens' KV pairs:
- [[h2o]]: retain "heavy hitter" tokens (high accumulated attention) + recency window
- [[triattention]]: score keys via trigonometric series in pre-RoPE space; avoids RoPE rotation instability that limits attention-based methods; 2.5× throughput or 10.7× KV reduction at matched accuracy on AIME25
- [[lazyeviction]]: observation window-based lagged eviction for reasoning tasks; tracks Maximum Recurrence Interval (MRI) per token to retain tokens during low-attention intervals before their next recurrence; 50–70% KV reduction at matched accuracy on GSM8K/MATH500
- **Limitation**: irreversible eviction can miss critical tokens in retrieval tasks — though scoring quality matters; TriAttention's pre-RoPE approach is most stable, LazyEviction's MRI tracking is best for reasoning

### Quantization / Compression (precision reduction)
Reduce the bit-width of stored K and V tensors:
- [[polarquant]]: polar coordinate transform eliminates normalization overhead; >4.2× compression
- [[turboquant]]: random rotation + MSE quantizer + 1-bit QJL residual; neutral at 3.5 bits
- [[spectralquant]]: calibrated eigenvector rotation + selective QJL on 3% signal dims; 5.95× compression, strictly dominates TurboQuant
- [[kvtc]]: transform coding (PCA + DP bit allocation + DEFLATE); 20× compression at lossless accuracy; 40× at modest drop; oriented toward storage/serving (extending KV cache lifetime on-GPU and reducing prefill→decode transfer bandwidth)
- Traditional methods: per-block normalization adds >1 bit overhead
- QJL: 1-bit sketching, data-oblivious

### Architectural Reduction
Modify the model to produce fewer K/V pairs:
- **Multi-Query Attention (MQA)**: single shared K/V head across all query heads
- **Grouped-Query Attention (GQA)**: groups of query heads share one K/V head
- **Multi-Head Latent Attention (MLA)**: compress K/V into low-rank latent space (DeepSeek)
- **CSA/HCA** in [[deepseek-v4]]: 3.7–9.8× reduction in KV cache vs DeepSeek-V3.2
- **Projection sharing (Q-K=V)**: force K=V at the projection level — only K needs to be cached, V is reused. [[qkv-projection-sharing]] (ICML 2026): 50% cache, +3.1% PPL at 300M. Orthogonal to head sharing — combined Q-MQA achieves 96.9% cache reduction at +4.8% PPL.
- **Hybrid linear/full attention**: [[kimi-linear]]'s Kimi Delta Attention (channel-wise gated delta rule) interleaved 3:1 with full attention (NoPE) cuts KV cache by up to 75% and gives 6.3× faster TPOT at 1M-token decoding, joining [[nemotron-3-ultra]] and [[deepseek-v4]] as hybrid-attention approaches to the same bottleneck.
- **KV cache sharing + local:global ratio tuning**: [[gemma-4]] reuses values as keys in global attention layers and tunes a 5:1 local-sliding-window:global-attention ratio, cutting global KV cache footprint by up to 37.5% without a fundamentally new attention mechanism.

### Offloading
Move KV cache from GPU HBM to CPU RAM or disk:
- [[engram]] prefetching demonstrates <3% overhead for 100B parameter lookup table
- [[deepseek-v4]] supports on-disk KV cache storage

### Serving-Layer Loading (Agentic Workloads)
At the infrastructure level, KV cache loading bandwidth — not compression — becomes the bottleneck for agentic (multi-turn) inference:
- KV-cache hit rates ≥95%; cache-compute ratio ~22 GB/PFLOP (DeepSeek-V3.2); storage NICs saturate on prefill engines
- [[dualpath]]: dual-path KV-cache loading in PD-disaggregated systems — adds storage-to-decode path + RDMA to prefill; 1.87× offline throughput, 1.96× online without SLO violation

### Sparse Query Computation (GQE)
Rather than compressing stored KV tensors, [[gqe]] reduces the Q-side compute that reads the KV cache at long context:
- MoE routing within GQA groups selects top-k query-head experts per token; KV heads stay dense
- 1.7–1.8× prefill speedup at ≥32k context with no KV cache profile change

## Quantization vs. Eviction Trade-offs

| Property | Eviction (H₂O/LazyEviction) | Quantization (TurboQuant/kvtc) |
|---|---|---|
| Memory reduction | High (retain 30–50% tokens) | 5×–20× compression of all tokens |
| Lossy? | Yes, irreversible | Yes, but retains all tokens |
| Needle-in-haystack | Risky (H₂O); better with MRI tracking | Safe |
| Reasoning tasks | H₂O fails; LazyEviction designed for it | Untested focus area |
| Serving/storage | Not applicable | kvtc's primary use case (cache lifetime extension) |
| Compute overhead | Low | Low-moderate (transform cost; one-time calibration) |

These are **complementary** — quantization + eviction can be combined (e.g., TriAttention eviction + SpectralQuant on retained tokens).

## See Also

- [[h2o]] — eviction approach (post-RoPE); fails on reasoning
- [[triattention]] — pre-RoPE eviction via trigonometric series; better for long-context
- [[lazyeviction]] — MRI-tracking eviction for reasoning tasks (GSM8K, MATH500); 50–70% KV reduction
- [[polarquant]] — polar quantization
- [[turboquant]] — vector quantization
- [[spectralquant]] — calibrated spectral quantization; best quality/compression of quantization methods
- [[kvtc]] — transform coding (PCA+DP+DEFLATE); 20× lossless, 40× with modest drop; storage/serving focus
- [[kv-cache-compression-comparison]] — detailed comparison
- [[quantization]] — broader quantization context
- [[qkv-projection-sharing]] — architectural reduction via K=V projection constraint; 50% cache, orthogonal to GQA/MQA
- [[speculative-decoding]] — orthogonal inference speedup technique
- [[dualpath]] — serving-layer KV loading bottleneck in agentic inference; 1.87× offline throughput
- [[gqe]] — MoE routing on GQA query heads; 1.7–1.8× prefill speedup at long context
- [[nemotron-3-ultra]] — architectural KV reduction via a Mamba-heavy hybrid backbone (only 2 KV heads across 108 layers) rather than eviction/quantization
- [[kimi-linear]] — channel-wise gated linear attention hybridized 3:1 with full attention; up to 75% KV cache reduction
- [[gemma-4]] — KV cache sharing (values reused as keys) + tuned local:global attention ratio; up to 37.5% global KV cache reduction
- [[z-token-compression]] — complementary axis: compresses the input sequence itself rather than the KV cache
