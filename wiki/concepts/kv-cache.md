---
title: KV Cache
created: 2026-05-14
updated: 2026-06-17
type: concept
tags: [kv-cache, inference, attention, quantization]
sources: [raw/papers/2306.14048v3.pdf, raw/papers/2502.02617v1.pdf, raw/papers/2504.19874v1.pdf, raw/papers/2604.04921v1.pdf]
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
- **Limitation**: irreversible eviction can miss critical tokens in retrieval tasks — though scoring quality matters; TriAttention's trigonometric approach is more stable than post-RoPE methods

### Quantization (precision reduction)
Reduce the bit-width of stored K and V tensors:
- [[polarquant]]: polar coordinate transform eliminates normalization overhead; >4.2× compression
- [[turboquant]]: random rotation + MSE quantizer + 1-bit QJL residual; neutral at 3.5 bits
- Traditional methods: per-block normalization adds >1 bit overhead
- QJL: 1-bit sketching, data-oblivious

### Architectural Reduction
Modify the model to produce fewer K/V pairs:
- **Multi-Query Attention (MQA)**: single shared K/V head across all query heads
- **Grouped-Query Attention (GQA)**: groups of query heads share one K/V head
- **Multi-Head Latent Attention (MLA)**: compress K/V into low-rank latent space (DeepSeek)
- **CSA/HCA** in [[deepseek-v4]]: 3.7–9.8× reduction in KV cache vs DeepSeek-V3.2
- **Projection sharing (Q-K=V)**: force K=V at the projection level — only K needs to be cached, V is reused. [[qkv-projection-sharing]] (ICML 2026): 50% cache, +3.1% PPL at 300M. Orthogonal to head sharing — combined Q-MQA achieves 96.9% cache reduction at +4.8% PPL.

### Offloading
Move KV cache from GPU HBM to CPU RAM or disk:
- [[engram]] prefetching demonstrates <3% overhead for 100B parameter lookup table
- [[deepseek-v4]] supports on-disk KV cache storage

## Quantization vs. Eviction Trade-offs

| Property | Eviction (H₂O) | Quantization (PolarQuant/TurboQuant) |
|---|---|---|
| Memory reduction | High (retain ~5-20%) | Moderate (2-4× compression) |
| Lossy? | Yes, irreversible | Yes, but retains all tokens |
| Needle-in-haystack | Risky | Safe |
| Compute overhead | Low | Low-moderate (transform cost) |

These are **complementary** — quantization + eviction can be combined.

## See Also

- [[h2o]] — eviction approach (post-RoPE)
- [[triattention]] — pre-RoPE eviction via trigonometric series; better for reasoning/long-context
- [[polarquant]] — polar quantization
- [[turboquant]] — vector quantization
- [[kv-cache-compression-comparison]] — detailed comparison
- [[quantization]] — broader quantization context
- [[qkv-projection-sharing]] — architectural reduction via K=V projection constraint; 50% cache, orthogonal to GQA/MQA
- [[speculative-decoding]] — orthogonal inference speedup technique
