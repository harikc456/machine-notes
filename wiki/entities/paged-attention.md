---
title: PagedAttention (vLLM)
created: 2026-05-14
updated: 2026-05-14
type: entity
tags: [inference, kv-cache, attention, architecture]
sources: [raw/papers/2309.06180v1.pdf]
confidence: high
---

# PagedAttention (vLLM)

**Efficient Memory Management for Large Language Model Serving with PagedAttention**
*Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Yu, Joseph Gonzalez, Hao Zhang, Ion Stoica — UC Berkeley, arXiv:2309.06180, 2023*

## The KV Cache Fragmentation Problem

KV caches for in-flight requests are allocated as contiguous blocks of GPU memory. This causes three sources of waste:

1. **Internal fragmentation**: each request pre-allocates memory up to `max_seq_len`, but most sequences end early — the tail is wasted
2. **External fragmentation**: freed slots between live sequences are too small to reuse for new requests
3. **No sharing**: requests with common prefixes (system prompts, few-shot examples) each store their own copy

In practice, 60–80% of reserved KV cache memory goes unused due to fragmentation.

## The Paging Solution

PagedAttention divides KV cache into fixed-size **pages** (physical blocks), mapping to sequence positions via a **block table** — exactly like OS virtual memory:

```
Logical sequence:  [tok0, tok1, tok2, ... tok_n]
                         ↓ block table
Physical blocks:   [block_7][block_2][block_15]...  (non-contiguous OK)
```

- Block size is fixed (e.g., 16 tokens per block)
- Blocks are allocated on demand as the sequence grows
- Freed blocks are immediately available to other sequences
- The attention kernel is rewritten to follow block table indirections

## Copy-on-Write for Parallel Sampling

When generating multiple outputs from the same prompt (beam search, sampling N completions):
- All candidate sequences **share** the prompt's physical blocks (reference counted)
- On divergence, the diverging sequence gets its own copy only at the block being modified
- Result: prompt KV cache is stored once, not N times

## Memory Efficiency

PagedAttention achieves near-zero fragmentation (~4% waste vs. 60–80% in prior systems). This translates to:
- **2–4× higher throughput** vs. HuggingFace TGI at the same hardware
- **Up to 4× higher effective throughput** vs. FasterTransformer
- Enables much larger batch sizes, which is the primary lever for throughput

## Prefix Caching (Radix Attention)

An extension: when multiple requests share a common prefix (e.g., the same system prompt), their prefix blocks are cached and reused across requests — not just within a request. [[radix-attention]] (SGLang) extends this with a **radix tree** that matches the longest common prefix across all requests in the cache, achieving a further 2–4× throughput improvement over vLLM on shared-prefix workloads.

## Relationship to Other Work

- [[flash-attention]] handles how attention is computed over KV blocks; PagedAttention handles where blocks live — they compose without modification
- [[continuous-batching]] relies on PagedAttention to allocate and free KV blocks per token rather than per request
- [[kv-cache]] — background on why the cache is the bottleneck

## See Also

- [[flash-attention]] — complementary attention kernel optimization
- [[kv-cache]] — KV cache fundamentals
- [[continuous-batching]] — serving scheduler that PagedAttention enables
- [[radix-attention]] — cross-request prefix caching built on top of PagedAttention
- [[h2o]] — KV cache eviction (orthogonal: PagedAttention manages memory layout, H₂O evicts tokens)
