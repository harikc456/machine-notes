---
title: Continuous Batching
created: 2026-05-14
updated: 2026-05-14
type: concept
tags: [inference, optimization, training]
sources: [raw/papers/2308.16369v2.pdf]
confidence: high
---

# Continuous Batching

## Overview

Traditional LLM serving systems process requests in static batches: all requests in a batch must finish before new ones are accepted. This wastes GPU cycles whenever any request finishes early. **Continuous batching** (also called iteration-level scheduling) replaces finished requests with new ones at the token generation level, keeping the GPU continuously saturated.

First described in ORCA (Yu et al., 2022, arXiv:2207.04836). Now standard in vLLM, TensorRT-LLM, and TGI.

## The Static Batching Problem

In static batching, a batch of B requests runs until all B finish. If one request generates 10 tokens and another generates 500, the short request's allocated GPU memory and batch slot sit idle for 490 steps.

**Padding waste**: batching variable-length prompts requires padding all to the same length. For batch size B=8 with one new prompt of length n=100, this introduces up to `(n-1)(B-1) = 693` wasted padding tokens per step.

## Continuous Batching Algorithm

```
While serving:
  1. Collect all DECODE-phase requests (generate 1 token each)
  2. Fill remaining memory budget with PREFILL-phase requests
     (using chunked prefill to split long prompts)
  3. Run one forward pass — decode and prefill chunks interleaved
  4. Remove finished sequences (hit <eos> or max length)
  5. Admit waiting requests into freed slots
```

The batch composition changes every single forward pass. No padding: use ragged batching (concatenated sequences with attention masks preventing cross-request attention).

## Ragged Batching

Removes the batch dimension entirely. Sequences are concatenated:

```
Input tensor: [seq0_tok0, seq0_tok1, ... seq1_tok0, seq1_tok1, ...]
```

Attention masks ensure tokens only attend within their own sequence. This eliminates all padding, regardless of prompt length variation.

## Chunked Prefill (SARATHI)

**SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills**
*Amey Agrawal et al., arXiv:2308.16369, 2023*

**The prefill stall problem**: prefill is compute-intensive and occupies an entire engine step. During that step, no decode requests make progress — long prompts can stall other in-flight requests for hundreds of milliseconds (a "prefill monopolization bubble").

**Chunked prefill**: split long prompt prefills into equal-sized chunks (e.g., 512 tokens). Each chunk occupies one engine step alongside decode requests:

```
Step N:   [prefill_chunk_1_of_3] + [decode_req_A] + [decode_req_B] + ...
Step N+1: [prefill_chunk_2_of_3] + [decode_req_A] + [decode_req_B] + ...
Step N+2: [prefill_chunk_3_of_3] + [decode_req_A] + [decode_req_B] + ...
```

Decode requests "piggyback" on compute-intensive prefill work at near-zero marginal cost (decode adds memory bandwidth, not compute).

**Results (SARATHI paper)**:
- LLaMA-13B on A6000: 10× decode throughput improvement, 1.33× end-to-end
- LLaMA-33B on A100: 4.25× decode throughput, 1.25× end-to-end
- GPT-3 with pipeline parallelism: 6.29× pipeline bubble reduction, 1.91× end-to-end

## Interaction with PagedAttention

Continuous batching requires per-token memory management — blocks must be allocated on decode and freed immediately on completion. [[paged-attention]] is what makes this feasible by managing KV blocks at the granularity of individual tokens rather than whole requests.

## Prefix Caching

When many requests share a common prefix (e.g., system prompt), the prefix's KV cache is computed once and shared. Combined with continuous batching, this eliminates redundant prefill work across requests. Implemented as radix attention in SGLang.

## Throughput vs. Latency Trade-off

Continuous batching maximizes **throughput** (tokens/second across all users) at the cost of slightly higher **per-request latency** (time-to-first-token) for decode requests when a large prefill chunk is in the batch. Chunked prefill directly addresses this by bounding prefill chunk size and thus bounding the stall per step.

## See Also

- [[paged-attention]] — memory allocator that makes per-token scheduling feasible
- [[flash-attention]] — attention kernel; orthogonal (computes attention over whatever tokens are batched)
- [[kv-cache]] — KV cache fundamentals; continuous batching manages its lifecycle
- [[speculative-decoding]] — orthogonal algorithmic speedup; composes with continuous batching
