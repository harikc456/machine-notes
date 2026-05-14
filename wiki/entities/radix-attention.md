---
title: RadixAttention (SGLang)
created: 2026-05-14
updated: 2026-05-14
type: entity
tags: [inference, kv-cache, optimization, attention]
sources: [raw/papers/2312.07104v1.pdf]
confidence: high
---

# RadixAttention (SGLang)

**Efficient Memory Management for Large Language Model Serving with RadixAttention**
*Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Jeff Huang, Chunyi Li, Yao Lu, Clark Barrett, Hao Zhang, Joseph E. Gonzalez, Ion Stoica — UC Berkeley / Stanford / CMU, arXiv:2312.07104, 2023*

## Overview

SGLang is a serving system for LLMs and vision-language models that introduces **RadixAttention** — automatic KV cache reuse across requests sharing common prefixes via a radix tree. Achieves 2–4× throughput improvement over vLLM by eliminating redundant prefill computation for shared prefixes.

## The Problem: Redundant Prefix Computation

Many real-world LLM workloads share large common prefixes across requests:
- **System prompts**: every request in a deployment starts with the same system prompt (hundreds to thousands of tokens)
- **Few-shot examples**: shared in-context demonstrations
- **Multi-turn conversations**: earlier turns are a prefix of later turns
- **Tree-of-thought / multi-step reasoning**: branching continuations from a common stem
- **Speculative decoding**: multiple candidate continuations from the same prefix

[[paged-attention]] solves *within-request* fragmentation but each new request re-computes its prefix from scratch. RadixAttention eliminates this cross-request redundancy.

## Radix Tree Data Structure

A radix tree (compressed trie) maps token sequences → cached KV blocks:

```
Root
├── [system_prompt_tokens] → KV block A (refcount=N)
│   ├── [user_A_tokens] → KV block B
│   │   └── [assistant_A_tokens] → KV block C
│   └── [user_B_tokens] → KV block D
└── [other_prefix] → KV block E
```

- Each edge represents a token subsequence
- Each node stores the KV cache for that prefix
- Nodes are **reference-counted** — shared by all active requests using that prefix
- Tree branches when requests diverge; branches are collapsed when they're the only child

## Cache Lookup and Sharing

On each new request:
1. Walk the radix tree to find the **longest matching prefix**
2. Load cached KV blocks for that prefix — no prefill computation needed
3. Only compute prefill for the **suffix** (tokens after the matching prefix)
4. Extend the tree with new nodes for the computed suffix

## LRU Eviction

When GPU memory is full, evict the **least recently used** leaf nodes first (deepest nodes with no active users). This preserves frequently accessed prefixes (system prompts, popular few-shot examples) while reclaiming memory from stale branches.

Copy-on-write semantics: shared nodes are never evicted while any request references them (refcount > 0).

## Performance

- **2–4× throughput improvement over vLLM** on workloads with shared prefixes
- Greatest gains on: chatbot deployments (shared system prompts), batch inference with shared few-shot examples, agentic pipelines with repeated tool descriptions
- Minimal overhead on workloads with no sharing (tree lookup is O(prefix_length))

## SGLang: Structured Generation Language

Beyond RadixAttention, SGLang provides a Python-embedded DSL for structured LLM programs:

```python
@sgl.function
def multi_turn_qa(s, question):
    s += sgl.system("You are a helpful assistant")
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answer"))
```

The runtime automatically identifies prefix sharing opportunities across concurrent `sgl.function` calls and routes them through RadixAttention. The programmer doesn't need to manually manage the cache.

## Relationship to PagedAttention

RadixAttention and [[paged-attention]] are **complementary, not competing**:

| | PagedAttention | RadixAttention |
|---|---|---|
| Problem solved | Within-request KV fragmentation | Cross-request KV redundancy |
| Mechanism | Virtual memory paging | Radix tree prefix cache |
| Scope | Single request lifecycle | Across all requests |
| Eviction | Per-sequence block free | LRU across the whole cache |

SGLang uses PagedAttention for block management and RadixAttention for prefix sharing — both active simultaneously.

## See Also

- [[paged-attention]] — complementary within-request memory management
- [[continuous-batching]] — serving scheduler; RadixAttention integrates with it via prefix-aware scheduling
- [[kv-cache]] — KV cache fundamentals
- [[kv-cache-compression-comparison]] — H₂O/PolarQuant/TurboQuant: orthogonal approaches (eviction/quantization)
