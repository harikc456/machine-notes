---
title: Serving Infrastructure (Algorithmic) — Detail
created: 2026-07-06
updated: 2026-07-06
type: query
tags: [inference, kv-cache, survey]
sources: []
confidence: high
---

# Serving Infrastructure (Algorithmic) — Detail

Split out of [[inference-improvements-summary]] §5 on 2026-07-06 (page-size threshold). Scheduling
and memory management techniques that improve throughput at the serving layer — no model changes
required.

### Flash Attention

[[flash-attention]] (Dao et al., 2022): tiles Q/K/V into SRAM blocks; computes attention with online softmax without materializing the full N×N matrix. IO complexity: O(Nd + N²/B) vs O(N²d). **7.6× speedup on GPT-2** (A100); O(N) memory — enables long contexts. Now default in PyTorch, HuggingFace, vLLM.

### PagedAttention

[[paged-attention]] (vLLM, 2023): fixed-size KV pages mapped via block tables (OS-style virtual memory). Eliminates 20–80% memory waste from fragmentation. **2–4× throughput over TGI** at same hardware. Enables prefix caching: system prompts computed once, shared across requests.

### RadixAttention

[[radix-attention]] (SGLang, 2023): radix tree maps token sequences → cached KV blocks. Longest matching prefix served on cache hit; LRU eviction clears unused leaves. **2–4× throughput improvement over vLLM** for shared-prefix workloads (system prompts, few-shot examples, agentic pipelines with repeated tool descriptions). Composes with PagedAttention: both active simultaneously.

### Continuous Batching + Chunked Prefill

[[continuous-batching]]: swap finished requests out at the token level (not request level); split long prompts into fixed-size chunks interleaved with decode steps. Results (SARATHI): **1.25–1.91× end-to-end**, **4–10× decode throughput**, 6.29× pipeline bubble reduction for GPT-3.

### DualPath (Agentic KV Loading)

[[dualpath]] (Peking U / DeepSeek-AI, Feb 2026): for agentic (multi-turn) inference in PD-disaggregated systems, KV-cache loading — not compute — dominates. Hit rates ≥95%, cache-compute ratio ~22 GB/PFLOP (DeepSeek-V3.2) saturate prefill-side storage NICs while decode NICs sit idle. DualPath adds a storage-to-decode path + RDMA to prefill, doubling effective storage bandwidth with no hardware changes. **1.87× offline throughput, 1.96× online** without SLO violation.

## See Also

- [[inference-improvements-summary]] — full inference survey overview this section was split from
- [[flash-attention]] — IO-aware tiled attention kernel
- [[paged-attention]] — OS-style KV cache memory management
- [[radix-attention]] — radix tree cross-request prefix caching (SGLang)
- [[continuous-batching]] — iteration-level scheduling + chunked prefill
- [[dualpath]] — dual-path KV loading for agentic inference; 1.87× offline throughput
- [[kv-cache]] — KV cache fundamentals these techniques manage
