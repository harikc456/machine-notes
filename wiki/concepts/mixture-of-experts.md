---
title: Mixture of Experts (MoE)
created: 2026-05-14
updated: 2026-06-24
type: concept
tags: [sparsity, architecture, model, training]
sources: [raw/papers/2601.07372v1.pdf, raw/papers/DeepSeek_V4.pdf, raw/papers/2512.02556v1.pdf, raw/papers/2606.20945v2.md]
confidence: high
---

# Mixture of Experts (MoE)

## Overview

Mixture of Experts is a **conditional computation** architecture that scales model capacity without proportionally increasing FLOPs. Instead of all parameters participating in every forward pass, a routing mechanism selects a subset of "expert" sub-networks per token.

```
output = Σ_{i ∈ top-k} gate(x)_i · Expert_i(x)
```

A typical MoE transformer replaces dense FFN layers with many experts (e.g., 64 FFN networks) and activates only 2-4 per token.

## Key Properties

- **Capacity vs. FLOPs decoupled**: A 1T-parameter MoE can have the same per-token FLOPs as a 50B dense model
- **Conditional computation**: Different tokens activate different experts → natural specialization
- **Routing**: Usually learned (top-k gating) — load balancing is a training challenge

## Why MoE Became Dominant

MoE is now "de facto standard for frontier models" (per [[engram]] paper). Used in:
- [[deepseek-v4]]: 1.6T total, 49B active (V4-Pro)
- [[deepseek-v3-2]]: MoE architecture with DSA
- GPT-4, Gemini, Mistral Mixture-of-Experts, etc.

## Limitations That Conditional Memory Addresses

MoE handles **dynamic, compositional reasoning** well but has a mismatch for **static knowledge retrieval**. Common entities and formulaic patterns are highly stereotyped — they benefit more from lookup than from routed computation. The [[conditional-memory]] concept (instantiated in [[engram]]) is proposed as the complementary sparsity axis.

## The Sparsity Allocation Problem

Given a fixed parameter budget: how much should go to MoE experts vs. static memory (Engram)?

- [[engram]] paper finds a **U-shaped curve**: optimal mix outperforms either extreme
- Even a small fraction of parameters in conditional memory yields significant gains

## Infrastructure Challenges

- **Load balancing**: Routing must distribute tokens across experts; auxiliary losses often required
- **Communication**: Expert parallelism requires all-to-all communication across GPUs
- [[deepseek-v4]] addresses this with fine-grained expert parallelism + communication-computation overlap

## MoE Beyond FFN Layers: Attention Query Experts

The MoE idea is not limited to FFN layers. [[gqe]] (Grouped Query Experts, Jun 2026) applies it to the query-head computation within grouped-query attention (GQA):
- Each GQA group's query heads become experts; a router selects top-k per token
- KV heads remain dense (GQA memory savings preserved)
- Matches all-active GQA quality at 250M params, 30B tokens; 1.7–1.8× prefill speedup at long context
- A complementary sparsity axis: MoE in FFN reduces parameter-compute cost; GQE reduces attention-compute cost at long context

## See Also

- [[conditional-memory]] — complementary sparsity axis (static lookup vs. routed compute)
- [[engram]] — conditional memory instantiation
- [[deepseek-v4]] — trillion-parameter MoE
- [[deepseek-v3-2]] — MoE with scalable RL
- [[hyper-connections]] — architectural improvement to residual connections within MoE layers
- [[gqe]] — MoE routing applied to GQA query heads (attention sparsity)
