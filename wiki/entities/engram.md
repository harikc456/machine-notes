---
title: Engram
created: 2026-05-14
updated: 2026-05-14
type: entity
tags: [architecture, sparsity, inference, model, deepseek]
sources: [raw/papers/2601.07372v1.pdf]
confidence: high
---

# Engram

**Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models**
*Xin Cheng et al., Peking University + DeepSeek-AI, arXiv:2601.07372, Jan 2026*

## Overview

Engram is a **conditional memory module** that modernizes classic N-gram embedding lookup for O(1) access. It introduces a new axis of sparsity complementary to [[mixture-of-experts]] (MoE):

| Sparsity Type | Mechanism | What it handles |
|---|---|---|
| Conditional computation (MoE) | Routes tokens to experts | Dynamic, compositional reasoning |
| Conditional memory (Engram) | N-gram lookup in embedding table | Static, stereotyped knowledge retrieval |

The core hypothesis: standard Transformers waste early-layer computation "reconstructing" common N-gram patterns (named entities, formulaic phrases) that could simply be *looked up*.

## Technical Design

- **N-gram embedding table**: local context (N-gram) indexes a large static embedding table
- **Contextualized gating**: gates the lookup result before injecting into the residual stream
- **Multi-head hashing**: multiple hash functions for robust coverage
- **Multi-branch integration**: integrates lookup output at multiple points in the transformer

The **Sparsity Allocation problem**: given a fixed parameter budget, how to split between MoE experts and Engram memory? The experiments reveal a **U-shaped scaling law** — there is an optimal allocation, and both extremes (all-MoE or all-memory) are suboptimal.

## Scale

- Engram-27B achieves superior performance over iso-parameter, iso-FLOPs MoE baseline
- Memory module can be offloaded to host memory with <3% overhead (deterministic addressing enables prefetching)
- Scales to 100B-parameter lookup table stored in host memory

## Benchmark Results (Engram-27B vs. MoE baseline)

| Domain | Gain |
|---|---|
| MMLU | +3.4 |
| CMMLU | +4.0 |
| MMLU-Pro | +1.8 |
| BBH (reasoning) | +5.0 |
| ARC-Challenge | +3.7 |
| HumanEval (code) | +3.0 |
| MATH | +2.4 |
| Multi-Query NIAH (long context) | 84.2 → 97.0 |

Notably, gains are largest in **reasoning and code**, not just knowledge recall — suggesting the freed-up early layers contribute meaningfully to general capability.

## Mechanistic Interpretation

LogitLens and CKA analyses show:
- Engram relieves early transformer layers from static reconstruction tasks
- Increases effective network depth available for complex reasoning
- Frees attention capacity for global context (hence exceptional long-context gains)

## Relationship to Other Work

- Complements [[mixture-of-experts]] rather than replacing it
- Connects to [[conditional-memory]] as a general concept
- Engram's host-memory offload relates to [[kv-cache]] offloading challenges in inference
- Not yet integrated into [[deepseek-v4]] (related line of research)
