---
title: Conditional Memory
created: 2026-05-14
updated: 2026-05-14
type: concept
tags: [sparsity, architecture, inference]
sources: [raw/papers/2601.07372v1.pdf]
confidence: high
---

# Conditional Memory

## Concept

Conditional memory is a **sparsity axis** for language models that is complementary to conditional computation ([[mixture-of-experts]]). The idea: language modeling involves two qualitatively different sub-tasks:

| Sub-task | Type | Best served by |
|---|---|---|
| Compositional reasoning | Dynamic, computation-heavy | MoE (conditional computation) |
| Knowledge retrieval | Static, stereotyped | Conditional memory (lookup) |

Current Transformers have only conditional computation (MoE). They *simulate* static knowledge retrieval by running attention and FFN over common N-grams — wasting compute on something that could be a table lookup.

## The Core Hypothesis

Named entities, formulaic phrases, common multi-token sequences — these are resolved by **lookup**, not computation. Early transformer layers waste depth on reconstructing these patterns from scratch at every forward pass. By offloading this to a fast static memory, those layers are freed for higher-level reasoning.

This was empirically validated in [[engram]]:
- LogitLens analysis shows early layers are relieved of static reconstruction
- CKA analysis shows increased effective depth for complex reasoning
- Long-context retrieval improves dramatically (NIAH: 84.2 → 97.0)

## Sparsity Allocation

The **Sparsity Allocation problem**: given a fixed parameter budget, how to partition between:
- Neural compute parameters (MoE experts)
- Static memory parameters (conditional memory table)

The answer is a **U-shaped scaling law**: both extremes (all-MoE, all-memory) are suboptimal. The optimal allocation is somewhere in between, and the curve favors having *some* conditional memory even at large scale.

## Instantiation: Engram

[[engram]] is the first concrete instantiation of conditional memory at scale:
- N-gram hash → lookup in large embedding table → O(1) access
- Scales to 27B-parameter table
- Can be offloaded to host memory (deterministic addressing = prefetchable)

## Relationship to Other Sparsity Work

- [[mixture-of-experts]]: the complementary sparsity axis
- [[h2o]]/[[kv-cache]]: a different kind of "memory" — caching past KV pairs (dynamic, per-inference)
- Conditional memory is *static* (weight-level), KV cache is *dynamic* (activation-level)

## Open Questions

- Does conditional memory help equally across all model sizes?
- What is the optimal memory architecture beyond N-grams (e.g., subword, phrase-level)?
- Can the lookup be made differentiable end-to-end during training?
