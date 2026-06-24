---
title: GQE (Grouped Query Experts)
created: 2026-06-24
updated: 2026-06-24
type: entity
tags: [attention, sparsity, architecture]
sources: [raw/papers/2606.20945v2.md]
confidence: medium
---

# GQE — Grouped Query Experts

**Grouped Query Experts: Mixture-of-Experts on GQA Self-Attention**  
*Vishesh Tripathi, Abhay Kumar — FrontiersMind, Jun 2026, arXiv:2606.20945*

## Overview

GQE applies the MoE conditional-routing idea to the query-head computation inside grouped-query attention (GQA) blocks. KV heads remain dense and fully computed — only query computation becomes sparse. At long context lengths, where Q×Kᵀ dominates, this yields direct FLOP savings.

## Setup

In standard GQA with G groups and M query heads per group (plus one shared KV head per group):
- Each group contains M query-head **experts** E_{g,1}, ..., E_{g,M}
- Each expert owns its own query projection and produces one attention-head output
- A per-group router scores all M experts: `p_{i,g} = softmax(r_g(x_i))`
- Token x_i activates top-k experts within each group
- One additional **always-on shared head** per group ensures routing stability and provides a learning signal even when routing collapses

Output projection W_O is resized from N×d to (kG + 2)×d (k experts selected per G groups, plus 2 always-on heads total across groups for the shared heads).

## Why Query-Side Sparsity?

In GQA, the KV cache is already reduced by head sharing. The remaining compute cost at long contexts is Q×Kᵀ, which scales as O(N² × n_query_heads). KV heads must be dense (reducing KV heads further would conflict with GQA semantics and change the cache profile). Query-side routing selectively evaluates only k of M query projections — GQA's memory benefit is preserved exactly.

## Key Properties

| Property | GQE | MoH (Mixture of Heads) | MoMHA/LLaMA-MoE v2 |
|---|---|---|---|
| Routing granularity | Within GQA group | Free across all heads | Route entire GQA groups on/off |
| KV cache | GQA profile (unchanged) | Full (per-head KV) | GQA (groups stay together) |
| Training | From scratch | From scratch | Convert dense model |

GQE is the most constrained: it can only route within a group's query heads, cannot change the KV profile, and trains from scratch jointly with the router.

## Results

Evaluated at 250M parameters, 30B training tokens (fixed compute budget vs standard all-active GQA baseline).

- **Downstream accuracy**: matches all-active GQA baseline
- **Active query heads**: 9 of 16 per token (8 groups × top-1 expert + 1 shared/group = 9 active)
- **Prefill speedup**: 1.7–1.8× at ≥32k context; grows monotonically with sequence length
- Speedup plateaus because only the query Matmul (Q×Kᵀ) is sparse; P×V and linear layer costs are unchanged

## Limitations / Confidence

- Evaluated at 250M parameters only — scaling behavior at 7B+ unknown
- 30B token budget is modest; longer training may shift the accuracy-compute tradeoff
- `confidence: medium` — single small-scale study from a small lab (FrontiersMind)

## Relationship to Other Work

- MoE concept: [[mixture-of-experts]] — standard MoE applies to FFN layers; GQE transfers the idea to attention query heads
- GQA: [[qkv-projection-sharing]] — explores K=V sharing; GQE keeps K≠V, instead sparsifies Q computation
- Sparsity in general: [[engram]] (static lookup sparsity), [[mixture-of-experts]] (FFN sparsity), GQE (query-head sparsity) form different axes

## See Also

- [[mixture-of-experts]] — MoE concept; GQE is MoE applied to attention query heads
- [[kv-cache]] — GQE preserves GQA's KV cache savings
- [[qkv-projection-sharing]] — different approach to reducing attention compute/cache
- [[deepseek-v4]] — CSA/HCA: extreme KV reduction at the architecture level
