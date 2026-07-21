---
title: Kimi Linear
created: 2026-07-21
updated: 2026-07-21
type: entity
tags: [model, architecture, attention, inference, kv-cache, open-source]
sources: [raw/papers/2510.26692v2.md]
confidence: high
---

# Kimi Linear

**Kimi Linear: An Expressive, Efficient Attention Architecture**
*Kimi Team, Moonshot AI, Nov 2025*

## Overview

A hybrid linear-attention architecture (3B activated / 48B total params) that, for the first time, outperforms full attention under matched-recipe comparisons across short-context, long-context, and RL scaling regimes — while cutting KV cache usage by up to 75% and achieving up to 6× decoding throughput at 1M context.

## Key Technical Contributions

- **Kimi Delta Attention (KDA)**: extends Gated DeltaNet with a **channel-wise (fine-grained) gate** rather than a scalar forget gate, giving each feature dimension its own forgetting rate. Formally a specialized Diagonal-Plus-Low-Rank (DPLR) transition matrix that binds its two free parameters together, roughly doubling operator efficiency over general DPLR.
- **Hardware-efficient chunkwise algorithm** (WY representation + UT transform) for parallel training/inference.
- **3:1 hybrid ratio**: 3 KDA layers per 1 full Multi-Head Latent Attention (MLA) layer — found optimal via ablation over 0:1 (pure full attention) through 15:1.
- **NoPE on MLA layers**: KDA supplies all positional/recency signal, letting MLA run as plain (RoPE-free) attention — simplifies long-context training and lets MLA collapse to efficient Multi-Query Attention at inference.

## Benchmark Results

- 6.3× faster time-per-output-token than MLA at 1M-token decoding length.
- Pareto-optimal on MMLU-Pro (4k context) and RULER (128k context) at 1.4T training tokens vs. MLA and GDN-H.
- On synthetic state-tracking/recall tasks (palindrome, MQAR, stack), KDA converges faster than GDN and succeeds where pure-decay Mamba2 fails outright.

## Relationships to Other Entities

- Sibling to [[deepseek-v4]] (CSA/HCA hybrid attention) and [[nemotron-3-ultra]] (Mamba-Attention hybrid) as a 2025/2026 hybrid-attention architecture targeting the same long-context KV-cache bottleneck.
- Extends the Gated DeltaNet / delta-rule lineage that [[qkv-projection-sharing]] and other KV-reduction techniques sit alongside in the [[kv-cache]] landscape.

## Open Questions

- How does KDA's channel-wise gating interact with [[speculative-decoding]] draft/verify schemes at 1M context?
- Does the 3:1 hybrid ratio hold at significantly larger scale (100B+ total params)?

## See Also

- [[kv-cache]] — KDA's 75% KV cache reduction is an architectural-reduction approach
- [[deepseek-v4]] — comparable hybrid attention (CSA/HCA) for 1M-token context
- [[nemotron-3-ultra]] — comparable Mamba-Attention hybrid at larger scale
- [[qkv-projection-sharing]] — another KV-reduction technique orthogonal to hybridization
