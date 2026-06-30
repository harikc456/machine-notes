---
title: LazyEviction
created: 2026-06-29
updated: 2026-06-29
type: entity
tags: [kv-cache, inference, attention]
sources: [raw/papers/2506.15969v3.md]
confidence: high
---

# LazyEviction

**LazyEviction: Lagged KV Eviction with Attention Pattern Observation for Efficient Long Reasoning**
*Haoyue Zhang, Hualei Zhang, Xiaosong Ma, Jie Zhang, Song Guo — HKUST / HK PolyU, arXiv:2506.15969, Oct 2025*

## Problem

Existing KV cache eviction strategies (H₂O, TOVA, Scissorhands) use greedy per-step eviction based on current or accumulated attention scores. They perform well on standard long-context tasks but fail on **long reasoning tasks** (Chain-of-Thought, MATH500, GSM8K). The root cause: they discard tokens during temporarily-low-attention intervals, not knowing the tokens will regain importance later.

## Key Insight: Token Importance Recurrence (TIR)

Empirical analysis of DeepSeek-R1-Distill models on GSM8K and MATH500 shows that **>95% of tokens exhibit TIR** — their attention score cycles between high and low rather than decaying monotonically. These recurring tokens typically carry:
- Initial problem conditions (re-referenced throughout multi-step reasoning)
- Intermediate conclusions (reactivated when later reasoning steps need them)

The Maximum Recurrence Interval (MRI) of most tokens is short relative to total output length: 80th percentile MRI < 175 tokens for Qwen on MATH500, even at 8k output lengths.

## Method

Two components running during autoregressive decoding:

### Recurrence Interval Tracking
Per retained token i:
- Track timestamp TS_i = last decode step where attention exceeded threshold α
- Track MRI_i = max(MRI_{t-1}, TS_t − TS_{t-1}) — longest observed gap between activations

### MRI-Centric Eviction (at intervals of W steps)
When KV cache size exceeds budget B (B ≫ W), evict B−W tokens using a combined score:
- **H1-score**: `2σ(−(t − TS[i]) / MRI[i])` — probability token is still within recurrence interval
- **H2-score**: `2σ(−1 / MRI[i])` — tokens with smaller MRI (more frequent recurrences) ranked higher

Always retain the W most recent KVs unconditionally (preserve local coherence). Tokens where `(t − TS[i]) > MRI[i]` are predicted safe to evict.

The key shift: from step-wise greedy eviction to **window-wise predictive retention**.

## Results

| Method | GSM8K (50% budget) | MATH500 |
|---|---|---|
| Full KV cache | baseline | baseline |
| H₂O | −20% relative | degrades |
| TOVA | −20% relative | degrades |
| **LazyEviction** | ~baseline | ~baseline |

- 50–70% KV cache reduction while maintaining comparable reasoning accuracy
- Verified on 4B–32B models (DeepSeek-R1-Distill-Llama-4B, 8B, 32B)
- Compatible with CPU offload and quantization (orthogonal techniques)

## Contrast with Prior Work

[[h2o]] uses accumulated attention — underestimates future importance of temporarily-quiet recurring tokens. TOVA uses current attention — same problem but worse (no history). LazyEviction's MRI explicitly models the temporal structure that makes tokens recurrently important.

[[triattention]] addresses a different axis: pre-RoPE eviction scoring to avoid rotation instability. LazyEviction's contribution is the observation window + recurrence tracking for the reasoning setting.

## See Also

- [[kv-cache]] — eviction approaches and broader compression landscape
- [[h2o]] — cumulative attention eviction; fails on reasoning tasks
- [[triattention]] — pre-RoPE eviction; strong on long-context, different mechanism
- [[kv-cache-compression-comparison]] — side-by-side comparison table
