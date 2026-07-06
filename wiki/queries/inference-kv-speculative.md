---
title: KV Cache Compression and Speculative Decoding — Overview
created: 2026-05-19
updated: 2026-07-06
type: query
tags: [inference, kv-cache, quantization, speculative, attention, survey]
sources: []
confidence: high
---

# KV Cache Compression and Speculative Decoding — Overview

Companion to [[inference-improvements-summary]] for §3 (KV cache pruning and quantization) and §4
(speculative decoding). For memory-impact framing of these techniques, see
[[memory-inference-techniques]]. For research gaps and untested compositions, see
[[memory-inference-research-gaps]].

This page previously held the full detail for both topics; it was split on 2026-07-06 (exceeded
the ~200-line page-size threshold in SCHEMA.md) into two dedicated detail pages:

- **[[kv-cache-compression-detail]]** — H₂O, TriAttention (pre-RoPE eviction), PolarQuant,
  TurboQuant, SpectralQuant (KV quantization), and the SageAttention family (attention compute
  quantization)
- **[[speculative-decoding-detail]]** — standard SD algorithm and rejection sampling proof, the
  EAGLE family (EAGLE / EAGLE-2 / EAGLE-3), DFlash, DSpark, self-speculative decoding
  (LayerSkip / SWIFT / DASH), and Saguaro (SSD)

## See Also

- [[kv-cache-compression-detail]] — full KV cache pruning and quantization detail
- [[speculative-decoding-detail]] — full speculative decoding detail
- [[inference-improvements-summary]] — full inference survey overview (architecture, serving, DLMs, cross-cutting table)
- [[memory-inference-techniques]] — memory-focused inference survey with quantitative memory impact per technique
- [[memory-inference-research-gaps]] — methodological gaps, Pareto frontier analysis, untested compositions
- [[kv-cache]] — KV cache mechanics and bottleneck analysis
- [[speculative-decoding]] — speculative decoding concept page
