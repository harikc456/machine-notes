---
title: Gemma 4
created: 2026-07-21
updated: 2026-07-21
type: entity
tags: [model, architecture, inference, kv-cache, quantization, speculative, open-source]
sources: [raw/papers/2607.02770v1.md]
confidence: high
---

# Gemma 4

**Gemma 4 Technical Report**
*Gemma Team, Google DeepMind, Jul 2026*

## Overview

Open-weight, natively multimodal LLM family (2.3B–31B params) spanning dense and MoE architectures, with a unified encoder-free 12B variant that ingests raw audio/image patches directly. Gemma 4 31B is the top-ranked open **dense** model on Arena (Elo 1451).

| Model | Type | Params | Notes |
|---|---|---|---|
| E2B / E4B | Dense | 2.3B / 4.5B effective | vision (150M) + audio (305M) encoders |
| 12B | Dense | 12B | encoder-free — raw patches |
| 26B-A4B | MoE | 26B total / 3.8B active | |
| 31B | Dense | 31B | |

## Key Technical Contributions

- **Thinking mode**: reasoning trace generated before the final response, improving math/coding/reasoning benchmarks.
- **KV cache efficiency**: 5:1 (4:1 for 2.3B) local-sliding-window : global-full-attention ratio, `p`-RoPE (p=0.25) on global layers, and **KV cache sharing** (values reused as keys in global layers, except E2B/E4B) — together cut global KV cache footprint by up to 37.5%.
- **Encoder-free architecture** (12B only): drops the separate 550M vision + 305M audio encoders in favor of lightweight projections of raw image patches and raw 40ms audio chunks straight into the LLM embedding space.
- **MTP speculative-decoding drafter**: small 4-layer autoregressive head cross-attending to the main model's KV; for E2B/E4B its output projection is reduced to a top-k 4096-token cluster (from the full 262k vocabulary) to cut decode overhead.
- **Quantization-Aware Training (QAT)**: released for LLM weights (mobile int2/int4 + int8 activations, and Q4_0 blockwise), the vision encoder (8-bit, 2× memory reduction), and the audio encoder (mixed 2/4/8-bit, 78% smaller on-disk footprint with improved WER/BLEU vs. Gemma 3n).

## Benchmark Results

- Arena: 31B ranks #43 overall (Elo 1451, top open dense model); 26B-A4B ranks #61 (Elo 1438).
- 31B vs. Gemma 3 27B: MMLU-Pro 85.2 vs. 67.6, AIME 2026 89.2 vs. 20.8, GPQA Diamond 84.3 vs. 42.4.
- Long context: RULER 128k accuracy jumps from 66.0 (Gemma 3 27B) to 89.8–96.4; GraphWalks F1 from 32.8 to 50.9–82.3.
- Audio: E2B/E4B beat same-size Gemma 3n despite dropping the dedicated audio encoder, with a 78% smaller on-disk audio footprint.

## Relationships to Other Entities

- Comparable frontier-efficiency release to [[deepseek-v4]] and [[nemotron-3-ultra]], though oriented toward edge/on-device deployment and multimodality rather than extreme (1M-token) context.
- MTP drafter parallels [[nemotron-3-ultra]]'s shared-weight MTP and the broader [[speculative-decoding]] literature (EAGLE family, Medusa).

## See Also

- [[kv-cache]] — KV cache sharing + local:global ratio tuning
- [[mixture-of-experts]] — 26B-A4B MoE variant
- [[speculative-decoding]] — MTP drafter design
- [[quantization]] — QAT across LLM, vision, and audio encoders
- [[deepseek-v4]] — comparable frontier open release
- [[nemotron-3-ultra]] — comparable frontier open release with shared-weight MTP
