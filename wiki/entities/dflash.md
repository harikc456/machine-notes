---
title: DFlash (Block Diffusion for Flash Speculative Decoding)
created: 2026-05-31
updated: 2026-05-31
type: entity
tags: [inference, speculative, architecture]
sources: [raw/papers/2602.06036v2.pdf]
confidence: high
---

# DFlash

**DFlash: Block Diffusion for Flash Speculative Decoding**
*Jian Chen, Yesheng Liang, Zhijian Liu — UC San Diego, ICML 2026, arXiv:2602.06036, May 2026*

## Core Idea

All prior speculative decoding methods — including [[eagle-3]] — use **autoregressive drafting**: the draft model generates tokens sequentially, so drafting cost scales as T_draft = γ × t_step. This caps practical speedups at ~2–3× because adding more draft tokens linearly increases drafting cost while the marginal acceptance benefit diminishes.

DFlash replaces AR drafting with a **lightweight block diffusion model** that generates all γ draft tokens **in a single parallel forward pass**: T_draft = t_parallel (constant, independent of γ). The key insight is "**the target knows best**" — conditioning the diffusion drafter on the target model's hidden features makes it accurate without being large.

## Architecture

1. **Target model extraction**: during each decoding step, the target LLM's forward pass produces rich hidden-state features implicitly encoding future-token information
2. **Block diffusion adapter**: a small trainable adapter takes these features as context and denoises a block of masked tokens in parallel — generates γ draft tokens simultaneously
3. **Standard verification**: the target model verifies the draft block in one forward pass, exactly as in standard speculative decoding (lossless guarantee preserved)

The draft cost is dominated by t_parallel (one forward pass of the small adapter), not γ × t_step, fundamentally breaking the linear scaling wall.

## Drafting Cost Comparison

| Drafter type | Draft cost formula | Scales with γ? |
|---|---|---|
| Autoregressive (EAGLE-3) | γ × t_step | ✅ grows linearly |
| Block Diffusion (DFlash) | t_parallel | ❌ constant |

For moderate block sizes (γ = 4–16), t_parallel ≪ γ × t_step for comparable model sizes, giving DFlash a large drafting-latency advantage even before accounting for acceptance rates.

## Results

Evaluated on Qwen3-8B using SGLang, compared to EAGLE-3:

| Benchmark | EAGLE-3 | DFlash | Speedup over AR |
|---|---|---|---|
| GSM8K | 2.23× | 5.15× | — |
| Math500 | 2.05× | 6.08× | — |
| AIME25 | 2.05× | 5.62× | — |
| HumanEval | 2.17× | 5.14× | — |
| MBPP | 1.93× | 4.65× | — |
| LiveCodeBench | 1.81× | 5.51× | — |
| MT-Bench | 1.90× | 2.75× | — |

- **Over 6× lossless acceleration** on most benchmarks
- **Up to 2.5× higher speedup than EAGLE-3** across tasks
- Production-grade via SGLang integration

## Relationship to Existing DLM Work

DFlash uses [[block-diffusion]] (BD3-LM) as the drafting model architecture, but the purpose is entirely different: DFlash is a speculative decoding draft engine, not a standalone generative model. The block diffusion structure provides constant-cost parallel draft generation; the target AR model still performs final verification and determines output quality. This is a novel use of DLMs as accelerators for AR models rather than as replacements.

## Limitations

- Draft model must be trained per target model family (like EAGLE)
- Acceptance rates on conversational tasks (MT-Bench) are lower vs math/code — 2.75× vs 5×+
- Not yet evaluated at large batch sizes or with tensor parallelism at scale

## See Also

- [[speculative-decoding]] — the foundational algorithm DFlash accelerates
- [[eagle-3]] — the AR-drafting baseline DFlash surpasses; 1.38× throughput in SGLang at batch 64
- [[eagle]] — original EAGLE; feature-level AR drafting
- [[block-diffusion]] — the BD3-LM architecture used as DFlash's draft engine
- [[diffusion-language-models]] — broader context for diffusion as LM accelerators
- [[saguaro]] — orthogonal hardware-parallelism approach to speculative decoding
- [[continuous-batching]] — SGLang serving framework used in evaluations
