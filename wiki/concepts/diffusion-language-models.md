---
title: Diffusion Language Models
created: 2026-05-16
updated: 2026-07-21
type: concept
tags: [architecture, training, inference]
sources: [raw/papers/2503.09573v3.pdf, raw/papers/2604.11035v1.pdf, raw/papers/2602.06036v2.pdf, raw/papers/llada-mercury-blog.md, raw/papers/optimal-architecture-slm-blog.md, raw/papers/2502.09992v3.md]
confidence: high
---

# Diffusion Language Models

## What They Are

Diffusion language models (DLMs) apply the diffusion framework to discrete token sequences: a forward process progressively corrupts clean text (e.g., by masking tokens), and the model learns to reverse this process — predicting clean tokens from noisy ones. Unlike autoregressive (AR) models that generate tokens left-to-right one at a time, DLMs can in principle generate an entire sequence in parallel via iterative denoising.

## Appeal vs. AR Models

| Property | Autoregressive | Diffusion LM |
|---|---|---|
| Generation quality | ✅ High | ❌ Historically lower |
| Arbitrary-length output | ✅ | ❌ Fixed-length (standard) |
| KV caching | ✅ | ❌ Bidirectional attention |
| Parallel token generation | ❌ | ✅ |
| Controllability | Harder | Easier (infilling, etc.) |

The quality gap has historically prevented DLM adoption.

## The Quality Gap and Its Causes

Two root causes have been identified:

**1. High training gradient variance** (BD3-LM, 2025): Even when diffusion and AR objectives are equivalent in expectation (block size = 1), diffusion training has much higher gradient variance. This alone causes DLMs to underperform AR — not architectural limitations. Data-driven noise schedules that minimize variance largely close the perplexity gap.

**2. Lack of introspective consistency** (I-DLM, 2026): AR training enforces that the model "agrees with its own generations" via causal masking and next-token prediction. DLMs trained with bidirectional attention and random masking learn to predict, but are not trained to endorse what they produce. The introspective acceptance rate α = (1/L) Σ min(1, p_k(x_k)/q_k(x_k)) measures this — standard DLMs score 0.57–0.70, AR models score 1.0.

## Key Systems

### [[llada]] (NeurIPS 2025)
The foundational **from-scratch, full-scale** DLM: 8B params, 2.3T tokens, standard pretrain+SFT paradigm, no AR conversion. Bidirectional-attention mask predictor trained on a provable NLL upper bound; low-confidence remasking at inference. Matches/exceeds AR baselines at matched compute on math and Chinese-language tasks, and beats GPT-4o on a poem-reversal task — directly addressing the reversal curse. Incompatible with KV caching (vanilla bidirectional multi-head attention, no GQA). The reference point against which [[i-dlm]] and other conversion-based DLMs benchmark (e.g., LLaDA-2.1-mini throughput/accuracy comparisons).

### [[block-diffusion]] (ICLR 2025)
Interpolates between AR and diffusion by defining an autoregressive distribution over *blocks* of tokens, with diffusion operating within each block. Recovers arbitrary-length generation and KV caching. Sets SOTA perplexity among discrete DLMs on LM1B.

### [[i-dlm]] (Apr 2026)
Converts pretrained AR models into DLMs using **introspective-consistency training**: causal attention + logit shift + all-masked objective. Introduces Introspective Strided Decoding (ISD) — single-pass generation + verification. First DLM to match strong same-scale AR quality; 3.1× higher throughput than prior SOTA DLMs at concurrency.

### [[dflash]] (May 2026, ICML 2026)
A novel use of diffusion models as **speculative decoding draft engines** for AR models — not as standalone generators. DFlash trains a small block diffusion adapter conditioned on the target AR model's hidden features; it generates all γ draft tokens in a single parallel forward pass. The AR target model performs verification (lossless). "The target knows best" — large AR models implicitly encode future-token information in their hidden states, making them ideal conditioning context for the diffusion drafter. Achieves 6×+ lossless speedup over AR decoding, 2.5× over [[eagle-3]]. Demonstrates DLMs as *accelerators* for AR models rather than replacements.

## Key Noise Processes

**Masking diffusion** (D3PM, MDLM, MD4): tokens are replaced by a [MASK] token; the model predicts the original token given surrounding context and noise level. Most effective and widely used in recent DLMs.

**Absorbing state**: similar to masking — tokens are absorbed into a special token and must be recovered. Simpler than full categorical diffusion.

## Common Serving Challenges

Standard DLMs break AR serving infrastructure:
- Bidirectional attention is incompatible with causal attention kernels
- Block diffusion and iterative denoising create synchronization points between tokens
- Continuous batching assumptions (uniform token-by-token advance) don't hold

I-DLM specifically addresses this by using causal attention throughout, enabling direct deployment on AR serving stacks (SGLang, paged KV cache).

## Open Questions

- Can the BD3-LM block structure and I-DLM introspective training be combined for further gains?
- What is the optimal block size for BD3-LMs across different task types?
- Does introspective consistency training transfer to models beyond the masked diffusion framework?
- How does the DLM accuracy gap evolve at 7B+ scale? (70M empirics may not extrapolate)
- Can Mercury's confidence-based adaptive refinement be replicated open-source?

## See Also

- [[llada]] — entity page: from-scratch full-scale DLM, no KV caching, beats GPT-4o on reversal task
- [[block-diffusion]] — entity page: block-level AR + within-block diffusion
- [[i-dlm]] — entity page: introspective consistency training + ISD algorithm
- [[dflash]] — entity page: block diffusion as a speculative decoding draft engine for AR models
- [[speculative-decoding]] — ISD in I-DLM and DFlash both connect DLMs to SD
- [[kv-cache]] — DLMs lose KV caching; BD3-LM and I-DLM restore it
- [[early-exit-inference]] — another form of adaptive compute at inference
