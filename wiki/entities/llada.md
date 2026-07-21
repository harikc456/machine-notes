---
title: LLaDA
created: 2026-07-21
updated: 2026-07-21
type: entity
tags: [model, architecture, training, inference]
sources: [raw/papers/2502.09992v3.md]
confidence: high
---

# LLaDA (Large Language Diffusion with mAsking)

**Large Language Diffusion Models**
*Nie, Zhu, You, Zhang, Ou, Hu, Zhou, Lin, Wen, Li — Renmin University of China / Ant Group, NeurIPS 2025*

## Overview

A diffusion language model trained **from scratch** at genuine LLM scale (8B params, 2.3T tokens) under the standard pretrain+SFT paradigm — not adapted from a pretrained AR model. Demonstrates that scalability, in-context learning, and instruction-following are properties of generative modeling principles (maximum-likelihood / KL minimization) broadly, not exclusive to the autoregressive factorization.

## Key Technical Contributions

- **Forward/reverse masking process**: tokens masked independently at ratio `t ~ U(0,1)`; a bidirectional-attention Transformer mask predictor recovers all masked tokens simultaneously, trained on a provable upper bound of the negative log-likelihood.
- **Low-confidence remasking** at inference (à la MaskGIT) rather than purely random remasking.
- **Flexible sampling**: supports AR-style and block-diffusion sampling post-hoc without retraining.
- Uses vanilla multi-head attention (no GQA) — incompatible with KV caching, a direct cost of full-recompute bidirectional attention at every denoising step.

## Benchmark Results

- 8B Base surpasses LLaMA2 7B Base on nearly all of 15 zero/few-shot tasks; competitive overall with LLaMA3 8B Base, with particular strength on math (GSM8K, Math) and Chinese-language tasks.
- Instruction-tuned (SFT only, no RL) trails LLaMA3 8B Instruct slightly but shows strong multi-turn dialogue and instruction-following.
- **Beats GPT-4o on a poem-completion reversal task**, addressing the "reversal curse" that afflicts left-to-right AR models — attributed to LLaDA's non-causal, multi-directional training objective.

## Relationships to Other Entities

- Foundational full-scale reference point for the [[diffusion-language-models]] landscape: [[block-diffusion]] (BD3-LM) restores KV caching by wrapping AR structure around blocks of diffusion; [[i-dlm]] converts a pretrained AR model into a DLM via introspective-consistency training rather than training from scratch like LLaDA.
- Referenced as a serving/production baseline (LLaDA-2.1-mini) in [[i-dlm]]'s throughput and accuracy comparisons.

## Open Questions

- Does the from-scratch DLM approach close the remaining quality gap to AR models at 70B+ scale?
- Can LLaDA's bidirectional training be combined with I-DLM's introspective consistency or BD3-LM's block structure for a DLM that is both from-scratch-scalable and KV-cache-compatible?

## See Also

- [[diffusion-language-models]] — concept page covering the DLM landscape and quality-gap causes
- [[block-diffusion]] — restores KV caching via AR-over-blocks
- [[i-dlm]] — introspective-consistency conversion of AR models into DLMs
- [[kv-cache]] — LLaDA's incompatibility with KV caching is a direct consequence of bidirectional attention
