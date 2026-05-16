---
title: Block Diffusion (BD3-LM)
created: 2026-05-16
updated: 2026-05-16
type: entity
tags: [architecture, inference, training]
sources: [raw/papers/2503.09573v3.pdf]
confidence: high
---

# Block Diffusion (BD3-LM)

**Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models**
*Arriola, Gokaslan, Chiu, Yang et al., Cornell Tech / Stanford / Cohere, ICLR 2025, arXiv:2503.09573*

## Core Idea

Discrete diffusion language models offer parallelizable generation but have two critical limitations: fixed-length output and no KV caching (bidirectional attention blocks reuse). Block Diffusion (BD3-LM) overcomes both by defining an **autoregressive distribution over blocks**, with **diffusion operating within each block**.

The generative process decomposes as:

```
log p(x) = Σ_b log p(x^b | x^<b)        (AR over blocks)
p(x^b | x^<b) via discrete diffusion      (diffusion within each block)
```

Block size L' controls the AR/diffusion tradeoff: L'=1 recovers standard AR; L'=L recovers a standard fixed-length diffusion model.

## Architecture

- Single transformer x_θ with **block-causal attention mask**: tokens in block b attend to all tokens in blocks 1..b
- At inference, KV pairs from previous blocks are cached (K^{1:b-1}, V^{1:b-1}) exactly as in AR decoding
- At training, a vectorized algorithm processes all B blocks in one forward pass by concatenating clean and noisy inputs with a specialized attention mask
- Built on the **masked diffusion** framework (D3PM/MDLM): tokens are progressively masked, with noise schedule α_t controlling masking probability

## Key Technical Contributions

1. **Efficient vectorized training**: computes the block diffusion objective for all B blocks simultaneously in one forward pass — avoids the naïve B-pass loop
2. **Gradient variance analysis**: identifies high training variance (not algorithmic issues) as the root cause of the perplexity gap between diffusion and AR models
3. **Data-driven noise schedules**: custom per-dataset noise schedules that minimize gradient variance, largely closing the gap to AR perplexity
4. **Variable-length generation**: block-AR structure enables arbitrary sequence lengths, unlike prior diffusion models

## Properties vs. Baselines

| Property | AR | Diffusion | BD3-LM |
|---|---|---|---|
| High quality (perplexity) | ✅ | ❌ | ✅ |
| Arbitrary-length output | ✅ | ❌ | ✅ |
| KV caching | ✅ | ❌ | ✅ |
| Parallel token sampling | ❌ | ✅ | ✅ (within block) |

## Results

- Sets new SOTA perplexity among discrete diffusion models on LM1B
- With tuned noise schedule + block size L'=1, matches AR perplexity (22.88 PPL on LM1B)
- Tractable likelihood estimates (unlike Gaussian continuous diffusion LMs)
- Generates sequences longer than training context

## Relationship to I-DLM

[[i-dlm]] (Apr 2026) takes a different approach to closing the AR/diffusion quality gap: rather than block-AR structure, it introduces **introspective consistency training** and **causal attention** into diffusion models. Both target the same quality gap but from different angles — BD3-LM is more structurally similar to AR, while I-DLM converts pretrained AR models into DLMs.

## See Also

- [[i-dlm]] — complementary approach to high-quality diffusion LMs; introspective consistency training
- [[diffusion-language-models]] — concept page for the DLM landscape
- [[speculative-decoding]] — orthogonal inference speedup; BD3-LM within-block parallelism is related
- [[kv-cache]] — BD3-LM re-enables KV caching that standard diffusion LMs cannot use
- [[early-exit-inference]] — another axis of inference efficiency
