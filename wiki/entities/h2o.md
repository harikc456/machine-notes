---
title: H₂O (Heavy-Hitter Oracle)
created: 2026-05-14
updated: 2026-05-14
type: entity
tags: [kv-cache, inference, attention, sparsity]
sources: [raw/papers/2306.14048v3.pdf]
confidence: high
---

# H₂O — Heavy-Hitter Oracle

**H₂O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models**
*Zhang et al., NeurIPS 2023, arXiv:2306.14048*

## Overview

H₂O is a **KV cache eviction framework** based on the observation that a small subset of tokens — called **Heavy Hitters (H₂)** — contribute most of the value when computing attention scores. Retaining H₂ tokens plus a recency window achieves near-lossless compression.

See [[kv-cache]] for background on why KV cache memory is a bottleneck.

## Key Observations

1. **Attention sparsity**: KV cache matrices in LLMs are >95% sparse at inference time
2. **Heavy-hitter concentration**: Accumulated attention scores follow a power-law distribution — a small set of tokens absorb most attention weight
3. **Heavy-hitters correlate with frequent co-occurrence**: H₂ emergence is natural and tied to text statistics
4. **Greedy local statistics suffice**: Retaining H₂ based on *local* accumulated scores (preceding tokens only) is as effective as oracle future-aware selection

## Algorithm

H₂O maintains a fixed-size KV cache budget by:
1. Always keeping recent tokens (recency window)
2. Tracking accumulated attention scores per token
3. Evicting tokens with lowest scores when cache is full

The eviction policy is formulated as a **dynamic submodular optimization problem** with theoretical guarantees under mild assumptions.

## Results

With 20% heavy hitters:
- Throughput improvement: up to **29×** on OPT-6.7B and OPT-30B
- Latency reduction: up to **1.9×**
- Tested against DeepSpeed Zero-Inference, Hugging Face Accelerate, FlexGen

Models validated: OPT, LLaMA, GPT-NeoX

## Relationship to Other Work

- [[kv-cache]]: general KV cache compression landscape
- [[polarquant]] and [[turboquant]]: quantization-based approaches that are orthogonal/complementary to eviction
- [[kv-cache-compression-comparison]]: head-to-head comparison

## Limitations

- Eviction strategies face challenges in needle-in-haystack retrieval scenarios — critical tokens may be incorrectly evicted if they receive low attention scores early but are queried late
- [[polarquant]] and [[turboquant]] avoid this by compressing all tokens rather than evicting
