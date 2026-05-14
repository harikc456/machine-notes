---
title: DeepSeek-V4
created: 2026-05-14
updated: 2026-05-14
type: entity
tags: [model, architecture, inference, attention, training, deepseek, open-source]
sources: [raw/papers/DeepSeek_V4.pdf]
confidence: high
---

# DeepSeek-V4

**DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence**
*DeepSeek-AI*

## Overview

Preview release of the DeepSeek-V4 series — two MoE LLMs designed for 1M-token context at high efficiency:

| Model | Total Params | Active Params | Context |
|---|---|---|---|
| DeepSeek-V4-Pro | 1.6T | 49B | 1M tokens |
| DeepSeek-V4-Flash | 284B | 13B | 1M tokens |

DeepSeek-V4-Pro-Max (maximum reasoning mode) redefines SOTA for open models. In the 1M-context setting, DeepSeek-V4-Pro requires only **27% of single-token inference FLOPs** and **10% of KV cache** compared with DeepSeek-V3.2.

## Key Technical Contributions

### 1. Hybrid Attention: CSA + HCA
- **Compressed Sparse Attention (CSA)**: reduces long-context compute complexity
- **Heavily Compressed Attention (HCA)**: further compresses attention at extreme context lengths
- Together they reduce per-token FLOPs and KV cache by ~3.7× and ~9.8× respectively at 1M tokens

### 2. Manifold-Constrained Hyper-Connections (mHC)
- Replaces standard residual connections with dynamic residual matrices
- Projects onto Birkhoff polytope (doubly-stochastic manifold) for training stability
- See [[mhc]] for the original paper and [[hyper-connections]] for background

### 3. Muon Optimizer
- Replaces AdamW for faster convergence and greater training stability
- Applies orthogonal gradient updates via Nesterov momentum + Newton-Schulz iteration
- See also [[deepseek-v3-2]] which also uses Muon

### 4. Infrastructure
- TileLang for flexible GPU kernel development
- On-disk KV cache storage for extended context serving
- Fine-grained expert parallelism with communication-computation overlap

## Benchmark Results (DeepSeek-V4-Pro-Max vs. state-of-art)

- SimpleQA Verified: 77.3% (vs Claude-Opus-4.6-Max 75.6%)
- HLE: 44.4% (vs GPT-5.4-High 44.4%)
- Apex ShortList: competitive with frontier closed-source models
- Codeforces / SWE-Verified / Terminal Bench: strong agentic capabilities

## Relationship to Prior Work

- Inherits [[deepseek-v3-2]] designs (MLA, DeepSeekMoE, DualPipe)
- Introduces CSA/HCA as a step beyond [[deepseek-v3-2]]'s DSA
- mHC provides stable residual connections at trillion-parameter scale
- Engram ([[engram]]) is a related DeepSeek research line on conditional memory, not yet in V4

## Open Questions

- How does CSA interact with [[speculative-decoding]] at 1M-token context?
- Extent of KV cache reduction with [[kv-cache]] quantization on top of HCA
