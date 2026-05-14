---
title: DeepSeek-V3.2
created: 2026-05-14
updated: 2026-05-14
type: entity
tags: [model, architecture, inference, training, sparsity, deepseek, open-source, benchmark]
sources: [raw/papers/2512.02556v1.pdf]
confidence: high
---

# DeepSeek-V3.2

**DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models**
*DeepSeek-AI, arXiv:2512.02556, Dec 2025*

## Overview

DeepSeek-V3.2 addresses three deficiencies in prior open-source models:
1. Vanilla attention is inefficient for long sequences
2. Insufficient post-training compute
3. Weak agentic generalization

Key result: closes the performance gap with proprietary models in reasoning and agentic tasks at substantially lower cost.

## Key Technical Contributions

### 1. DeepSeek Sparse Attention (DSA)
- Highly efficient attention mechanism reducing computational complexity
- Preserves performance in long-context scenarios
- Precursor to CSA/HCA in [[deepseek-v4]]

### 2. Scalable Reinforcement Learning Framework
- Robust RL protocol with scalable post-training compute
- Post-training budget exceeds **10% of pre-training cost** — significantly more than prior models
- Enables DeepSeek-V3.2-Speciale to match Gemini-3.0-Pro

### 3. Large-Scale Agentic Task Synthesis Pipeline
- Generates 1,800+ distinct environments and 85,000+ complex prompts
- Synthesized data drives RL process for instruction-following in complex interactive environments
- Cold-start phase uses DeepSeek-V3 methodology to unify reasoning + tool use

## Benchmark Highlights

| Benchmark | DeepSeek-V3.2-Speciale | Notes |
|---|---|---|
| AIME 2025 | 96.0 (Pass@1) | Competitive with top closed-source |
| HMMT 2025 | 90.2 (Pass@1) | Feb competition |
| Codeforces | 2708 rating | Gold-level competitive programming |
| IMO 2025 | Gold medal | |
| IOI 2025 | Gold medal | |
| CMO 2025 | Gold medal | |
| SWE-Verified | 46.4% | |

## Relationship to Prior Work

- Builds on DeepSeek-V3 training methodology
- Predecessor to [[deepseek-v4]], which replaces DSA with CSA/HCA
- Both V3.2 and V4 use [[mixture-of-experts]] (MoE) architecture
- Post-training RL framework is distinct from the [[speculative-decoding]] line of inference acceleration

## Notable Positioning

DeepSeek-V3.2 is characterized as a cost-efficient open alternative — "significantly narrowing the performance gap between open and frontier proprietary models while incurring substantially lower costs."
