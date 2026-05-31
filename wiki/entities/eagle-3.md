---
title: EAGLE-3 (Training-Time Test for Speculative Decoding)
created: 2026-05-31
updated: 2026-05-31
type: entity
tags: [inference, speculative]
sources: [raw/papers/2503.01840v3.pdf]
confidence: high
---

# EAGLE-3

**EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test**
*Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang — Peking University / Microsoft Research / University of Waterloo / Vector Institute, arXiv:2503.01840, Apr 2025*

## Core Idea

[[eagle]] and [[eagle-2]] hit a data scaling wall: adding more training data yields diminishing acceptance rate improvements. EAGLE-3 diagnoses the root cause and fixes it with two architectural changes:

1. **Abolish feature prediction**: removes the feature-level prediction constraint (l_fea loss) and switches to **direct token prediction** (l_token only). This gives the draft model full expressiveness.
2. **Multi-layer feature fusion**: instead of reusing only top-layer features, EAGLE-3 integrates low-, mid-, and high-level features from the target model, capturing richer semantic information across abstraction levels.
3. **Training-time test**: fixing the distribution shift problem created by removing feature prediction.

## The Distribution Shift Problem and Training-Time Test

Removing feature prediction fixes Step 1 (first draft token) but causes distribution shift for Step 2 (second+ draft tokens): the model's output at Step 1 is now an unconstrained vector â_{t+1} that differs from the true feature f_{t+1}, so Step 2 sees an out-of-distribution input during inference.

**Training-time test** (bottom of Figure 3 in paper): during training, Step 1 runs normally and produces â_{t+1}. Rather than discarding it, Step 2 **uses â_{t+1} as its input** (as in test time) instead of the ground-truth feature. This makes training distribution match inference distribution, enabling the draft model to handle its own imperfect predictions.

## Key Results

Evaluated on LLaMA-Instruct 3.1 8B (MT-bench, chat models) and DeepSeek-R1-Distill-LLaMA-8B (GSM8K, reasoning); comparison against Medusa, HASS, EAGLE, EAGLE-2:

| Model | EAGLE-2 | EAGLE-3 |
|---|---|---|
| Vicuna 13B | 3.1× | 5.6× |
| LLaMA-Instruct 3.1 8B | 3.2× | 4.4× |
| LLaMA-Instruct 3.3 70B | 2.8× | 4.1× |
| DeepSeek R1 LLaMA 8B | 3.4× | 5.0× |

- **Up to 6.5× speedup** vs AR baseline
- **~1.4× improvement over EAGLE-2**
- **1.38× throughput gain** in SGLang at batch size 64
- **Data scaling law unlocked**: EAGLE-3's acceptance rate grows proportionally with training data — EAGLE-1/2 showed near-flat scaling curves; EAGLE-3 shows increasing speedup as data × increases

## Scaling Behavior

The critical new property: EAGLE-3 exhibits a proper scaling curve (Figure 1 in paper). Trained on ~8× more data than EAGLE achieves 1.4× speedup over EAGLE-2. This opens a path to further gains via data scaling that was unavailable in prior EAGLE versions.

## See Also

- [[eagle]] — EAGLE-1: the base method; feature-level AR
- [[eagle-2]] — dynamic draft trees; 20-40% over EAGLE-1
- [[dflash]] — uses block diffusion for parallel drafting; achieves 2.5× over EAGLE-3 on most benchmarks
- [[speculative-decoding]] — foundational algorithm
- [[saguaro]] — orthogonal hardware-parallelism approach
- [[layerskip]] — self-speculative decoding without a draft model
