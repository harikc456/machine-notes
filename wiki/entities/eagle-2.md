---
title: EAGLE-2 (Dynamic Draft Trees)
created: 2026-05-31
updated: 2026-05-31
type: entity
tags: [inference, speculative]
sources: [raw/papers/2406.16858v2.pdf]
confidence: high
---

# EAGLE-2

**EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees**
*Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang — Peking University / Microsoft Research / University of Waterloo / Vector Institute, arXiv:2406.16858, Jun 2024*

## Core Idea

[[eagle]] uses a **static draft tree**: the same fixed shape regardless of context. EAGLE-2 observes that draft token acceptance rates are **context-dependent**, not just position-dependent — some queries are easy (next token is nearly certain) while others branch widely.

Key insight: EAGLE's draft model is **well-calibrated** — its confidence scores (token probabilities) closely approximate the true acceptance rates. This makes it feasible to use confidence scores to dynamically expand or prune the draft tree at runtime without any extra training.

## Algorithm

1. Use the existing EAGLE draft model's probability estimates as acceptance rate proxies
2. Dynamically adjust the draft tree: add more candidate branches where confidence is high; prune where confidence is low
3. Derive the optimal dynamic tree structure from the calibrated draft model at each decoding step

No extra training required — EAGLE-2 applies directly on top of any trained EAGLE draft model.

## Results

Evaluated on Vicuna (7B, 13B), LLaMA2-Chat (7B, 13B, 70B), LLaMA3-Instruct (8B, 70B); MT-bench, HumanEval, GSM8K, CNN/DM, NQ:

- **3.05×–4.26× speedup** (temperature=0)
- **2.5×–5× speedup** (temperature=1, non-greedy)
- **20–40% faster than EAGLE-1** on all tested models/tasks
- Lossless: output distribution identical to target model, provably
- No additional training; works directly with existing EAGLE checkpoints

## Key Distinction from EAGLE-1

EAGLE-1 uses a tree with candidates fixed at each position (k candidates always added). For a simple query like "10+2=", the next token is clearly "12" — adding a second candidate wastes computation. EAGLE-2 adds only 1 candidate here. For an ambiguous query, EAGLE-2 expands the tree more aggressively.

## See Also

- [[eagle]] — EAGLE-1: the base method with static draft trees
- [[eagle-3]] — further improvement: removes feature prediction constraint, enables data scaling
- [[speculative-decoding]] — foundational algorithm
- [[saguaro]] — orthogonal: parallelizes drafting and verification on separate hardware; 30% over SD
- [[dflash]] — replaces AR drafting with parallel block-diffusion drafting; 2.5× over EAGLE-3
