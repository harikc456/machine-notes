---
title: EAGLE (Speculative Sampling via Feature Uncertainty)
created: 2026-05-31
updated: 2026-05-31
type: entity
tags: [inference, speculative]
sources: [raw/papers/2401.15077v3.pdf]
confidence: high
---

# EAGLE

**EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty**
*Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang — Peking University / Microsoft Research / University of Waterloo / Vector Institute, arXiv:2401.15077, Mar 2025*

## Core Idea

Standard [[speculative-decoding]] uses a token-level draft model, but token sequences are harder to autoregressively predict than *feature* sequences. EAGLE instead performs **autoregression at the second-to-top layer** (the feature/hidden-state level, just before the LM head) of the target model.

Two key observations:
1. Feature-level autoregression is simpler than token-level — features vary more smoothly than discrete tokens
2. Feature uncertainty: the feature following token t_I is contingent on which token was sampled (e.g., "am" vs "always" each produce a different next-feature). EAGLE resolves this by **inputting the token sequence shifted one time step ahead** alongside the feature sequence, so the draft model knows the sampling outcome.

## Architecture

- Draft model: a single lightweight transformer decoder layer (the "plug-in"), trained on target model features
- Input: concatenated [feature sequence F_{1:t}, token sequence T_{2:t+1}] (features + shifted tokens)
- Output: predicted next feature F̂_{t+1}; target LM head converts F̂ to draft token distribution
- Only the plug-in layer is trained; target model is frozen

Training cost: ~1-2 days on 4× A100 (40G) for 70B models, using <1B tokens from ShareGPT.

## Results

Evaluated on Vicuna (7B, 13B, 33B), LLaMA2-Chat (7B, 13B, 70B), Mixtral 8×7B; MT-bench, GSM8K, HumanEval, Alpaca:

- **2.7×–3.5× speedup** on LLaMA2-Chat 70B at temperature=0
- **Doubled throughput** at temperature=1 (non-greedy)
- Lossless: output distribution theoretically identical to target model in both greedy and non-greedy settings
- Draft model accuracy ~0.8 — significantly better than Medusa (~0.6) or Lookahead (~0.4)

## Key Advantages over Prior Methods

| Property | Medusa | Lookahead | EAGLE |
|---|---|---|---|
| Draft accuracy | ~0.6 | lower | ~0.8 |
| Lossless (non-greedy) | ❌ | ❌ | ✅ |
| Fine-tunes backbone | No | No | No |
| Training cost | Low | None | Low |

## Limitations / What EAGLE-2 Fixed

- Static draft tree structure: assumes acceptance rate is position-dependent only, ignoring context
- Scaling plateau: scaling up training data provided diminishing returns due to the feature prediction constraint

## See Also

- [[eagle-2]] — adds context-dependent dynamic draft trees; 20-40% faster than EAGLE-1
- [[eagle-3]] — replaces feature prediction with direct token prediction + training-time test; enables data scaling law
- [[dflash]] — uses block diffusion for parallel drafting; 2.5× over EAGLE-3
- [[speculative-decoding]] — foundational algorithm EAGLE builds on
- [[saguaro]] — orthogonal: parallelizes drafting and verification across separate hardware
- [[layerskip]] — self-speculative decoding (no draft model); lower overhead but lower quality
