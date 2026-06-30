---
title: DSpark (Semi-Autoregressive Speculative Decoding)
created: 2026-06-30
updated: 2026-06-30
type: entity
tags: [inference, speculative, deepseek]
sources: [raw/papers/dspark.md]
confidence: high
---

# DSpark

**DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation**
*Wenfeng Liang et al. — alphaxiv, 2026-06-28*
*Deployed in DeepSeek-V4-Flash production serving*

## Problem

Speculative decoding draft models fall into two categories with complementary failure modes:

- **Autoregressive (EAGLE-3)**: high acceptance rate but drafting cost grows linearly with draft length (T_draft = γ × t_step) — caps speedup
- **Parallel (DFlash)**: constant draft cost but suffers **suffix decay** — accuracy of later positions drops sharply because each token is predicted independently of its predecessors in the block ("multi-modal collisions")

Additionally: standard SD verifies a **fixed-length draft** regardless of how likely it is to be accepted, wasting verifier compute on low-confidence drafts under load.

## DSpark's Three Contributions

### 1. Semi-Autoregressive Generation (SAG)

Combines DFlash's parallel backbone with a lightweight sequential refinement module:

**Architecture**:
- **Parallel backbone** (DFlash): generates hidden states + base logits `U_k` for all γ positions simultaneously (constant cost)
- **Markov head** (sequential): computes transition bias `B_k` conditioning on the previously sampled token `x_{k-1}`:

  `y_k = U_k + B_k`

  `B(x_{k-1}, x_k) = (W_1[x_{k-1}]) W_2ᵀ[x_k]` — low-rank factorization

The Markov head adds token-to-token dependency at negligible cost (~1.5% of total draft latency) because it only runs small matrix lookups on a single token at a time. Result: **suffix decay eliminated** while preserving DFlash's O(1) parallel draft cost.

Compared to more complex sequential modules (RNN head), the Markov head achieves nearly identical acceptance rates with faster sampling.

### 2. Confidence Head + Sequential Temperature Scaling (STS)

A learned **confidence head** attached to the drafter predicts, for each position k:
`c_k ∈ (0,1)` = P(token k accepted | tokens 1..k-1 accepted)

**Prefix survival probability** for request r at position j:
`a_{r,j} = ∏_{i≤j} c_{r,i}`

Raw confidence scores are calibrated via **Sequential Temperature Scaling (STS)** — a calibration step ensuring predicted probabilities match empirical acceptance rates in practice (e.g., predicted 80% → actual 80% acceptance).

### 3. Hardware-Aware Prefix Scheduler

Dynamically determines how many tokens to verify per request in each batch, maximizing:

`Θ = τ × SPS(B)`

where `B` = total tokens in the batch, `τ` = expected accepted tokens, `SPS(B)` = profiled GPU steps/second curve.

**Load-aware behavior**:
- Lightly loaded: verify more tokens (even lower-confidence ones) → minimize per-user latency
- Heavily loaded: prune low-confidence tails → preserve throughput across all users

Greedy selection: rank draft token additions across all active requests by marginal gain in expected accepted tokens per unit batch-size increase.

## Results

### Offline (Draft Quality)

- +25–30% improvement in average accepted length vs AR baselines
- Eliminates suffix decay (Figure 2: DSpark acceptance flat across positions; DFlash decays sharply)
- Starts from higher baseline accuracy than small sequential drafters

### Production (DeepSeek-V4-Flash)

| Operating point | Metric | Value |
|---|---|---|
| 80 TPS/user SLA | Aggregate throughput gain | **+51%** vs MTP-1 baseline |
| >120 TPS/user | Availability | **Newly enabled** (previously unsustainable under load) |

Significantly shifts the throughput-interactivity Pareto frontier — more users can be served at high generation speeds simultaneously.

## Position in the Speculative Decoding Landscape

| System | Draft type | Draft cost | Suffix decay | Confidence-aware |
|---|---|---|---|---|
| EAGLE-3 | AR (feature-level) | O(γ) | No | No |
| DFlash | Parallel (block diffusion) | O(1) | Yes | No |
| **DSpark** | Semi-AR (parallel + Markov) | O(1) | **No** | **Yes** |
| Saguaro (SSD) | Any + speculator parallelism | varies | — | No |

DSpark and [[saguaro]] are orthogonal: DSpark optimizes *what* to draft and *how much* to verify; Saguaro parallelizes *when* to draft vs. verify across separate hardware.

## See Also

- [[speculative-decoding]] — algorithm and landscape (draft-then-verify, rejection sampling guarantee)
- [[dflash]] — DFlash: the parallel backbone DSpark builds on; block diffusion drafting; 6×+ lossless
- [[eagle-3]] — AR drafting baseline DSpark outperforms in average accepted length
- [[saguaro]] — orthogonal hardware-parallelism approach (separate speculator/verifier)
- [[deepseek-v4]] — deployment target (DeepSeek-V4-Flash production serving)
- [[inference-kv-speculative]] — full speculative decoding deep-dive including DFlash, EAGLE family, Saguaro
