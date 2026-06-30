---
source_url: https://www.alphaxiv.org/overview/2026.dspark
ingested: 2026-06-30
sha256: pending
---

# DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation

**Wenfeng Liang et al.**
Published on alphaxiv — 2026-06-28

## Core Contributions

Addresses two bottlenecks in speculative decoding: (1) parallel drafters suffer "suffix decay" — accuracy drops sharply for later tokens in the draft block; (2) verifying a fixed-length draft regardless of confidence wastes compute under load.

### 1. Semi-Autoregressive Generation (SAG)

Combines a parallel backbone (DFlash architecture) with a lightweight sequential refinement module:

**Final logit at position k**: `y_k = U_k + B_k`
- `U_k`: parallel backbone logits (all positions simultaneously, constant cost)
- `B_k`: sequential transition bias from a **Markov head**

**Markov head**: `B(x_{k-1}, x_k) = (W_1[x_{k-1}]) W_2ᵀ[x_k]` — low-rank; conditions each position on the previous token sample. Latency overhead: ~1.5% of total drafting time. Effectively eliminates suffix decay while maintaining DFlash's constant O(1) draft cost.

### 2. Confidence-Scheduled Verification

A **confidence head** predicts scalar `c_k ∈ (0,1)` — probability that token at position k is accepted given all prior tokens in the block were accepted. Calibrated via **Sequential Temperature Scaling (STS)** so predicted probabilities match empirical acceptance rates.

### 3. Hardware-Aware Prefix Scheduler

Maximizes system throughput `Θ = τ × SPS(B)` where `B` = total batch token count, `τ` = expected accepted tokens, `SPS(B)` = profiled GPU steps/second.

Prefix survival probability for request r at position j: `a_{r,j} = ∏_{i≤j} c_{r,i}`

Greedy selection of tokens across all requests by marginal throughput gain. Load-aware: verifies more tokens when system is lightly loaded; prunes low-confidence tokens under congestion.

## Results

- +25–30% average accepted length over AR baselines (offline)
- Eliminates suffix decay; starts from higher accuracy baseline than small sequential drafters
- **Deployed on DeepSeek-V4-Flash**: +51% aggregate throughput at 80 TPS SLA; enables sustained >120 TPS tiers previously impossible under load
- Evaluated on Qwen and Gemma model families; compared to EAGLE-3 and DFlash baselines
