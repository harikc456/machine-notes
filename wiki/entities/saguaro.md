---
title: Saguaro (Speculative Speculative Decoding)
created: 2026-05-16
updated: 2026-05-16
type: entity
tags: [inference, speculative]
sources: [raw/papers/2603.03251v3.pdf]
confidence: high
---

# Saguaro (Speculative Speculative Decoding)

**Speculative Speculative Decoding**
*Tanishq Kumar, Tri Dao, Avner May — Stanford / Princeton / Together AI, arXiv:2603.03251, May 2026*

## Core Idea

Standard [[speculative-decoding]] (SD) parallelizes token *verification* but retains a **sequential dependency between drafting and verification**: the draft model must wait for the verifier to finish before speculating the next round. Speculative Speculative Decoding (SSD) eliminates this sequential bottleneck by **predicting likely verification outcomes in advance** and pre-speculating for all of them in parallel while verification is still running.

The key insight: if the draft model can correctly predict which verification outcome will occur, it can have the next speculation ready immediately when verification completes — eliminating all drafting overhead.

## Algorithm

SSD runs the speculator and verifier on **separate hardware** (e.g., 1×H100 for speculator, 4×H100 for target):

1. **Speculate**: draft model sends speculated tokens to verifier
2. **Predict outcomes**: while verification runs, draft model predicts the most likely verification outcomes (how many tokens accepted + which bonus token)
3. **Pre-speculate for each outcome**: build a "speculation cache" S^T — a dictionary from predicted outcomes to pre-computed token sequences
4. **On verification result**: if outcome ∈ cache → immediately return pre-speculated tokens (cache hit, zero drafting latency); else → fall back to synchronous speculation

## Three Challenges Addressed

### 1. Outcome Prediction (§4.1)
- Verification outcome = (k accepted tokens, bonus token t*)
- Predicting k is straightforward; predicting t* is hard (residual distribution is complex)
- Solution: use draft logits to estimate most likely bonus token — achieves ~90% accuracy

### 2. Acceptance–Speculation Tradeoff (§4.2)
- More aggressive speculation → higher cache hit rate but lower per-token acceptance
- SSD derives the optimal balance; a new sampling scheme allows trading off these two objectives

### 3. Fallback Strategy (§4.3)
- At large batch sizes and high temperatures, cache misses become frequent
- Optimal fallback depends on batch size; Saguaro uses batch-size-adaptive fallback
- This ensures Saguaro outperforms SD by ≥20% even at large batches

## Results

Evaluated on Llama-3.1-70B (target) with Llama-3.2-1B (draft), TP=4 H100s, batch size 1, greedy decoding:

| Method | Decode tok/s | Speedup vs AR |
|---|---|---|
| AR | ~55 | 1.0× |
| Speculative Decoding | ~115 | ~2.1× |
| **Saguaro (SSD)** | **~220** | **~4.0×** |

- **30% faster than strongest SD baselines** (vLLM, SGLang)
- **Up to 5× faster than AR decoding**
- Lossless: produces the same distribution as the target model
- Improves the throughput-latency Pareto frontier across all batch sizes

## Key Distinction from Tree-Based Methods

Tree-based SD (SpecBranch, etc.) increases verifier compute by verifying a tree of tokens. SSD instead scales up *speculator* compute — predicting outcomes in parallel — with **no additional verifier compute**. These are complementary and can be combined.

## See Also

- [[speculative-decoding]] — foundational algorithm that SSD extends
- [[layerskip]] — self-speculative decoding (no separate draft model)
- [[early-exit-inference]] — another compute-efficiency angle
- [[continuous-batching]] — serving scheduler; interacts with SSD's batching strategy
- [[paged-attention]] — memory management for serving; orthogonal to SSD
