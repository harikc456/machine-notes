---
title: TriAttention
created: 2026-05-16
updated: 2026-05-16
type: entity
tags: [kv-cache, inference, attention]
sources: [raw/papers/2604.04921v1.pdf]
confidence: high
---

# TriAttention

**TriAttention: Efficient Long Reasoning with Trigonometric KV Compression**
*Weian Mao, Xi Lin, Wei Huang, Bohan Zhuang, Yuxin Xie, Tianfu Fu, Song Han, Yukang Chen — MIT / NVIDIA / ZJU, arXiv:2604.04921, Apr 2026*

## Core Idea

Existing KV cache compression methods estimate token importance from **post-RoPE** Q/K representations. This is problematic: RoPE rotates Q/K vectors with position, making representative queries unstable — only the most recent queries have up-to-date orientations, leaving only a tiny window for importance estimation.

TriAttention instead works in **pre-RoPE space**, exploiting a structural property of Q/K vectors: they are **highly concentrated around fixed non-zero centers** that remain stable across positions and contexts ("Q/K concentration"). This allows attention patterns to be approximated by a **trigonometric series** in Q-K distance, enabling reliable importance scoring without the rotation instability.

## The Q/K Concentration Phenomenon

In pre-RoPE space, for a query q at position p_q and key k at position p_k:

```
logit(q, k) = Σ_f ‖q_f‖ ‖k_f‖ cos(ω_f·Δ + φ_f)
            = Σ_f [a_f cos(ω_f·Δ) + b_f sin(ω_f·Δ)]   (trigonometric series in Δ = p_q - p_k)
```

where coefficients a_f, b_f are determined by the pre-RoPE Q/K centers. This shows that when Q/K are concentrated, attention logits are a function of **Q-K distance alone** — predictable from the centers via offline calibration.

Validated across 1152 attention heads in Qwen3-8B: mean Pearson r̄ ≈ 0.72 between predicted and actual attention logits. Holds across Qwen3, Qwen2.5, and Llama3.

## Scoring Function

TriAttention combines two signals:

**1. Trigonometric series score** (captures distance preference):
```
S_trig(k, Δ) = Σ_f ‖E[q_f]‖ · ‖k_f‖ · cos(ω_f·Δ + φ_f)
```

**2. Norm-based score** (catches low-norm keys at any distance):
```
S_norm(k) = Σ_f (1 - R_f) · E[‖q_f‖] · ‖k_f‖
```

The two are adaptively weighted by **Q/K concentration** (Mean Resultant Length R_f):
- High R_f → S_trig dominates (concentration is strong, trigonometric series is reliable)
- Low R_f → S_norm contributes more (fall back to norms as complementary signal)

Final score: **S(k, Δ) = S_trig(k, Δ) + S_norm(k)**

## Key Differences from Existing Methods

| Aspect | H₂O / Attention-based | Norm-based (VATP) | **TriAttention** |
|---|---|---|---|
| Representation space | Post-RoPE | Post-RoPE | **Pre-RoPE** |
| Observation window | Recent queries only | Recent queries only | **All positions** (stable centers) |
| Captures direction? | Partly (rotation unstable) | No | **Yes** (trigonometric series) |
| Captures norms? | No | Yes | **Yes** (norm-based complement) |

## Results

On AIME25 with 32K-token generation (Qwen3-8B reasoning model):

| Method | Accuracy | Throughput | KV Memory |
|---|---|---|---|
| Full Attention | 40.8% | 1× | 100% |
| R-KV | 32.9% | 2.5× | — |
| **TriAttention** | **40.8%** | **2.5× faster** | **10.7× smaller** |

On MATH-500 (1024/32K KV budget = 3% cache):
- TriAttention: 68.4% vs R-KV: 69.6% vs Full: 72.5% — competitive at extreme compression
- Enables OpenClaw deployment on a single consumer GPU for long-context reasoning

## Offline Calibration

Q distribution centers E[q_f] are computed once per model from a calibration dataset in pre-RoPE space. At inference, no extra compute is needed — keys are already in the KV cache, and scoring uses the precomputed centers.

## See Also

- [[kv-cache]] — background and compression landscape
- [[h2o]] — attention-based eviction (post-RoPE); TriAttention improves on this approach
- [[polarquant]] — quantization approach to KV compression; complementary (could combine)
- [[turboquant]] — quantization approach; complementary
- [[kv-cache-compression-comparison]] — side-by-side comparison of KV compression methods
- [[qknorm]] — Q/K normalization; related attention space geometry
