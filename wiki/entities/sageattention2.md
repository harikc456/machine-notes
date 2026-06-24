---
title: SageAttention2
created: 2026-06-24
updated: 2026-06-24
type: entity
tags: [quantization, attention, inference]
sources: [raw/papers/2411.10958v7.md]
confidence: high
---

# SageAttention2

**SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization**  
*Jintao Zhang, Haofeng Huang, Pengle Zhang, Jia Wei, Jun Zhu, Jianfei Chen — Tsinghua University, ICML 2025*

## Overview

Extends [[sageattention]] from INT8 to INT4/FP8. Quantizes Q,K to INT4 and P,V to FP8. The upgrade introduces two new precision challenges (INT4's narrow range; FP8's non-standard accumulator) and a corresponding new technique for each, on top of SageAttention's K-smoothing.

## Core Approach

- **Q,K**: INT4 via per-thread quantization
- **P̃**: FP8 with two-level accumulation
- **V**: FP8 (with optional V-smoothing)

### Technique 1: Per-Thread INT4 Quantization

**Problem**: INT4 range is [-7, +7] — 14 representable non-zero values. Even after K-smoothing, Q has outliers that destroy per-block INT4 accuracy.

**Standard per-token quantization** (one scale per token) would fix accuracy but requires dequantization overhead on every thread, reducing the throughput gain.

**Fix**: use the GPU thread-memory mapping dictated by the PTX `mma` instruction. Each thread corresponds to a contiguous memory block. Assign one INT4 quantization scale per thread — no extra dequantization instruction needed. Achieves accurate per-thread granularity with zero overhead.

Additionally: Q-smoothing (subtract per-block mean of Q block before quantization). Combined with SageAttention's K-smoothing, smoothing *both* Q and K is the most effective order (empirically: `smooth Q+K > smooth Q > smooth K > no smoothing`).

### Technique 2: Two-Level FP8 Accumulation for P̃V

**Problem**: The FP8 Matmul instruction (`mma.f32.f8.f8.f32`) uses FP22 accumulators internally (1 sign + 8 exp + 13 mantissa bits = FP22, not true FP32). For the P̃V Matmul, accumulated errors lead to visible artifacts.

**Fix**: two-level accumulation. After each FlashAttention block of P̃V, accumulate into a real FP32 buffer using rowsum/update. This confines FP22 errors to per-block scope, then corrects them at block boundaries.

### Optional V-Smoothing

Subtract per-channel mean of V before quantization; add back to attention output. Addresses V channel outliers (optional — benefit depends on model).

## Performance

| Metric | SA2 | SA1 (INT8) | FA2 | xformers |
|---|---|---|---|---|
| TOPS (RTX4090) | 481 | 340 | 164 | ~110 |
| Speedup vs FA2 | 3× | 2.1× | 1× | — |
| Speedup vs xformers | 4.5× | 2.7× | — | 1× |

**Hopper variant (SageAttention2-8b)**: quantizes Q,K to INT8 for compatibility; matches FlashAttention3(fp8) kernel speed while delivering higher accuracy (FA3-fp8 has visible quality degradation on CogVideoX; SA2-8b does not).

## Relationship to Other Work

- Extends [[sageattention]] (INT8 → INT4/FP8)
- Succeeded by [[sageattention3]] (FP4 microscaling for Blackwell GPUs)
- Both techniques (per-thread quant, two-level accumulation) are general — could apply to other low-bit attention designs

## See Also

- [[sageattention]] — predecessor (INT8 Q/K, FP16 P/V)
- [[sageattention3]] — successor (FP4 Q/K/P/V, Blackwell)
- [[flash-attention]] — base tiling structure
- [[quantization]] — quantization landscape
