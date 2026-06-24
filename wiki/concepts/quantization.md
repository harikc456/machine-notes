---
title: Quantization
created: 2026-05-14
updated: 2026-06-24
type: concept
tags: [quantization, inference, training, attention]
sources: [raw/papers/A Visual Guide to Quantization.md, raw/papers/2502.02617v1.pdf, raw/papers/2504.19874v1.pdf, raw/papers/2410.02367v9.md, raw/papers/2411.10958v7.md, raw/papers/2505.11594v3.md]
confidence: high
---

# LLM Quantization

## Overview

Quantization reduces the numerical precision of model parameters (and/or activations) to decrease memory footprint and increase computational throughput. Core trade-off: **compression ratio vs. task accuracy**.

## Data Types

| Type | Bits | Use Case |
|---|---|---|
| FP32 | 32 | Full precision training |
| BF16 | 16 | Mixed-precision training/inference |
| FP16 | 16 | Standard inference |
| INT8 | 8 | Post-training quantization (PTQ) |
| FP8 | 8 | Training + inference (emerging) |
| INT4/NF4 | 4 | Aggressive weight compression |
| INT2/1-bit | 2/1 | Extreme compression (BitNet etc.) |

A floating-point number = sign bit + exponent bits + mantissa (fraction) bits.
**BF16** has the same exponent range as FP32 (8 exponent bits) but less precision — better for ML than FP16 which clips large values.

## Categories

### Weight Quantization
Quantize model weights; activations remain in higher precision:
- **Post-Training Quantization (PTQ)**: quantize after training, no retraining
- **Quantization-Aware Training (QAT)**: train with quantization in the loop
- Key challenge: outlier weights — a few large-magnitude weights cause disproportionate quantization error

### KV Cache Quantization
Quantize the K/V tensors stored during inference:
- [[polarquant]]: polar coordinate transform, no normalization needed
- [[turboquant]]: random rotation + MSE quantizer + QJL residual; data-oblivious, near-optimal within that class
- [[spectralquant]]: calibrated eigenvector rotation + selective QJL on signal dims only; exploits 97% spectral gap in KV keys; 15s calibration; strictly better than TurboQuant (+1.7–2.8 pp, 18.6% better compression)
- Challenge: per-block normalization parameters add overhead — addressed by all three via rotation preconditioning

### Attention Computation Quantization

Quantize the Matmul operations *within* the attention forward pass (distinct from storing KV tensors — see KV Cache Quantization above). Targets the O(N²) attention cost at long sequences.

**SageAttention family** (Tsinghua, ICLR/ICML/NeurIPS 2025): plug-and-play drop-in for [[flash-attention]]; no model modification required.

| Version | Q,K dtype | P,V dtype | Peak TOPS | Speedup vs FA2 | Hardware |
|---|---|---|---|---|---|
| [[sageattention]] | INT8 | FP16/FP16-acc | 340 | 2.1× | RTX4090/3090+ |
| [[sageattention2]] | INT4 (per-thread) | FP8 (2-level acc) | 481 | 3× | RTX4090/L20+ |
| [[sageattention3]] | FP4 (NVFP4 micro) | FP4 | 1038 | 5× | RTX5090 (Blackwell only) |

**Shared insight across the series**: outliers in Q and K are the main precision challenge.
- [[sageattention]]: smooth K by subtracting per-channel mean (K-smoothing)
- [[sageattention2]]: additionally smooth Q; use per-thread INT4 quantization (zero dequant overhead)
- [[sageattention3]]: FP4 microscaling (1×16 group); two-level quantization for attention map P̃

All three achieve near-zero end-to-end accuracy loss across LLM, image-gen, and video-gen models.

### Activation Quantization
Quantize activations (inputs/outputs of layers) during inference — most challenging due to outliers.

## Key Concepts

### Calibration
Most PTQ methods require a small calibration dataset to determine quantization ranges. Data-oblivious methods ([[turboquant]], QJL) avoid this entirely. Data-aware methods ([[spectralquant]]: 15s on one GPU) trade minimal calibration for meaningful quality gains by exploiting structural properties invisible to oblivious methods.

### Symmetric vs Asymmetric
- Symmetric: quantization range is [-max, max] — zero point = 0, one parameter (scale)
- Asymmetric: range is [min, max] — two parameters (scale + zero point); the extra storage overhead is the "memory overhead" problem addressed by PolarQuant/TurboQuant

### Block Quantization
Group elements into blocks; normalize each block independently. The scale/zero-point stored in FP32 per block adds overhead. At 1 byte per quantized value, storing even 1 FP32 scale per 8 values adds 0.5 bytes — >50% overhead.

## The Random Preconditioning Insight

[[polarquant]], [[turboquant]], and [[spectralquant]] all use rotation preconditioning before quantization. PolarQuant and TurboQuant use random rotation (oblivious); SpectralQuant uses calibrated eigenvector rotation. The calibrated version is strictly better: it aligns the coordinate axes with the actual signal/noise structure, enabling selective error correction that random rotation cannot.

## The Spectral Gap Insight (SpectralQuant)

KV cache key vectors exhibit d_eff ≈ 3–4% of head dimension as effective dimensionality — universally across model families. This means 96–97% of dimensions carry only noise. Applying error correction (QJL) uniformly to all dims, as TurboQuant does, worsens MSE on noise dims (variance added exceeds bias reduced). Selectively applying correction only to signal dims strictly improves quality *and* compression simultaneously.

## See Also

- [[kv-cache]] — KV cache quantization specifically
- [[polarquant]] — polar quantization for KV
- [[turboquant]] — vector quantization for KV
- [[spectralquant]] — calibrated spectral quantization for KV
- [[h2o]] — complementary eviction approach
- [[kv-cache-compression-comparison]] — methods compared
- [[sageattention]] — INT8 attention compute quantization; 2.1× FA2
- [[sageattention2]] — INT4/FP8 attention compute quantization; 3× FA2
- [[sageattention3]] — FP4 microscaling; 5× FA2 on Blackwell
- [[flash-attention]] — base kernel that SageAttention builds on
